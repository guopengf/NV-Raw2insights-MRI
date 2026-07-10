# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import random
import socket
import time
import warnings
from collections.abc import Mapping, Sequence as SequenceABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy
import torch
import torch.distributed as dist
import torchvision
from monai.apps.reconstruction.complex_utils import complex_abs, complex_conj_t, complex_mul_t
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.networks.utils import copy_model_state
from path_safety import assert_not_in_known_raw_data_path
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

# Default Hugging Face repo for nv_raw2insights MRI checkpoints. Override with env HF_NV_RAW2INSIGHTS_REPO.
HF_NV_RAW2INSIGHTS_REPO = os.environ.get("HF_NV_RAW2INSIGHTS_REPO", "nvidia/NV-Raw2insights-MRI")

# Map checkpoint filename -> (repo_id, filename_on_hub). Enables auto-download when local path missing.
CHECKPOINT_HF_REGISTRY = {
    "nv_raw2insights_mri_small": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_small.pt"),
    "nv_raw2insights_mri_base": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_base.pt"),
    "nv_raw2insights_mri_large": (HF_NV_RAW2INSIGHTS_REPO, "nv_raw2insights_mri_large.pt"),
}

MODALITY_MAPPING = {
    "aorta": "Aorta",
    "aorta_sag": "Aorta",
    "aorta_tra": "Aorta",
    "cine": "Cine",
    "cine_ax": "Cine",
    "cine_ot": "Cine",
    "cine_sax": "Cine",
    "cine_lax": "Cine",
    "cine_lax_2ch": "Cine",
    "cine_lax_3ch": "Cine",
    "cine_lax_4ch": "Cine",
    "cine_lax_r2ch": "Cine",
    "cine_lvot": "Cine",
    "cine_rvot": "Cine",
    "cine_lv": "Cine",
    "t1map": "Mapping",
    "t1map_lax_4ch": "Mapping",
    "t2map": "Mapping",
    "t2smap": "Mapping",
    "t1mappost": "Mapping",  # Assuming T1 map post-contrast is a type of mapping
    "t1rho": "T1rho",
    "lge": "LGE",
    "lge_sax": "LGE",
    "lge_lax": "LGE",
    "lge_lax_2ch": "LGE",
    "lge_lax_3ch": "LGE",
    "lge_lax_4ch": "LGE",
    "perfusion": "Perfusion",
    "t2w": "T2w",
    "blackblood": "BlackBlood",  # As per guidelines
    "flow2d": "Flow2d",  # As per guidelines
    "flow2d_inplane": "Flow2d",  # As per guidelines
    "flow2d_throughplane_m": "Flow2d",  # As per guidelines
    "flow2d_throughplane_d": "Flow2d",  # As per guidelines
    "t1w": "T1w",  # As per guidelines
    "t2w_lax_2ch": "T2w",
    "t2w_lax_4ch": "T2w",
    "tagging": "Tagging",
}

__all__ = [
    "adjust_learning_rate",
    "visualize",
    "save_args_to_file_json",
    "find_free_network_port",
    "is_ddp_enabled",
    "mini_dataloader",
    "sensitivity_map_reduce",
    "sensitivity_map_expand",
    "reshape_complex_to_channel_dim",
    "load_config",
    "validate_phase3_config",
    "resolve_checkpoint_path",
    "reshape_channel_complex_to_last_dim",
    "complex_normalize",
    "MultiEpochsDataLoader",
    "TimedDefaultCollate",
    "load_net",
    "save_checkpoint",
    "get_acs_image",
    "save_img4ranking",
    "ind2xy",
    "xy2ind",
    "reshape_batch_channel_to_channel_dim",
    "reshape_channel_to_batch_dim",
    "Lookahead",
    "windowed_input",
    "normalize_recon_mode",
    "normalize_ssim_spatial_dims",
    "is_slab_recon",
    "slab_num_slices",
    "windowed_input_x_slab",
    "select_mra_prior_for_microbatch",
    "select_mra_prior_slab_for_microbatch",
    "complex_zscore",
    "get_training_set",
    "report_nan_from_any_rank",
    "zero_grad_scalar_fast",
    "MODALITY_MAPPING",
]


def adjust_learning_rate(optimizer, epoch, args, is_resume_first_ten=False):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if is_resume_first_ten:
        lr = 0.0
    else:
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            if args.lr_schedule == "constant":
                lr = args.lr
            elif args.lr_schedule == "cosine":
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs))
                )
            else:
                raise NotImplementedError
    for param_group in optimizer.param_groups:
        group_lr = lr
        if "use_muon" in param_group and param_group["use_muon"]:
            group_lr = group_lr * args.muon_scale
        if "lr_scale" in param_group:
            group_lr = group_lr * param_group["lr_scale"]
        param_group["lr"] = group_lr
    return lr


def save_img4ranking(img4ranking, folder_path, file_path):
    folder_path = assert_not_in_known_raw_data_path(folder_path, what="img4ranking output folder")
    os.makedirs(folder_path, exist_ok=True)
    scipy.io.savemat(os.path.join(folder_path, file_path), {"img4ranking": img4ranking})


def visualize(input, output, target, epoch, writer):
    def make_grid(image, normalize=True):
        if normalize:
            image -= image.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
            image /= image.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values

        grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        return grid

    input, output, target = (
        torch.Tensor(input),
        torch.Tensor(output),
        torch.Tensor(target),
    )

    # See PR: https://github.com/Project-MONAI/MONAI/pull/8407
    output_k = fftn_centered(output, spatial_dims=2, is_complex=(output.shape[-1] == 2))
    input_k = fftn_centered(input, spatial_dims=2, is_complex=(input.shape[-1] == 2))
    target_k = fftn_centered(target, spatial_dims=2, is_complex=(target.shape[-1] == 2))

    output = complex_abs(output).unsqueeze(1) if output.shape[-1] == 2 else output.unsqueeze(1)
    target = complex_abs(target).unsqueeze(1) if target.shape[-1] == 2 else target.unsqueeze(1)
    input = complex_abs(input).unsqueeze(1) if input.shape[-1] == 2 else input.unsqueeze(1)

    error = make_grid(torch.abs(target - output))
    input = make_grid(input)
    target = make_grid(target)
    output = make_grid(output)

    input_k_mag = torch.log10(complex_abs(input_k) + 1e-9).unsqueeze(1)
    target_k_mag = torch.log10(complex_abs(target_k) + 1e-9).unsqueeze(1)
    output_k_mag = torch.log10(complex_abs(output_k) + 1e-9).unsqueeze(1)

    error_k = make_grid(torch.abs(target_k_mag - output_k_mag))
    input_k = make_grid(input_k_mag)
    target_k = make_grid(target_k_mag)
    output_k = make_grid(output_k_mag)

    canvas = torch.cat((input, target, output, error, input_k, target_k, output_k, error_k), -1)

    writer.add_image("Result", canvas, epoch)
    return [input, target, output, error, input_k, target_k, output_k, error_k]


def save_args_to_file_json(args, filename):
    filename = assert_not_in_known_raw_data_path(filename, what="config output file")
    def to_jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "to_dict"):
            return to_jsonable(value.to_dict())
        if isinstance(value, dict):
            return {k: to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        return value

    args_dict = {key: to_jsonable(value) for key, value in vars(args).items() if not key.startswith("_")}

    with open(filename, "w") as f:
        json.dump(args_dict, f, indent=4)


class Config:
    def __init__(self, d):
        setattr(self, "_explicit_keys", set(d.keys()))
        setattr(self, "data_path_train", None)
        setattr(self, "data_path_val", None)
        setattr(self, "pretrained_csm", None)
        setattr(self, "pretrained_recon", None)
        setattr(self, "pp_z_score_norm", False)
        setattr(self, "fixed_mask_types", None)
        setattr(self, "uniform_input_kspace", False)
        setattr(self, "val_interval", 4)
        setattr(self, "num_samples_per_case", 16)
        setattr(self, "data_aug", True)
        setattr(self, "use_multi_epochs_train_loader", False)
        setattr(self, "resume_rng_state", False)
        setattr(self, "do_mapping_shuffle", False)
        setattr(self, "do_center_crop", True)
        setattr(self, "seed", None)
        setattr(self, "lookahead", False)
        setattr(self, "muon", False)
        setattr(self, "muon_scale", 5)
        setattr(self, "enable_onelogger", False)
        setattr(self, "flow", False)
        setattr(self, "balance_data", False)
        setattr(self, "enable_cas_skips", True)
        setattr(self, "finetune_ms", False)
        setattr(self, "adaptive_batch_size", False)
        setattr(self, "constant_input_flow", False)
        setattr(self, "amp", True)
        setattr(self, "acs_lines", 20)
        setattr(self, "pp_norm", True)
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

        if hasattr(self, "mask_types") and not hasattr(self, "train_mask_types"):
            setattr(self, "train_mask_types", self.mask_types)
            setattr(self, "val_mask_types", self.mask_types)
            delattr(self, "mask_types")

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in vars(self).items()
            if not k.startswith("_")
        }


def load_config(file_path: Path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A Python dictionary representing the JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            config_dict = json.load(file)
            config = Config(config_dict)
            config.data_path_train = (
                [config.data_path_train] if isinstance(config.data_path_train, str) else config.data_path_train
            )
            config.data_path_val = (
                [config.data_path_val] if isinstance(config.data_path_val, str) else config.data_path_val
            )
            validate_phase3_config(config)
            return config
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file: {file_path}: {e}")
        return None


def _config_key_set(obj) -> set[str]:
    if obj is None:
        return set()
    if isinstance(obj, dict):
        return set(obj.keys())
    if hasattr(obj, "_explicit_keys"):
        return set(getattr(obj, "_explicit_keys"))
    if hasattr(obj, "to_dict"):
        return set(obj.to_dict().keys())
    if hasattr(obj, "__dict__"):
        return set(vars(obj).keys())
    return set()


def _warn_unknown_config_keys(obj, path: str, allowed: set[str]) -> None:
    unknown = sorted(_config_key_set(obj) - allowed)
    if unknown:
        warnings.warn(
            f"Unknown config key(s) under {path}: {unknown}. "
            "They will be ignored unless another code path explicitly reads them.",
            RuntimeWarning,
        )


def _get_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_attr(obj, name: str, value) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def normalize_ssim_spatial_dims(value, recon_mode: str):
    """Normalize phase3.loss.ssim_spatial_dims while preserving auto semantics."""
    if value is None:
        return "auto"
    if isinstance(value, str):
        value_l = value.lower()
        if value_l == "auto":
            return "auto"
        if value_l in {"2", "3"}:
            return int(value_l)
        raise ValueError(
            f"Unsupported phase3.loss.ssim_spatial_dims={value!r}. Use 'auto', 2, or 3. "
            "auto means slice/2p5d -> 2D SSIM and slab/3d -> 3D SSIM."
        )
    try:
        value_i = int(value)
    except (TypeError, ValueError):
        value_i = None
    if value_i in {2, 3}:
        return value_i
    raise ValueError(
        f"Unsupported phase3.loss.ssim_spatial_dims={value!r}. Use 'auto', 2, or 3. "
        "auto means slice/2p5d -> 2D SSIM and slab/3d -> 3D SSIM."
    )


def validate_phase3_config(config) -> None:
    """Validate and normalize Phase3 config fields that must not be silent no-ops."""
    phase3 = _get_attr(config, "phase3", None)
    if phase3 is None:
        return

    _warn_unknown_config_keys(
        phase3,
        "phase3",
        {
            "enable_vaa",
            "recon_mode",
            "num_slices",
            "mra",
            "vaa",
            "mask",
            "freeze",
            "gamma",
            "loss",
            "inference",
        },
    )

    raw_recon_mode = _get_attr(phase3, "recon_mode", "slice")
    recon_mode = normalize_recon_mode(raw_recon_mode)
    if str(raw_recon_mode).lower() != recon_mode:
        warnings.warn(
            f"phase3.recon_mode={raw_recon_mode!r} normalized to {recon_mode!r}. "
            "Accepted aliases: slice/2p5d/2.5d/2d and slab/3d.",
            RuntimeWarning,
        )
    _set_attr(phase3, "recon_mode", recon_mode)
    if recon_mode == "slab":
        num_slices = int(_get_attr(phase3, "num_slices", 3))
        if num_slices <= 0 or num_slices % 2 == 0:
            raise ValueError(f"phase3.num_slices must be a positive odd integer for slab mode, got {num_slices}")

    mra = _get_attr(phase3, "mra", None)
    if mra is not None:
        _warn_unknown_config_keys(
            mra,
            "phase3.mra",
            {
                "source",
                "method",
                "use_mip",
                "projection",
                "projection_axis",
                "use_cache",
                "cache_dir",
                "device",
                "vessel_map",
                "sense",
            },
        )
        source = str(_get_attr(mra, "source", "gt")).lower()
        if source not in {"gt", "zf", "sense", "phase2"}:
            raise ValueError("phase3.mra.source must be one of {'gt', 'zf', 'sense', 'phase2'}")
        _set_attr(mra, "source", source)
        if source == "phase2":
            warnings.warn(
                "phase3.mra.source='phase2' is accepted as an interface placeholder, "
                "but Phase2 reconstruction -> PC-MRA cache is not implemented in this branch.",
                RuntimeWarning,
            )
        method = str(_get_attr(mra, "method", "pcmra")).lower()
        if method not in {"pcmra"}:
            raise ValueError("phase3.mra.method currently supports only 'pcmra'")
        _set_attr(mra, "method", method)
        _warn_unknown_config_keys(
            _get_attr(mra, "vessel_map", None),
            "phase3.mra.vessel_map",
            {"description", "binary", "lower_percentile", "upper_percentile", "smooth_sigma", "threshold"},
        )
        _warn_unknown_config_keys(
            _get_attr(mra, "sense", None),
            "phase3.mra.sense",
            {"description", "niter", "lam"},
        )

    vaa = _get_attr(phase3, "vaa", None)
    if vaa is not None:
        _warn_unknown_config_keys(
            vaa,
            "phase3.vaa",
            {"prior_source", "attention", "locations", "reduction", "num_heads", "attention_stride", "use_mask_bias"},
        )
        attention = str(_get_attr(vaa, "attention", "legacy")).lower()
        if attention not in {"legacy", "gate", "qkv"}:
            raise ValueError("phase3.vaa.attention must be 'legacy', 'gate', or 'qkv'")
        _set_attr(vaa, "attention", attention)
        prior_source = str(_get_attr(vaa, "prior_source", "mra")).lower()
        if prior_source not in {"mra", "mask"}:
            raise ValueError("phase3.vaa.prior_source must be 'mra' or 'mask'")
        _set_attr(vaa, "prior_source", prior_source)
        if not bool(_get_attr(phase3, "enable_vaa", False)) and "attention" in _config_key_set(vaa):
            warnings.warn(
                "phase3.vaa.attention is configured but phase3.enable_vaa=false; "
                "VAA will be hard-bypassed and attention will not run.",
                RuntimeWarning,
            )

    mask = _get_attr(phase3, "mask", None)
    if mask is not None:
        _warn_unknown_config_keys(mask, "phase3.mask", {"field", "keys", "axis_order", "binary", "threshold"})
        axis_order = str(_get_attr(mask, "axis_order", "zyx")).lower()
        if sorted(axis_order) != ["x", "y", "z"]:
            raise ValueError(f"phase3.mask.axis_order must be a permutation of zyx, got {axis_order!r}")
        _set_attr(mask, "axis_order", axis_order)

    freeze = _get_attr(phase3, "freeze", None)
    if freeze is not None:
        _warn_unknown_config_keys(freeze, "phase3.freeze", {"backbone", "vaa"})

    gamma = _get_attr(phase3, "gamma", None)
    if gamma is not None:
        _warn_unknown_config_keys(gamma, "phase3.gamma", {"init", "trainable", "mode", "use_sigmoid"})
        if "use_sigmoid" in _config_key_set(gamma) and "mode" not in _config_key_set(gamma):
            use_sigmoid = bool(_get_attr(gamma, "use_sigmoid", True))
            gamma_mode = "shifted_sigmoid" if use_sigmoid else "direct_clamp"
            _set_attr(gamma, "mode", gamma_mode)
            warnings.warn(
                f"phase3.gamma.use_sigmoid is deprecated; mapped to phase3.gamma.mode={gamma_mode!r}.",
                RuntimeWarning,
            )
        elif "use_sigmoid" in _config_key_set(gamma):
            warnings.warn(
                "phase3.gamma.use_sigmoid is deprecated and ignored because phase3.gamma.mode is set.",
                RuntimeWarning,
            )
        gamma_mode = str(_get_attr(gamma, "mode", "shifted_sigmoid")).lower()
        if gamma_mode not in {"direct_clamp", "shifted_sigmoid"}:
            raise ValueError("phase3.gamma.mode must be 'direct_clamp' or 'shifted_sigmoid'")
        _set_attr(gamma, "mode", gamma_mode)

    loss = _get_attr(phase3, "loss", None)
    if loss is not None:
        _warn_unknown_config_keys(
            loss,
            "phase3.loss",
            {
                "use_phase",
                "use_vascular",
                "use_ssim_zy",
                "use_ssim_xy",
                "ssim_spatial_dims",
                "ssim_win_size",
                "phase",
                "vascular",
                "weights",
            },
        )
        spatial_dims = normalize_ssim_spatial_dims(_get_attr(loss, "ssim_spatial_dims", "auto"), recon_mode)
        _set_attr(loss, "ssim_spatial_dims", spatial_dims)
        phase_loss = _get_attr(loss, "phase", None)
        vascular_loss = _get_attr(loss, "vascular", None)
        _warn_unknown_config_keys(phase_loss, "phase3.loss.phase", {"method", "weight", "eps"})
        _warn_unknown_config_keys(vascular_loss, "phase3.loss.vascular", {"method", "weight", "normalize_by_mask"})
        _warn_unknown_config_keys(
            _get_attr(loss, "weights", None),
            "phase3.loss.weights",
            {"magnitude", "phase", "ssim_zy", "vascular", "ssim_xy"},
        )
        flowvn_methods = {
            "flowvn_complex_l1",
            "complex_l1",
            "flowvn_unit_complex_l1",
            "unit_complex_l1",
            "phase_l1",
            "mra_masked_complex_l1",
            "mra_masked_flowvn_complex_l1",
            "mra_masked_phase_l1",
            "mra_masked_flowvn_phase_l1",
        }
        if phase_loss is not None:
            phase_method = str(_get_attr(phase_loss, "method", "flowvn_complex_l1")).lower()
            if phase_method not in flowvn_methods:
                raise ValueError(f"Unsupported phase3.loss.phase.method={phase_method!r}")
            _set_attr(phase_loss, "method", phase_method)
        if vascular_loss is not None:
            vascular_method = str(_get_attr(vascular_loss, "method", "mra_masked_phase_l1")).lower()
            if vascular_method not in flowvn_methods:
                raise ValueError(f"Unsupported phase3.loss.vascular.method={vascular_method!r}")
            _set_attr(vascular_loss, "method", vascular_method)

    inference = _get_attr(phase3, "inference", None)
    if inference is not None:
        _warn_unknown_config_keys(inference, "phase3.inference", {"output_merge"})
        output_merge = str(_get_attr(inference, "output_merge", "center")).lower()
        if output_merge not in {"center"}:
            raise ValueError("phase3.inference.output_merge currently supports only 'center'")
        _set_attr(inference, "output_merge", output_merge)


def resolve_checkpoint_path(model_variant: str, model_ckpt: str | Path | None = None) -> Path:
    """
    Resolve checkpoint path: if the file exists, return it; otherwise try to download
    from Hugging Face Hub. Uses CHECKPOINT_HF_REGISTRY and
    HF_NV_RAW2INSIGHTS_REPO (or env HF_NV_RAW2INSIGHTS_REPO).

    Args:
        model_variant: Model variant (e.g. "nv_raw2insights_mri_small", "nv_raw2insights_mri_base",
                "nv_raw2insights_mri_large").
        model_ckpt: Local path (e.g. pretrained_ckpt/nv_raw2insights_mri_small_modified.pt). Default is None.
    Returns:
        Path to the checkpoint file (existing or downloaded).
    """
    if model_ckpt is not None:
        path = Path(model_ckpt)
        if path.exists():
            return path

    if model_variant not in CHECKPOINT_HF_REGISTRY:
        raise ValueError(f"Model variant {model_variant} not found in CHECKPOINT_HF_REGISTRY")
    repo_id, filename = CHECKPOINT_HF_REGISTRY[model_variant]
    try:
        from huggingface_hub import hf_hub_download

        _ = hf_hub_download(repo_id=repo_id, filename="config.json")
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded checkpoint from Hugging Face ({repo_id}): {filename} -> {local_path}")
        return Path(local_path)
    except BaseException as e:
        raise ValueError(f"Could not download checkpoint from Hugging Face: {e}") from e


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def is_ddp_enabled():
    # Check for PyTorch DDP environment variables that torchrun sets
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # If world_size > 1 and rank is set, we're in a DDP environment
    return world_size > 1 and rank >= 0 and local_rank >= 0


def mini_dataloader(
    data,
    batch_size,
    drop_last=False,
    pad_last=True,
    shuffle=False,
    max_num_batches=None,
    infinite=False,
):
    if len(data) == 0:
        raise ValueError("`data` must be non-empty.")

    indices = list(range(len(data)))
    batch_count = 0

    while True:  # epoch loop (repeats if `infinite=True`)
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Handle short last batch
            if len(batch_indices) < batch_size:
                if drop_last:
                    continue  # skip the short last batch
                if pad_last:
                    # pad with random samples (with replacement) to keep batch size
                    batch_indices.extend(random.choices(indices, k=batch_size - len(batch_indices)))

            yield [data[j] for j in batch_indices], i  # keep original return shape

            batch_count += 1
            if max_num_batches is not None and batch_count >= max_num_batches:
                return  # stop after N batches overall (across epochs)

        # one epoch done
        if not infinite:
            break  # stop after a single pass if not infinite


def sensitivity_map_reduce(img: torch.Tensor, sens_maps: torch.Tensor, k: int = 1, mode: str = "rand") -> torch.Tensor:
    if k != 1:
        res = complex_mul_t(img, complex_conj_t(sens_maps))
        coils = list(range(sens_maps.shape[1]))
        if mode == "rand":
            random.shuffle(coils)
        return torch.stack([res[:, i::k].sum(dim=1) for i in range(k)], dim=1)
    else:
        return complex_mul_t(img, complex_conj_t(sens_maps)).sum(dim=-4, keepdim=True)


def sensitivity_map_expand(img: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul_t(img.sum(dim=-4, keepdim=True), sens_maps)


def reshape_complex_to_channel_dim(x: torch.Tensor, temporal=False) -> torch.Tensor:
    if x.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]}.")

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    elif len(x.shape) == 6:
        if not temporal:  # this is 3D
            b, c, h, w, d, two = x.shape
            return x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b, 2 * c, h, w, d)
        else:
            b, t, c, h, w, two = x.shape
            return x.permute(0, 1, 5, 2, 3, 4).contiguous().view(b, t, 2 * c, h, w)
    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_channel_complex_to_last_dim(x: torch.Tensor, temporal=False) -> torch.Tensor:

    if len(x.shape) == 4:  # this is 2D
        if x.shape[1] % 2 != 0:
            raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
        b, c2, h, w = x.shape  # c2 means c*2
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    elif len(x.shape) == 5:  # this is 3D
        if not temporal:  # this is 3D
            if x.shape[1] % 2 != 0:
                raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
            b, c2, h, w, d = x.shape  # c2 means c*2
            c = c2 // 2
            return x.view(b, 2, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        else:
            if x.shape[2] % 2 != 0:
                raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")
            b, t, c2, h, w = x.shape  # c2 means c*2
            c = c2 // 2
            return x.view(b, t, 2, c, h, w).permute(0, 1, 3, 4, 5, 2)
    else:
        raise ValueError(f"only 2D (B,C*2,H,W) and 3D (B,C*2,H,W,D) data are supported but x has shape {x.shape}")


def complex_normalize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, cpx = x.shape
        assert cpx == 2, "last dim must be 2, but cpx is {cpx}".format(cpx=cpx)
        x = x.contiguous().view(b, c * h * w, 2)
        mean = x.mean(dim=1).view(b, 1, 1, 1, 2).expand(b, c, 1, 1, 2).contiguous()
        std = x.std(dim=1, unbiased=False).view(b, 1, 1, 1, 2).expand(b, c, 1, 1, 2).contiguous()
        x = x.view(b, c, h, w, 2)
        return (x - mean) / std, mean, std
    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, cpx = x.shape
        assert cpx == 2, "last dim must be 2, but cpx is {cpx}".format(cpx=cpx)
        x = x.contiguous().view(b, c * h * w * d, 2)
        mean = x.mean(dim=1).view(b, 1, 1, 1, 1, 2).expand(b, c, 1, 1, 1, 2).contiguous()
        std = x.std(dim=1, unbiased=False).view(b, 1, 1, 1, 1, 2).expand(b, c, 1, 1, 1, 2).contiguous()
        x = x.view(b, c, h, w, d, 2)
        return (x - mean) / std, mean, std
    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def complex_zscore(real_imag: torch.Tensor, dim=None, unbiased: bool = False):
    """
    Z‑score normalize a real‑imag tensor along the given spatial dims,
    preserving phase exactly.

    Params:
      real_imag: Tensor of shape (..., 2) in last dim = (real, imag)
      dim:      tuple of dims to reduce over (e.g. spatial dims). If None,
                will reduce over all dims except the last.
      unbiased: if True, uses Bessel’s correction in variance.
    Returns:
      normed:   same shape as input, real‑imag normalized
      mean:     per‑reduction complex mean (real‑imag stacked in last dim)
      std:      per‑reduction real std = sqrt(E[|x-μ|²])
    """
    # view as complex
    x_c = torch.view_as_complex(real_imag)

    # figure out which dims to reduce: exclude last dim and batch/channel dims
    if dim is None:
        dim = tuple(range(x_c.ndim - 1))  # everything except the imaginary axis
    mean = x_c.mean(dim=dim, keepdim=True)  # complex
    # variance = E[ |x - μ|² ]
    var = ((x_c - mean).abs() ** 2).mean(dim=dim, keepdim=True)
    if unbiased:
        # apply Bessel’s correction: var *= N/(N-1)
        n = 1
        for d in dim:
            n *= x_c.size(d)
        if n > 1:
            var = var * (n / (n - 1))
    std = torch.sqrt(var)  # real
    # z‑score (complex shift & real‑scale)
    z_c = (x_c - mean) / std

    # convert everything back to real‑imag pairs
    z_ri = torch.view_as_real(z_c)
    mean_ri = torch.view_as_real(mean)
    std_r = std.real.unsqueeze(-1)  # keep it real
    return z_ri, mean_ri, std_r


# https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778
def _tensor_payload_summary(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size(), 1
    if isinstance(value, np.ndarray):
        return value.nbytes, 1
    if isinstance(value, Mapping):
        total_bytes = 0
        tensor_count = 0
        for item in value.values():
            item_bytes, item_count = _tensor_payload_summary(item)
            total_bytes += item_bytes
            tensor_count += item_count
        return total_bytes, tensor_count
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        total_bytes = 0
        tensor_count = 0
        for item in value:
            item_bytes, item_count = _tensor_payload_summary(item)
            total_bytes += item_bytes
            tensor_count += item_count
        return total_bytes, tensor_count
    return 0, 0


def _shared_memory_snapshot():
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return {"shm_total_bytes": -1, "shm_free_bytes": -1, "shm_used_bytes": -1}
    total_bytes = int(stat.f_blocks * stat.f_frsize)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    return {
        "shm_total_bytes": total_bytes,
        "shm_free_bytes": free_bytes,
        "shm_used_bytes": total_bytes - free_bytes,
    }


def _singleton_view_collate(value):
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value).unsqueeze(0)
    if isinstance(value, np.generic):
        return torch.as_tensor(value).reshape(1)
    if isinstance(value, Mapping):
        return {name: _singleton_view_collate(item) for name, item in value.items()}
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_singleton_view_collate(item) for item in value]
    if isinstance(value, (int, float, bool)):
        return torch.tensor([value])
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    return [value]


class TimedDefaultCollate:
    """Measure default collation and the worker-to-consumer handoff boundary."""

    def __init__(self, enabled=True, track_shared_memory=True, singleton_view=False):
        self.enabled = enabled
        self.track_shared_memory = track_shared_memory
        self.singleton_view = singleton_view

    def __call__(self, batch):
        if not self.enabled:
            if self.singleton_view:
                if len(batch) != 1:
                    raise ValueError("singleton_view collate requires DataLoader batch_size=1")
                return _singleton_view_collate(batch[0])
            return torch.utils.data.default_collate(batch)

        collate_start_ns = time.monotonic_ns()
        input_payload_bytes, input_tensor_count = _tensor_payload_summary(batch)
        shm_before = _shared_memory_snapshot() if self.track_shared_memory else {}
        if self.singleton_view:
            if len(batch) != 1:
                raise ValueError("singleton_view collate requires DataLoader batch_size=1")
            collated = _singleton_view_collate(batch[0])
        else:
            collated = torch.utils.data.default_collate(batch)
        default_collate_end_ns = time.monotonic_ns()
        output_payload_bytes, output_tensor_count = _tensor_payload_summary(collated)
        shm_after = _shared_memory_snapshot() if self.track_shared_memory else {}
        collate_end_ns = time.monotonic_ns()

        meta = collated.get("kspace_meta_dict") if isinstance(collated, Mapping) else None
        if isinstance(meta, dict):
            meta["post_worker_timing"] = {
                "collate_start_ns": collate_start_ns,
                "default_collate_end_ns": default_collate_end_ns,
                "collate_end_ns": collate_end_ns,
                "default_collate_ms": (default_collate_end_ns - collate_start_ns) / 1.0e6,
                "collate_ms": (collate_end_ns - collate_start_ns) / 1.0e6,
                "input_payload_bytes": int(input_payload_bytes),
                "output_payload_bytes": int(output_payload_bytes),
                "input_tensor_count": int(input_tensor_count),
                "output_tensor_count": int(output_tensor_count),
                "collate_mode": "singleton_view" if self.singleton_view else "default",
                **{f"{name}_before": value for name, value in shm_before.items()},
                **{f"{name}_after": value for name, value in shm_after.items()},
            }
        return collated


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def load_shape_compatible_state_dict(module: torch.nn.Module, checkpoint_state_dict: dict):
    """Load checkpoint tensors only when both key and shape match current module."""
    current_state = module.state_dict()
    current_keys = set(current_state.keys())
    new_state = dict(current_state)
    loaded_keys = []
    skipped_shape_keys = []
    unexpected_keys = []

    for ckpt_key, ckpt_value in checkpoint_state_dict.items():
        key = ckpt_key
        if key not in current_keys and key.startswith("module."):
            key = key[len("module.") :]
        elif key not in current_keys and f"module.{key}" in current_keys:
            key = f"module.{key}"

        if key not in current_keys:
            unexpected_keys.append(ckpt_key)
            continue
        if tuple(ckpt_value.shape) == tuple(current_state[key].shape):
            new_state[key] = ckpt_value
            loaded_keys.append(key)
        else:
            skipped_shape_keys.append((key, tuple(ckpt_value.shape), tuple(current_state[key].shape)))

    module.load_state_dict(new_state, strict=True)
    loaded_set = set(loaded_keys)
    unchanged_keys = [key for key in current_state.keys() if key not in loaded_set]
    return loaded_keys, unchanged_keys, skipped_shape_keys, unexpected_keys


def load_net(
    net,
    resume_training_ckpt,
    device,
    is_ddp=False,
    find_unused_parameters=False,
    resume_rng_state=False,
    prepare_model_for_ddp=None,
):
    """
    Load the Net model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        is_ddp (bool): Whether to use distributed data parallel.
        find_unused_parameters (bool): Whether to find unused parameters.
        resume_rng_state (bool): Whether to resume the random number generator state.
        prepare_model_for_ddp (Callable, optional): Hook called after loading
            checkpoint weights and before wrapping the model with DDP.

    Returns:
        torch.nn.Module: Loaded Net model.
    """

    start_epoch = 0
    start_global_step = 0
    optimizer_state_dict = None
    scheduler_state_dict = None
    scaler_state_dict = None
    best_metric = -1
    best_metric_epoch = -1
    wandb_run_id = None

    if not os.path.exists(resume_training_ckpt):
        print("Training from scratch or pretrained model.")
    else:
        optimizer_state_dict = None
        scheduler_state_dict = None
        scaler_state_dict = None
        start_epoch = 0
        start_global_step = 0

        checkpoint_net = torch.load(resume_training_ckpt, map_location=device, weights_only=False)
        print(f"resume training net from {resume_training_ckpt}.")
        ignore_schduler_opt_state = False
        print(f"resume training net, ignore_schduler_opt_state: {ignore_schduler_opt_state}")

        updated_keys, unchanged_keys, skipped_shape_keys, unexpected_keys = load_shape_compatible_state_dict(
            net, checkpoint_net["net_state_dict"]
        )

        print(f"net updated_keys: {len(updated_keys)}")
        print(f"net unchanged_keys: {len(unchanged_keys)}")
        if skipped_shape_keys:
            preview = ", ".join(
                f"{key}: ckpt{ckpt_shape}->model{model_shape}"
                for key, ckpt_shape, model_shape in skipped_shape_keys[:8]
            )
            suffix = "..." if len(skipped_shape_keys) > 8 else ""
            print(f"net skipped_shape_keys: {len(skipped_shape_keys)} ({preview}{suffix})")
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:8])
            suffix = "..." if len(unexpected_keys) > 8 else ""
            print(f"net unexpected checkpoint keys: {len(unexpected_keys)} ({preview}{suffix})")

        if not ignore_schduler_opt_state:
            try:
                if checkpoint_net["epoch_finished"]:
                    start_epoch = checkpoint_net["epoch"]
                else:
                    start_epoch = checkpoint_net["epoch"] - 1
                start_global_step = checkpoint_net["global_step"]
            except BaseException:
                start_epoch = 0
                start_global_step = 0

            if "optimizer_state_dict" in checkpoint_net:
                print("load optimizer_state_dict from ckpt.")
                optimizer_state_dict = checkpoint_net["optimizer_state_dict"]
            if "scaler_state_dict" in checkpoint_net:
                print("load scaler_state_dict from ckpt.")
                scaler_state_dict = checkpoint_net["scaler_state_dict"]
            if "scheduler_state_dict" in checkpoint_net:
                print("load scheduler_state_dict from ckpt.")
                scheduler_state_dict = checkpoint_net["scheduler_state_dict"]
            if "best_metric" in checkpoint_net:
                best_metric = checkpoint_net["best_metric"]
            if "best_metric_epoch" in checkpoint_net:
                best_metric_epoch = checkpoint_net["best_metric_epoch"]
            if "wandb_run_id" in checkpoint_net:
                wandb_run_id = checkpoint_net["wandb_run_id"]
            if resume_rng_state:
                if "python_rng_state" in checkpoint_net:
                    random.setstate(checkpoint_net["python_rng_state"])
                if "numpy_rng_state" in checkpoint_net:
                    np.random.set_state(checkpoint_net["numpy_rng_state"])
                if "torch_rng_state" in checkpoint_net:
                    torch.random.set_rng_state(checkpoint_net["torch_rng_state"].cpu())
                if "cuda_rng_state" in checkpoint_net:
                    torch.cuda.set_rng_state(checkpoint_net["cuda_rng_state"].cpu())

    if prepare_model_for_ddp is not None:
        prepare_model_for_ddp(net)

    if is_ddp and torch.cuda.device_count() > 1:
        # Get device index from device object
        device_index = device.index if isinstance(device, torch.device) else 0

        # Ensure all processes are synchronized before converting to DDP
        dist.barrier(device_ids=[device_index])

        # Convert to SyncBatchNorm first
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # Ensure all processes are synchronized before DDP initialization
        dist.barrier(device_ids=[device_index])

        # Initialize DDP
        net = DistributedDataParallel(
            net,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

        # Final synchronization after DDP initialization
        dist.barrier(device_ids=[device_index])
    elif torch.cuda.is_available():
        net = net.to(device)

    return (
        net,
        optimizer_state_dict,
        scheduler_state_dict,
        scaler_state_dict,
        start_epoch,
        start_global_step,
        best_metric,
        best_metric_epoch,
        wandb_run_id,
    )


def save_checkpoint(
    epoch: int,
    global_step: int,
    net: torch.nn.Module,
    optimizer,
    lr_scheduler,
    scaler,
    ckpt_folder: str,
    best_metric,
    best_metric_epoch,
    model_filename: str,
    epoch_finished: bool = True,
    is_ddp: bool = False,
    wandb_run_id: str = None,
    python_rng_state=None,
    numpy_rng_state=None,
    torch_rng_state=None,
    cuda_rng_state=None,
) -> dict:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): net model.
        ckpt_folder (str): Checkpoint folder path.
        model_filename (str): model filename.
        epoch_finished (bool): epoch finished
    """
    checkpoint_started = time.perf_counter()
    ckpt_path = assert_not_in_known_raw_data_path(f"{ckpt_folder}/{model_filename}", what="checkpoint output file")
    state_prepare_started = time.perf_counter()
    net_state_dict = net.module.state_dict() if is_ddp else net.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scaler_state_dict = scaler.state_dict()
    if lr_scheduler is not None:
        scheduler_state_dict = lr_scheduler.state_dict()
    else:
        scheduler_state_dict = None
    state_prepare_s = time.perf_counter() - state_prepare_started
    torch_save_started = time.perf_counter()
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step + 1,
            "net_state_dict": net_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "scaler_state_dict": scaler_state_dict,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "epoch_finished": epoch_finished,
            "wandb_run_id": wandb_run_id,
            "python_rng_state": python_rng_state,
            "numpy_rng_state": numpy_rng_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_state": cuda_rng_state,
        },
        ckpt_path,
    )
    torch_save_s = time.perf_counter() - torch_save_started
    total_s = time.perf_counter() - checkpoint_started
    size_bytes = os.path.getsize(ckpt_path)
    print(
        f"Save ckpt to {ckpt_path}. state_prepare={state_prepare_s:.2f}s "
        f"torch_save={torch_save_s:.2f}s total={total_s:.2f}s "
        f"size={size_bytes / (1024**3):.2f}GiB"
    )
    return {
        "path": str(ckpt_path),
        "state_prepare_s": state_prepare_s,
        "torch_save_s": torch_save_s,
        "total_s": total_s,
        "size_bytes": size_bytes,
    }


def get_acs_region(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Given a 2D binary mask,
    finds the maximum acs region containing only positive (nonzero)
    elements that includes the center of the mask.

    Returns:
        A tuple (left, right, top, bottom) where:
          - left and right are the minimum and maximum x indices,
          - top and bottom are the minimum and maximum y indices.
    """
    h, w = mask.shape[-3], mask.shape[-2]
    # Determine the center coordinates.
    cy, cx = h // 2, w // 2

    # Ensure the center is positive.
    if not torch.all(mask[..., cy, cx, :]):
        raise ValueError("The center of the mask is not positive.")

    # Initialize the bounding box to the center pixel.
    left, right = cx, cx
    top, bottom = cy, cy

    # Try to expand the box outward while the newly included row/column is all positive.
    expanded = True
    while expanded:
        expanded = False

        # Attempt to expand to the left.
        if left > 0 and torch.all(mask[..., top : bottom + 1, left - 1, :] > 0):
            left -= 1
            expanded = True

        # Attempt to expand to the right.
        if right < w - 1 and torch.all(mask[..., top : bottom + 1, right + 1, :] > 0):
            right += 1
            expanded = True

        # Attempt to expand upward.
        if top > 0 and torch.all(mask[..., top - 1, left : right + 1, :] > 0):
            top -= 1
            expanded = True

        # Attempt to expand downward.
        if bottom < h - 1 and torch.all(mask[..., bottom + 1, left : right + 1, :] > 0):
            bottom += 1
            expanded = True

    return top, bottom, left, right


def get_acs_image(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    top, bottom, left, right = get_acs_region(mask)
    x_k = fftn_centered(x, spatial_dims=2, is_complex=True)
    num_low_freqs_x = right - left  # size of the fully-sampled center
    num_low_freqs_y = bottom - top

    # take out the fully-sampled region and set the rest of the data to zero
    x_k_acs = torch.zeros_like(x_k)
    start_x = (mask.shape[-2] - num_low_freqs_x + 1) // 2  # this marks the start of center extraction
    start_y = (mask.shape[-3] - num_low_freqs_y + 1) // 2

    x_k_acs[..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :] = x_k[
        ..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :
    ]
    x_acs = ifftn_centered(x_k_acs, spatial_dims=2, is_complex=True)
    return x_acs


def ind2xy(idx, num_frames, num_slices):
    """
    Convert a flattened frame×slice index to (slice_idx, frame_idx).
    """
    frame_idx = idx // num_slices
    slice_idx = idx % num_slices
    return slice_idx, frame_idx


def xy2ind(slice_idx, frame_idx, num_frames, num_slices):
    """
    Convert (slice_idx, frame_idx) back to a flattened index.
    """
    return frame_idx * num_slices + slice_idx


def windowed_input(input, micro_b, final_shape, num_frames, slice_window_single_frame=False):
    """
    Given:
      input       – a Tensor of shape [N, …] where N == total_frames * total_slices
      micro_b     – list or 1D Tensor of length B with flattened frame×slice indices
      final_shape – the shape of the volume, where
                     final_shape[-5] == total number of frames,
                     final_shape[-4] == total number of slices
      num_frames  – desired window size (can be odd or even)
    Returns:
      inp – Tensor of shape [B, num_frames, …],
            where for each b, we’ve gathered around the center frame (or slice, if single-frame input).
      window_idx – LongTensor of shape [B, num_frames] containing the gathered indices
    """
    total_frames = final_shape[-5]
    total_slices = final_shape[-4]
    half = num_frames // 2
    # symmetric offsets: e.g. num_frames=4 → offsets = [-2,-1,0,1]
    offsets = range(-half, num_frames - half)

    single_frame = total_frames == 1
    window_idx = []

    for idx in micro_b:
        slice_i, frame_i = ind2xy(idx, total_frames, total_slices)

        if single_frame and slice_window_single_frame:
            # sliding window in the slice dimension
            centers = slice_i
            neighbors = [(centers + off) % total_slices for off in offsets]
            idxs = [xy2ind(s, frame_i, total_frames, total_slices) for s in neighbors]
        else:
            # sliding window in the frame dimension (original behavior)
            centers = frame_i
            neighbors = [(centers + off) % total_frames for off in offsets]
            idxs = [xy2ind(slice_i, f, total_frames, total_slices) for f in neighbors]

        window_idx.append(idxs)

    # make a LongTensor [B, num_frames]
    window_idx = torch.tensor(window_idx, dtype=torch.long, device=input.device)

    # gather: shape [B, num_frames, …rest of input.shape…]
    inp = torch.Tensor(input[window_idx])
    return inp, window_idx


def normalize_recon_mode(args_or_mode) -> str:
    """Normalize Phase3 reconstruction mode aliases.

    Supported public names:
      - slice / 2p5d / 2.5d / 2d
      - slab / 3d

    The implementation remains a 2D/slab channel-stack model; "3d" is only
    accepted as a backward-compatible config alias for slab mode.
    """
    if isinstance(args_or_mode, str):
        mode = args_or_mode
    else:
        phase3 = getattr(args_or_mode, "phase3", None)
        mode = getattr(phase3, "recon_mode", "slice") if phase3 is not None else "slice"
    mode = str(mode).lower().replace("-", "_").replace(".", "")
    if mode in {"slice", "2p5d", "25d", "2_5d", "2d"}:
        return "slice"
    if mode in {"slab", "3d"}:
        return "slab"
    raise ValueError(f"Unsupported phase3.recon_mode: {mode}. Use slice/2p5d or slab/3d.")


def is_slab_recon(args) -> bool:
    return normalize_recon_mode(args) == "slab"


def slab_num_slices(args) -> int:
    if not is_slab_recon(args):
        return 1
    phase3 = getattr(args, "phase3", None)
    num_slices = int(getattr(phase3, "num_slices", 3)) if phase3 is not None else 3
    if num_slices < 1 or num_slices % 2 == 0:
        raise ValueError(f"phase3.num_slices must be a positive odd integer, got {num_slices}")
    return num_slices


def windowed_input_x_slab(input, micro_b, final_shape, num_frames, num_slices):
    """
    Gather a raw-x slice slab and temporal window for each center sample.

    input has flattened first dimension [time * raw_x_slice, ...].
    Returns inp with shape [B, S, T, ...] and window_idx [B, S, T].
    """
    total_frames = int(final_shape[-5])
    total_slices = int(final_shape[-4])
    frame_half = num_frames // 2
    slice_half = num_slices // 2
    frame_offsets = range(-frame_half, num_frames - frame_half)
    slice_offsets = range(-slice_half, num_slices - slice_half)

    def clamp_slice_index(value):
        return max(0, min(total_slices - 1, value))

    window_idx = []
    for idx in micro_b:
        slice_i, frame_i = ind2xy(int(idx), total_frames, total_slices)
        slice_rows = []
        for slice_off in slice_offsets:
            s = clamp_slice_index(slice_i + slice_off)
            frame_idxs = []
            for frame_off in frame_offsets:
                f = (frame_i + frame_off) % total_frames
                frame_idxs.append(xy2ind(s, f, total_frames, total_slices))
            slice_rows.append(frame_idxs)
        window_idx.append(slice_rows)

    window_idx = torch.tensor(window_idx, dtype=torch.long, device=input.device)
    inp = torch.Tensor(input[window_idx])
    return inp, window_idx


def select_mra_prior_for_microbatch(case_mra_prior, micro_b, final_shape):
    if case_mra_prior is None:
        return None
    prior = torch.as_tensor(case_mra_prior, dtype=torch.float32)
    if prior.dim() == 2:
        prior = prior.unsqueeze(0)
    if prior.dim() != 3:
        raise ValueError(f"Expected mra_prior with shape [1,z,y] or [x,z,y], got {tuple(prior.shape)}")

    if prior.shape[0] == 1:
        return prior.unsqueeze(0).expand(len(micro_b), -1, -1, -1)

    num_slices = int(final_shape[-4])
    slice_indices = torch.as_tensor([int(idx) % num_slices for idx in micro_b], dtype=torch.long)
    if prior.shape[0] != num_slices:
        raise ValueError(
            f"Per-slice mra_prior has {prior.shape[0]} slices, but final_shape reports {num_slices} raw-x slices."
        )
    return prior.index_select(0, slice_indices).unsqueeze(1)


def select_mra_prior_slab_for_microbatch(case_mra_prior, micro_b, final_shape, num_slices):
    if case_mra_prior is None:
        return None
    prior = torch.as_tensor(case_mra_prior, dtype=torch.float32)
    if prior.dim() == 2:
        prior = prior.unsqueeze(0)
    if prior.dim() != 3:
        raise ValueError(f"Expected mra_prior with shape [1,z,y] or [x,z,y], got {tuple(prior.shape)}")

    if prior.shape[0] == 1:
        return prior.expand(num_slices, -1, -1).unsqueeze(0).expand(len(micro_b), -1, -1, -1)

    total_slices = int(final_shape[-4])
    if prior.shape[0] != total_slices:
        raise ValueError(
            f"Per-slice mra_prior has {prior.shape[0]} slices, but final_shape reports {total_slices} raw-x slices."
        )
    slice_half = num_slices // 2
    offsets = range(-slice_half, num_slices - slice_half)
    rows = []
    for idx in micro_b:
        slice_i, _frame_i = ind2xy(int(idx), int(final_shape[-5]), total_slices)
        rows.append([max(0, min(total_slices - 1, slice_i + off)) for off in offsets])
    slice_indices = torch.tensor(rows, dtype=torch.long)
    return prior[slice_indices]


def reshape_channel_to_batch_dim(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Combines batch and channel dimensions.

    Args:
        x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

    Returns:
        A tuple containing:
            (1) output of shape (B*C,1,...)
            (2) batch size
    """

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, two), b

    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, d, two), b

    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_batch_channel_to_channel_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Detaches batch and channel dimensions.

    Args:
        x: input of shape (B*C,1,H,W,2) for 2D data or (B*C,1,H,W,D,2) for 3D data
        batch_size: batch size

    Returns:
        output of shape (B,C,...)
    """
    if len(x.shape) == 4:  # this is 2D real
        bc, chan, h, w = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, chan, h, w)
    if len(x.shape) == 5:  # this is 2D
        bc, one, h, w, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, two)

    elif len(x.shape) == 6:  # this is 3D
        bc, one, h, w, d, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, d, two)

    else:
        raise ValueError(f"only 2D (B*C,1,H,W,2) and 3D (B*C,1,H,W,D,2) data are supported but x has shape {x.shape}")


class Lookahead(Optimizer):
    """
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        """
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "step_counter": self.step_counter,
            "k": self.k,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["cached_params"])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state["cached_params"])  # crucial line
                    param_state["cached_params"].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"]
                        )
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss


def get_training_set(
    filenames: List[str | os.PathLike],
    modality_mapping: Dict[str, str],
    seed: Optional[int] = None,
    balanced: bool = True,
    shuffle: bool = False,
    cmrxrecon_23_data_ratio: float = 1.0,
    acq_types: List[str] = None,
) -> List[str]:
    """
    Given a list of filenames and a mapping from file-suffix to modality,
    returns a balanced subset of filenames (undersampling majority classes)
    so each modality has the same number of samples (min class size).

    Args:
        filenames: List of file paths or names.
        modality_mapping: dict mapping suffix keys (e.g. 'cine_sax') to modality names.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of filenames forming a balanced training set.
    """
    # Prepare
    if seed is not None:
        random.seed(seed)
    # Sort suffixes by length to match the longest first
    filenames = sorted(filenames)

    if acq_types is not None:
        acq_types = set([acq.lower() for acq in acq_types])
        for k in [k for k, v in modality_mapping.items() if v.lower() not in acq_types]:
            del modality_mapping[k]
    suffixes = sorted(modality_mapping.keys(), key=len, reverse=True)

    # Group files by modality
    groups = defaultdict(list)
    for fname in filenames:
        stem, _ = os.path.splitext(os.path.basename(fname))
        modality = None
        for suf in suffixes:
            if stem.lower().endswith(suf):
                modality = modality_mapping[suf]
                break
        if modality:
            if "CMRxRecon2023_Data" in str(fname):
                if random.random() > cmrxrecon_23_data_ratio:
                    continue
            groups[modality].append(fname)
        else:
            # if you prefer to catch unmapped files:
            # raise ValueError(f"No modality mapping for file '{fname}'")
            continue

    # find smallest class size
    if balanced:
        num_files = sum(len(v) for v in groups.values())
        num_modality = 9
        if not groups:
            return []
        n_count_per_modality = num_files // num_modality

        # sample min_count from each group
        balanced = []
        for _, files in groups.items():
            while len(files) < n_count_per_modality:
                files.extend(files)
            files = random.sample(files, n_count_per_modality)
            balanced.extend(files)
        # optional: shuffle final list
        if shuffle:
            random.shuffle(balanced)
        return balanced
    else:
        all = []
        for _, files in groups.items():
            all.extend(files)
        if shuffle:
            random.shuffle(all)
        return all


def report_nan_from_any_rank(loss, file_name, micro_b, is_ddp=False):
    if is_ddp:
        device = loss.device
        rank = dist.get_rank()
        world = dist.get_world_size()

        local_bad = loss.isnan()
        # Step 1: detect if any rank is bad
        any_bad = torch.tensor([int(local_bad)], device=device, dtype=torch.int32)
        dist.all_reduce(any_bad, op=dist.ReduceOp.MAX)
        any_bad = bool(any_bad.item())

        # Step 2: optionally collect details to rank 0
        if any_bad:
            # gather arbitrary python objects (PyTorch 1.8+)
            mine = {"rank": rank, "file": file_name, "micro_b": micro_b} if local_bad else None
            gather_list = [None] * world if rank == 0 else None
            dist.gather_object(mine, gather_list, dst=0)

            if rank == 0:
                offenders = [o for o in gather_list if o is not None]
                ranks = [o["rank"] for o in offenders]
                print(f"NaN/Inf loss detected on ranks {ranks}. Skipping this step.")
                for o in offenders:
                    print(f"  • rank {o['rank']}: file={o['file']} slice={o['micro_b']}")
    else:
        any_bad = loss.isnan()
        if any_bad:
            print(f"NaN Loss detected for {file_name} at slice {micro_b}. Skipping this step.")

    return any_bad


def zero_grad_scalar_fast(model):
    # touches exactly one element from each *trainable* parameter
    s = 0.0
    for p in model.parameters():
        if p.requires_grad:
            s = s + p.view(-1)[0]  # one element per param
    return s * 0.0


if __name__ == "__main__":
    # Suppose these are your training files:
    train_files = [file for file in Path("data/train").iterdir()]

    balanced = get_training_set(train_files, MODALITY_MAPPING, balanced=False)
    print("Balanced subset:")
    for f in balanced:
        print(f)
