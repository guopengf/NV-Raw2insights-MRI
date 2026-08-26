#!/usr/bin/env python3
"""Read one real 4D-flow case and optionally run a joint forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from monai.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from joint_encoding import (  # noqa: E402
    JointEncodingLoss,
    flatten_joint_model_batch,
    gather_joint_window,
    joint_encoding_spec,
    joint_windowed_input_x_slab,
    restore_joint_model_batch,
)
from models.latent_recon import create_mri_recon_model  # noqa: E402
from mri_data.data_utils import crop_k_space  # noqa: E402
from train_utils import get_train_transforms  # noqa: E402
from utils import load_config, load_net, sensitivity_map_reduce  # noqa: E402


def find_case(data_root: Path):
    for full_kspace in sorted(data_root.glob("Center*/*/P*/kdata_full.mat")):
        case_dir = full_kspace.parent
        undersampled = case_dir / "kdata_ktGaussian10.mat"
        mask = case_dir / "usmask_ktGaussian10.mat"
        coilmap = case_dir / "coilmap.mat"
        if undersampled.exists() and mask.exists() and coilmap.exists():
            return case_dir, full_kspace, undersampled, mask, coilmap
    raise FileNotFoundError(f"No complete acceleration-10 case under {data_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batch", "channel"), required=True)
    parser.add_argument("--forward-backward", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TrainSet/Aorta"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace/code/NV-Raw2insights-MRI-fork/cache/"
            "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_compatible.pt"
        ),
    )
    args = parser.parse_args()

    config_path = (
        REPO_ROOT
        / "configs"
        / f"nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_{args.mode}_pg.json"
    )
    config = load_config(config_path)
    spec = joint_encoding_spec(config)
    case_dir, full_kspace, undersampled, mask_path, coilmap = find_case(args.data_root)
    payload = {
        "kspace": str(undersampled),
        "target_kspace": str(full_kspace),
        "mask": [str(mask_path)],
        "mask_type": "ktGaussian10",
        "acquisition": "Flow4d",
        "joint_encodings": True,
        "encoding_indices": list(spec.order),
        "is_4dflow": True,
        "coilmap": str(coilmap),
    }
    segmask = case_dir / "segmask.mat"
    if segmask.exists():
        payload["segmask"] = str(segmask)

    (REPO_ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / "outputs") as temporary:
        manifest = Path(temporary) / "joint4.json"
        manifest.write_text(json.dumps(payload))
        started = time.perf_counter()
        sample = Dataset(data=[{"kspace": manifest}], transform=get_train_transforms(config))[0]
        elapsed = time.perf_counter() - started

    expected_encodings = spec.count
    tensor_shapes = {
        key: tuple(sample[key].shape)
        for key in ("kspace_masked_ifft", "kspace_ifft", "mask", "mean", "std", "sensitivity_maps")
    }
    for key, shape in tensor_shapes.items():
        if shape[1] != expected_encodings:
            raise AssertionError(f"{key} lost the encoding axis: {shape}")
    if int(config.batch_size) != 8 or int(config.num_samples_per_case) != 8:
        raise AssertionError("Smoke config changed fixed batch_size/num_samples_per_case")
    print(f"REAL_LOADER_OK mode={args.mode} case={case_dir} seconds={elapsed:.3f} shapes={tensor_shapes}")

    if not args.forward_backward:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("--forward-backward requires CUDA")
    device = torch.device("cuda")
    model = create_mri_recon_model(config).to(device)
    model = load_net(model, args.checkpoint, device, resume_training_state=False)[0]
    model.train()

    input_tensor = sample["kspace_masked_ifft"]
    target = sample["kspace_ifft"]
    mask = sample["mask"]
    mean = sample["mean"]
    std = sample["std"]
    sensitivity = sample["sensitivity_maps"]
    final_shape = [int(value) for value in sample["kspace_meta_dict"]["shape"]]
    micro_b = [0, 1]
    inp_joint, window_idx = joint_windowed_input_x_slab(
        input_tensor,
        micro_b,
        final_shape,
        num_frames=int(config.num_frames),
        num_slices=int(config.phase3.num_slices),
    )
    tar_joint = gather_joint_window(target, window_idx)
    mask_joint = gather_joint_window(mask, window_idx)
    mean_joint = gather_joint_window(mean, window_idx)
    std_joint = gather_joint_window(std, window_idx)
    sens_joint = gather_joint_window(sensitivity, window_idx)

    inp = flatten_joint_model_batch(inp_joint).to(device)
    tar = flatten_joint_model_batch(tar_joint).to(device)
    model_mask = flatten_joint_model_batch(mask_joint).to(device).bool()
    sens = flatten_joint_model_batch(sens_joint).to(device)
    mean_window = flatten_joint_model_batch(mean_joint).to(device)
    std_window = flatten_joint_model_batch(std_joint).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast("cuda", torch.bfloat16, enabled=bool(config.amp)):
        output = model(
            inp,
            model_mask,
            "ktGaussian10",
            10,
            "Flow4d",
            sensitivity_maps=sens,
            mra_prior=None,
        )
    center_t = int(config.num_frames) // 2
    output = output * std_window + mean_window
    output = output[:, :, center_t]
    tar = tar[:, :, center_t]
    output = crop_k_space(output, (final_shape[-2], final_shape[-1]))
    tar = crop_k_space(tar, (final_shape[-2], final_shape[-1]))
    sens_center = crop_k_space(sens[:, :, center_t], (final_shape[-2], final_shape[-1]))
    output = sensitivity_map_reduce(output.float(), sens_center.float())
    tar = sensitivity_map_reduce(tar.float(), sens_center.float())
    output = restore_joint_model_batch(output, spec.count)
    tar = restore_joint_model_batch(tar, spec.count)
    loss = JointEncodingLoss(encoding_count=spec.count)(output, tar)[0]
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite joint loss: {loss.item()}")
    print(
        f"REAL_FORWARD_BACKWARD_OK mode={args.mode} flat_batch={inp.shape[0]} "
        f"loss={loss.item():.6g} seconds={elapsed:.3f} peak_allocated_gib={peak_gib:.3f}"
    )


if __name__ == "__main__":
    main()
