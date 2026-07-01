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

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from evaluation import calmetric
from models.latent_recon import create_mri_recon_model
from monai.apps.reconstruction.complex_utils import complex_abs
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, partition_dataset
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.utils import set_determinism
from path_safety import assert_outputs_not_in_data
from mri_data.data_utils import (
    FlowVNPhaseLoss,
    crop_k_space,
    gather_metric,
    get_loss_function,
    postprocess_mri_recon,
    rearrange_mri_data,
)
from torch.amp import GradScaler, autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
from train_utils import apply_phase3_freeze, get_optimizer, get_train_transforms, get_val_transforms
from transforms import *
from utils import *

import wandb

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

warnings.filterwarnings("ignore")


def cfg_get(obj, path, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, default)
    return cur


def collect_vaa_gamma(model):
    module = model.module if hasattr(model, "module") else model
    gamma_by_location = {}
    for name, submodule in module.named_modules():
        if not hasattr(submodule, "gamma_raw") or not callable(getattr(submodule, "gamma", None)):
            continue
        location = name
        if "vaa_adapters." in name:
            location = name.split("vaa_adapters.", 1)[1].split(".", 1)[0]
        gamma_by_location.setdefault(location, []).append(float(submodule.gamma().detach().cpu()))
    return {name: float(np.mean(values)) for name, values in gamma_by_location.items()}


def short_train_log_name(name):
    return {
        "main_zy_loss_weighted": "main",
        "phase_loss_weighted": "phase",
        "vascular_phase_loss_weighted": "vphase",
        "loss_sum": "sum",
        "bottleneck": "gb",
        "intermediate": "gi",
    }.get(name, name)


def build_4dflow_aorta_manifests(data_roots, out_dir, accelerations=None, encodings=None):
    accelerations = accelerations or [10, 20, 30, 40, 50]
    encodings = encodings or [0, 1, 2, 3]
    out_dir = assert_outputs_not_in_data([out_dir], data_roots)[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = []
    for root_str in data_roots:
        root = Path(root_str)
        if not root.exists():
            continue
        for center_dir in sorted(root.glob("Center*")):
            if not center_dir.is_dir():
                continue
            for scanner_dir in sorted(center_dir.iterdir()):
                if not scanner_dir.is_dir():
                    continue
                for patient_dir in sorted(scanner_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue

                    full_kspace = patient_dir / "kdata_full.mat"
                    coilmap = patient_dir / "coilmap.mat"
                    segmask = patient_dir / "segmask.mat"
                    if not full_kspace.exists():
                        continue

                    for acc in accelerations:
                        us_kspace = patient_dir / f"kdata_ktGaussian{int(acc)}.mat"
                        us_mask = patient_dir / f"usmask_ktGaussian{int(acc)}.mat"
                        if not us_kspace.exists() or not us_mask.exists():
                            continue

                        for enc_idx in encodings:
                            item = {
                                "kspace": str(us_kspace),
                                "target_kspace": str(full_kspace),
                                "mask": [str(us_mask)],
                                "mask_type": f"ktGaussian{int(acc)}",
                                "acquisition": "Flow4d",
                                "encoding_idx": int(enc_idx),
                                "is_4dflow": True,
                            }
                            if coilmap.exists():
                                item["coilmap"] = str(coilmap)
                            if segmask.exists():
                                item["segmask"] = str(segmask)

                            out_name = (
                                f"{center_dir.name}__{scanner_dir.name}__{patient_dir.name}"
                                f"__ktGaussian{int(acc)}__enc{int(enc_idx)}.json"
                            )
                            out_path = out_dir / out_name
                            with open(out_path, "w") as f:
                                json.dump(item, f, indent=2)
                            manifest_paths.append(out_path)

    return manifest_paths


@record
def trainer(args):
    if args.ddp:
        # initialize the distributed training process, every GPU runs in a process
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Set device before initializing process group
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Initialize process group with timeout and device mapping
        timeout = timedelta(seconds=1800)  # 30 minutes timeout

        # Initialize process group with explicit device mapping
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

        # Ensure all processes are synchronized after initialization
        dist.barrier(device_ids=[local_rank])
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1

    if args.seed is not None:
        set_determinism(seed=args.seed)
    # use amp to accelerate training
    scaler = GradScaler(enabled=args.amp)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    outpath = os.path.join(args.exp_dir, args.exp)
    assert_outputs_not_in_data([outpath], [*args.data_path_train, *args.data_path_val])
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    use_multi_epochs_train_loader = bool(cfg_get(args, "use_multi_epochs_train_loader", False))

    # create training-validation data loaders
    if getattr(args, "is_4dflow_aorta", False):
        train_manifest_dir = Path(outpath) / "jsons_train"
        val_manifest_dir = Path(outpath) / "jsons_val"
        if rank == 0:
            for manifest_dir in (train_manifest_dir, val_manifest_dir):
                manifest_dir.mkdir(parents=True, exist_ok=True)
                for old_manifest in manifest_dir.glob("*.json"):
                    old_manifest.unlink()
            build_4dflow_aorta_manifests(
                args.data_path_train,
                train_manifest_dir,
                accelerations=getattr(args, "four_dflow_accelerations", [10, 20, 30, 40, 50]),
                encodings=getattr(args, "four_dflow_encodings", [0, 1, 2, 3]),
            )
            build_4dflow_aorta_manifests(
                args.data_path_val,
                val_manifest_dir,
                accelerations=getattr(args, "four_dflow_accelerations", [10, 20, 30, 40, 50]),
                encodings=getattr(args, "four_dflow_encodings", [0, 1, 2, 3]),
            )
        if args.ddp:
            dist.barrier(device_ids=[local_rank])
        train_files = sorted(train_manifest_dir.glob("*.json"))
        val_files = sorted(val_manifest_dir.glob("*.json"))
    else:
        train_files = [file for path_str in args.data_path_train for file in Path(path_str).iterdir()]
        val_files = [file for path_str in args.data_path_val for file in Path(path_str).iterdir()]

    training_file_seed = int(os.getenv("SLURM_JOB_ID", "42"))
    print(f"#using seed: {training_file_seed}, balanced sampling: {args.balance_data}")
    if args.dataset.lower() == "cmrxrecon" and not getattr(args, "is_4dflow_aorta", False):
        train_files = get_training_set(
            train_files,
            MODALITY_MAPPING,
            seed=training_file_seed,
            balanced=args.balance_data,
            shuffle=True,
            acq_types=args.acq_types if hasattr(args, "acq_types") else None,
        )
    elif args.dataset.lower() == "fastmri":
        train_files = train_files
    train_files = train_files[
        : int(args.sample_rate * len(train_files))
    ]  # select a subset of the data according to sample_rate
    train_files = [dict([("kspace", train_files[i])]) for i in range(len(train_files))]
    print(f"#training files: {len(train_files)}")
    if len(train_files) == 0:
        raise RuntimeError(
            "No training files were found. Check data_path_train, four_dflow_accelerations, "
            "and required files kdata_full/kdata_ktGaussian*/usmask_ktGaussian*/coilmap.mat."
        )
    if use_multi_epochs_train_loader:
        train_files = partition_dataset(
            data=train_files,
            num_partitions=world_size,
            shuffle=True,
            even_divisible=True,
        )[rank]

    val_files = val_files[
        : int(args.sample_rate * len(val_files))
    ]  # select a subset of the data according to sample_rate
    val_files = [dict([("kspace", val_files[i])]) for i in range(len(val_files))]
    print(f"#validation files: {len(val_files)}")
    if len(val_files) < world_size:
        raise RuntimeError(
            f"Not enough validation files ({len(val_files)}) for world_size={world_size}. "
            "Check data_path_val or reduce --nproc_per_node."
        )
    val_files = partition_dataset(
        data=val_files,
        num_partitions=world_size,
        shuffle=False,
        even_divisible=True,
    )[rank]

    if args.debug:
        train_files = train_files[:1]
        val_files = val_files[:1]

    # define mask transform type (e.g., whether it is equispaced or random)
    print(f"train_mask_types: {args.train_mask_types}")
    print(f"val_mask_types: {args.val_mask_types}")
    print(f"center_fractions: {args.center_fractions}")
    print(f"accelerations: {args.accelerations}")

    # create the model
    model = create_mri_recon_model(args).to(device)
    try:
        args.is_multi_coil = model.use_csm or model.use_latent_csm
    except BaseException:
        args.is_multi_coil = True

    # Auto resume from the current experiment first; use resume_ckpt only to
    # bootstrap a fresh output directory.
    pretrained_path = resolve_checkpoint_path(args.model_variant)
    resume_ckpt = getattr(args, "resume_ckpt", None)
    resume_path = os.path.join(outpath, args.model_filename)
    if os.path.exists(resume_path):
        print(f"Auto-resume from experiment checkpoint: {resume_path}")
    elif resume_ckpt:
        resume_path = str(resume_ckpt)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Configured resume_ckpt does not exist: {resume_path}")
        print(f"Resume from configured checkpoint: {resume_path}")
    else:
        resume_path = pretrained_path
    # Load the model, optimizer, and scheduler
    (
        model,
        optimizer_state_dict,
        scheduler_state_dict,
        scaler_state_dict,
        start_epoch,
        start_global_step,
        best_metric,
        best_metric_epoch,
        wandb_run_id,
    ) = load_net(
        model,
        resume_path,
        device,
        is_ddp=args.ddp,
        resume_rng_state=args.resume_rng_state,
        prepare_model_for_ddp=lambda m: apply_phase3_freeze(args, m),
    )
    model = torch.compile(model) if args.uniform_input_kspace else model
    model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#model_params: {model_params * 1.0e-6:.2f}M")
    print(f"#trainable_model_params: {trainable_model_params * 1.0e-6:.2f}M")

    train_transforms = get_train_transforms(args)
    val_transforms = get_val_transforms(args)

    train_ds = (
        Dataset(data=train_files, transform=train_transforms)
        if args.cache_rate == 0
        else CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
        )
    )
    if use_multi_epochs_train_loader:
        train_sampler = None
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if args.ddp else None
    train_loader_cls = MultiEpochsDataLoader if use_multi_epochs_train_loader else DataLoader
    print(f"train_loader: {train_loader_cls.__name__}")
    train_loader = train_loader_cls(
        train_ds,
        batch_size=1,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        in_order=False,
    )

    # since there's no randomness in train_transforms, we use it for val_transforms as well
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = MultiEpochsDataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        in_order=False,
    )

    # create the loss function
    loss_function = get_loss_function(args, device)
    phase_loss_function = FlowVNPhaseLoss(
        eps=float(cfg_get(args, "phase3.loss.phase.eps", 1e-8)),
        normalize_mask=bool(cfg_get(args, "phase3.loss.vascular.normalize_by_mask", True)),
        method=str(cfg_get(args, "phase3.loss.phase.method", "flowvn_complex_l1")),
    ).to(device)
    vascular_loss_function = FlowVNPhaseLoss(
        eps=float(cfg_get(args, "phase3.loss.vascular.eps", cfg_get(args, "phase3.loss.phase.eps", 1e-8))),
        normalize_mask=bool(cfg_get(args, "phase3.loss.vascular.normalize_by_mask", True)),
        method=str(cfg_get(args, "phase3.loss.vascular.method", "mra_masked_phase_l1")),
    ).to(device)
    use_main_zy_loss = bool(cfg_get(args, "phase3.loss.use_ssim_zy", True))
    use_phase_loss = bool(cfg_get(args, "phase3.loss.use_phase", False))
    use_vascular_loss = bool(cfg_get(args, "phase3.loss.use_vascular", False))
    recon_slab = is_slab_recon(args)
    recon_num_slices = slab_num_slices(args)
    if not (use_main_zy_loss or use_phase_loss or use_vascular_loss):
        raise RuntimeError(
            "At least one training loss must be enabled: phase3.loss.use_ssim_zy, "
            "phase3.loss.use_phase, or phase3.loss.use_vascular."
        )
    phase_loss_weight = float(cfg_get(args, "phase3.loss.phase.weight", cfg_get(args, "phase3.loss.weights.phase", 1.0)))
    vascular_loss_weight = float(
        cfg_get(args, "phase3.loss.vascular.weight", cfg_get(args, "phase3.loss.weights.vascular", 1.0))
    )

    # create the optimizer and the learning rate scheduler
    eff_batch_size = args.batch_size * dist.get_world_size() if args.ddp else args.batch_size

    print(f"lr_schedule: {args.lr_schedule}")
    print(f"min_lr: {args.min_lr}")
    print(f"warmup_epochs: {args.warmup_epochs}")
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    args.model_structure = str(model).split("\n")
    save_args_to_file_json(args, os.path.join(outpath, "config.json"))

    optimizer = get_optimizer(args, model)
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            print("optimizer state dict loaded from resume checkpoint.")
        except Exception as e:
            print(f"Rank {rank}: Failed to load optimizer state dict: {e}. Proceeding without optimizer state dict.")
    else:
        print("optimizer state dict is not found.")
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)
        print("scaler state dict loaded from resume checkpoint.")
    else:
        print("scaler state dict is not found.")

    scheduler = None
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        print("scheduler state dict loaded from resume checkpoint.")
    else:
        print("lr scheduler is None or state dict is not found.")

    # start a typical PyTorch training loop
    val_interval = args.val_interval
    print(f"val_interval: {val_interval}")
    global_step = start_global_step
    args.num_epochs = start_epoch + 1 if args.val else args.num_epochs

    if rank == 0:
        writer = SummaryWriter(
            outpath + "/" + datetime.now().strftime("%m-%d-%y_%H-%M")
        )  # create a date directory within the output directory for storing training logs
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project="SDUM",
            name=args.exp,
            id=wandb_run_id if wandb_run_id is not None else wandb.util.generate_id(),
            # Track hyperparameters and run metadata.
            config=args.to_dict(),
        )

    max_compute_cost = 0
    for epoch in range(start_epoch, args.num_epochs):
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tic = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        nan_loss_count = 0
        for b, batch_data in enumerate(train_loader):
            if args.val:
                break
            if start_epoch == epoch and step <= 10:
                adjust_learning_rate(optimizer, b / len(train_loader) + epoch, args, is_resume_first_ten=True)
            else:
                adjust_learning_rate(optimizer, b / len(train_loader) + epoch, args, is_resume_first_ten=False)
            (
                input,
                target,
                mask,
                mask_type,
                acc_factor,
                acq_type,
                mean,
                std,
                file_name,
                final_shape,
            ) = (
                batch_data["kspace_masked_ifft"][0],
                batch_data["kspace_ifft"][0],
                batch_data["mask"][0],
                batch_data["mask_type"][0],
                batch_data["acc_factor"][0],
                batch_data["acquisition"][0],
                batch_data["mean"][0],
                batch_data["std"][0],
                batch_data["kspace_meta_dict"]["filename"][0],
                batch_data["kspace_meta_dict"]["shape"][0],  # [t, z, c, y, x]
            )

            final_shape = [int(s) for s in final_shape]
            sensitivity_maps = batch_data.get("sensitivity_maps")
            sensitivity_maps = sensitivity_maps[0] if sensitivity_maps is not None else None
            case_mra_prior = batch_data.get("mra_prior")
            case_mra_prior = case_mra_prior[0] if case_mra_prior is not None else None

            # iterate through all slices
            sample_list = list(range(input.shape[0]))
            num_samples = min(input.shape[0], args.num_samples_per_case)
            micro_batch_size = args.batch_size
            compute_cost = input.shape[-2] * input.shape[-3]
            if compute_cost > max_compute_cost:
                max_compute_cost = torch.tensor(compute_cost, device=device)

            if args.ddp:
                dist.all_reduce(max_compute_cost, op=dist.ReduceOp.MAX)
            batch_size_scale = max_compute_cost.item() / compute_cost if args.adaptive_batch_size else 1
            adjusted_micro_batch_size = int(batch_size_scale * micro_batch_size)

            max_num_batches = math.ceil(num_samples / adjusted_micro_batch_size)
            # find min num_samples across GPUs
            if args.ddp:
                num_samples = torch.tensor(num_samples, device=device)
                dist.all_reduce(num_samples, op=dist.ReduceOp.MIN)
                max_num_batches = torch.tensor(max_num_batches, device=device)
                dist.all_reduce(max_num_batches, op=dist.ReduceOp.MIN)

            for micro_b, i in mini_dataloader(
                sample_list,
                adjusted_micro_batch_size,
                shuffle=True,
                infinite=True,
                max_num_batches=max_num_batches,
            ):

                step += 1
                optimizer.zero_grad()

                # forward pass
                if recon_slab:
                    inp, window_idx = windowed_input_x_slab(
                        input, micro_b, final_shape, num_frames=args.num_frames, num_slices=recon_num_slices
                    )
                else:
                    inp, window_idx = windowed_input(input, micro_b, final_shape, num_frames=args.num_frames)
                tar = torch.Tensor(target[window_idx])
                mas = torch.Tensor(mask[window_idx])
                sens = torch.Tensor(sensitivity_maps[window_idx]) if sensitivity_maps is not None else None
                mra_prior = (
                    select_mra_prior_slab_for_microbatch(case_mra_prior, micro_b, final_shape, recon_num_slices)
                    if recon_slab
                    else select_mra_prior_for_microbatch(case_mra_prior, micro_b, final_shape)
                )
                inp, tar, mas, mean, std = (
                    inp.to(device),
                    tar.to(device),
                    mas.to(device),
                    mean.to(device),
                    std.to(device),
                )
                sens = sens.to(device) if sens is not None else None
                mra_prior = mra_prior.to(device) if mra_prior is not None else None
                with autocast("cuda", torch.bfloat16, enabled=args.amp):
                    output = model(inp, mas.bool(), mask_type, acc_factor, acq_type, sensitivity_maps=sens, mra_prior=mra_prior)

                if recon_slab:
                    output_norm = output[:, :, args.num_frames // 2]
                    target_norm = ((tar - mean[window_idx]) / std[window_idx])[:, :, args.num_frames // 2]
                else:
                    output_norm = output[:, args.num_frames // 2]
                    target_norm = ((tar - mean[window_idx]) / std[window_idx])[:, args.num_frames // 2]
                output = output * std[window_idx] + mean[window_idx]  # [b, c/1, h, w, 2]
                if recon_slab:
                    output = output[:, :, args.num_frames // 2]
                    tar = tar[:, :, args.num_frames // 2]
                else:
                    output = output[:, args.num_frames // 2]
                    tar = tar[:, args.num_frames // 2]
                output_complex_norm = crop_k_space(output_norm, (final_shape[-2], final_shape[-1]))
                target_complex_norm = crop_k_space(target_norm, (final_shape[-2], final_shape[-1]))
                output_complex = crop_k_space(output, (final_shape[-2], final_shape[-1]))
                target_complex = crop_k_space(tar, (final_shape[-2], final_shape[-1]))
                output = complex_abs(output_complex)  # [b, c/1, h, w]
                tar = complex_abs(target_complex)

                if recon_slab:
                    output_rss = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)).transpose(1, 2)
                    targets_rss = torch.sqrt(torch.sum(tar**2, dim=2, keepdim=True)).transpose(1, 2)
                else:
                    output_rss = torch.sqrt(torch.sum(output**2, dim=1, keepdim=True))
                    targets_rss = torch.sqrt(torch.sum(tar**2, dim=1, keepdim=True))

                loss_dict = {}
                weighted_loss_log = {}
                if use_main_zy_loss:
                    with autocast("cuda", torch.bfloat16, enabled=False):
                        output_rss_pp = postprocess_mri_recon(
                            output_rss,
                            args,
                            file_name,
                            is_training=True,
                            pp_z_score_norm=args.pp_z_score_norm,
                            pp_norm=args.pp_norm,
                        )
                        targets_rss_pp = postprocess_mri_recon(
                            targets_rss,
                            args,
                            file_name,
                            is_training=True,
                            pp_z_score_norm=args.pp_z_score_norm,
                            pp_norm=args.pp_norm,
                        )
                        if args.loss_type == "ssim":
                            dims = tuple(range(1, targets_rss_pp.dim()))
                            max_value = (
                                targets_rss_pp.amax(dim=dims).to(device)
                                if "max" not in batch_data["kspace_meta_dict"]
                                else batch_data["kspace_meta_dict"]["max"][0].item()
                            )
                            loss_function.data_range = max_value
                        elif args.loss_type == "ssim_l1":
                            dims = tuple(range(1, targets_rss_pp.dim()))
                            max_value = (
                                targets_rss_pp.amax(dim=dims).to(device)
                                if "max" not in batch_data["kspace_meta_dict"]
                                else batch_data["kspace_meta_dict"]["max"][0].item()
                            )
                            loss_function.ssim_loss.data_range = max_value
                        loss_dict = loss_function(output_rss_pp, targets_rss_pp)

                    if isinstance(loss_dict, dict):
                        loss = loss_dict["combined_loss"]
                    else:
                        loss = loss_dict
                    weighted_loss_log["main_zy_loss_weighted"] = loss.detach()
                else:
                    loss = output_complex.sum() * 0.0

                aux_loss_log = {}
                if use_phase_loss or use_vascular_loss:
                    with autocast("cuda", torch.bfloat16, enabled=False):
                        phase_output = output_complex_norm.float()
                        phase_target = target_complex_norm.float()
                        if sens is not None:
                            sens_center = sens[:, :, args.num_frames // 2] if recon_slab else sens[:, args.num_frames // 2]
                            sens_center = crop_k_space(sens_center, (final_shape[-2], final_shape[-1]))
                            phase_output = sensitivity_map_reduce(phase_output, sens_center)
                            phase_target = sensitivity_map_reduce(phase_target, sens_center)
                        if recon_slab:
                            b_slab, s_slab = phase_output.shape[:2]
                            phase_output_for_loss = phase_output.reshape(b_slab * s_slab, *phase_output.shape[2:])
                            phase_target_for_loss = phase_target.reshape(b_slab * s_slab, *phase_target.shape[2:])
                            mra_prior_for_loss = (
                                mra_prior.reshape(b_slab * s_slab, 1, *mra_prior.shape[-2:])
                                if mra_prior is not None
                                else None
                            )
                        else:
                            phase_output_for_loss = phase_output
                            phase_target_for_loss = phase_target
                            mra_prior_for_loss = mra_prior
                        if use_phase_loss:
                            phase_loss = phase_loss_function(phase_output_for_loss, phase_target_for_loss)
                            weighted_phase_loss = phase_loss_weight * phase_loss
                            loss = loss + weighted_phase_loss
                            aux_loss_log["phase_loss"] = phase_loss.detach()
                            weighted_loss_log["phase_loss_weighted"] = weighted_phase_loss.detach()
                        if use_vascular_loss:
                            if mra_prior_for_loss is None:
                                raise RuntimeError("phase3.loss.use_vascular=True requires a vascular prior.")
                            vascular_phase_loss = vascular_loss_function(
                                phase_output_for_loss, phase_target_for_loss, mask=mra_prior_for_loss
                            )
                            weighted_vascular_phase_loss = vascular_loss_weight * vascular_phase_loss
                            loss = loss + weighted_vascular_phase_loss
                            aux_loss_log["vascular_phase_loss"] = vascular_phase_loss.detach()
                            weighted_loss_log["vascular_phase_loss_weighted"] = weighted_vascular_phase_loss.detach()

                loss_tensor = loss.clone().detach()
                report_nan_from_any_rank(loss_tensor, file_name, micro_b, is_ddp=args.ddp)
                if not torch.isfinite(loss):
                    loss = zero_grad_scalar_fast(model)  # zero-grad scalar
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if args.ddp:
                    # sum loss_tensor across all processes in-place
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    # compute average and add to epoch_loss
                    step_loss = loss_tensor.item() / world_size
                else:
                    step_loss = loss_tensor.item()
                step_loss_components = {}
                for loss_name, loss_value in weighted_loss_log.items():
                    component_tensor = loss_value.clone().detach()
                    if args.ddp:
                        dist.all_reduce(component_tensor, op=dist.ReduceOp.SUM)
                        step_loss_components[loss_name] = component_tensor.item() / world_size
                    else:
                        step_loss_components[loss_name] = component_tensor.item()
                step_loss_components["loss_sum"] = step_loss
                if step_loss == step_loss:
                    epoch_loss += step_loss
                else:
                    nan_loss_count += 1
                    step -= 1

                if rank == 0:
                    loss_parts = ", ".join(
                        f"{short_train_log_name(name)}={value:.4f}" for name, value in step_loss_components.items()
                    )
                    gamma_values = collect_vaa_gamma(model)
                    gamma_parts = ""
                    if gamma_values:
                        gamma_parts = " " + ", ".join(
                            f"{short_train_log_name(name)}={value:.6f}" for name, value in gamma_values.items()
                        )
                    print(
                        f"{b + 1}/{len(train_loader)} {i + adjusted_micro_batch_size}/{num_samples} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"train_loss={epoch_loss / (step + 1e-8):.4f} {loss_parts}{gamma_parts}",
                    )
                    if (global_step + step) % 10 == 0:
                        writer.add_scalar(
                            "train_step_loss",
                            epoch_loss / (step + 1e-8),
                            global_step + step,
                        )
                        if isinstance(loss_dict, dict):
                            for k, v in loss_dict.items():
                                if k != "combined_loss":
                                    writer.add_scalar(f"train_step_{k}", v, global_step + step)
                        for k, v in aux_loss_log.items():
                            writer.add_scalar(f"train_step_{k}", v, global_step + step)
                        for k, v in step_loss_components.items():
                            writer.add_scalar(f"train_step_{k}", v, global_step + step)
                        for k, v in gamma_values.items():
                            writer.add_scalar(f"train_step_gamma_{k}", v, global_step + step)

                    if step != 0 and step % 10000 == 0:
                        save_checkpoint(
                            epoch,
                            global_step,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            outpath,
                            best_metric,
                            best_metric_epoch,
                            model_filename=args.model_filename,
                            epoch_finished=False,
                            is_ddp=args.ddp,
                            wandb_run_id=run.id,
                        )
        global_step += step
        if scheduler is not None:
            scheduler.step()

        if rank == 0 and not args.val:
            writer.add_scalar("train_loss", epoch_loss / step, epoch + 1)
            run.log(
                {
                    "train/loss": epoch_loss / step,
                },
                step=epoch + 1,
            )
            save_checkpoint(
                epoch,
                global_step,
                model,
                optimizer,
                scheduler,
                scaler,
                outpath,
                best_metric,
                best_metric_epoch,
                model_filename=args.model_filename,
                epoch_finished=True,
                is_ddp=args.ddp,
                wandb_run_id=run.id,
            )
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    epoch,
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    outpath,
                    best_metric,
                    best_metric_epoch,
                    model_filename=args.model_filename[:-3] + "_epoch" + str((epoch + 1)) + ".pt",
                    epoch_finished=True,
                    is_ddp=args.ddp,
                    wandb_run_id=run.id,
                )

            print(
                f"epoch {epoch + 1} average loss: {epoch_loss/step:.4f} nan_loss_count: \
                {nan_loss_count} time elapsed: {(time.time() - tic) / 60:.2f} mins"
            )
        torch.cuda.empty_cache()

        # validation
        if ((epoch + 1) % val_interval == 0) or args.val or epoch == args.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_ssim, val_psnr, val_nmse = list(), list(), list()
                tic_val = time.time()
                for val_data in tqdm.tqdm(val_loader):
                    (
                        input,
                        target,
                        mask,
                        mask_type,
                        acc_factor,
                        acq_type,
                        mean,
                        std,
                        temporal_shuffle,
                        file_name,
                        final_shape,
                    ) = (
                        val_data["kspace_masked_ifft"][0],
                        val_data["kspace_ifft"][0],
                        val_data["mask"][0],
                        val_data["mask_type"][0],
                        val_data["acc_factor"][0],
                        val_data["acquisition"][0],
                        val_data["mean"][0],
                        val_data["std"][0],
                        val_data["temporal_shuffle"][0],
                        val_data["kspace_meta_dict"]["filename"][0],
                        val_data["kspace_meta_dict"]["shape"][0],
                    )
                    final_shape = [int(s) for s in final_shape]
                    sensitivity_maps = val_data.get("sensitivity_maps")
                    sensitivity_maps = sensitivity_maps[0] if sensitivity_maps is not None else None
                    case_mra_prior = val_data.get("mra_prior")
                    case_mra_prior = case_mra_prior[0] if case_mra_prior is not None else None
                    input = (
                        fftn_centered(input, spatial_dims=2, is_complex=True)
                        if args.model_type.lower() in ["varnet", "kspace_mar"]
                        else input
                    )

                    # iterate through all samples:
                    num_samples = input.shape[0]
                    outputs = []
                    targets = []
                    for micro_b, _ in mini_dataloader(
                        list(range(num_samples)),
                        2 * args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        pad_last=False,
                    ):
                        # forward pass
                        if recon_slab:
                            inp, window_idx = windowed_input_x_slab(
                                input, micro_b, final_shape, num_frames=args.num_frames, num_slices=recon_num_slices
                            )
                        else:
                            inp, window_idx = windowed_input(input, micro_b, final_shape, num_frames=args.num_frames)
                        tar = torch.Tensor(target[window_idx])
                        mas = torch.Tensor(mask[window_idx])
                        sens = torch.Tensor(sensitivity_maps[window_idx]) if sensitivity_maps is not None else None
                        mra_prior = (
                            select_mra_prior_slab_for_microbatch(case_mra_prior, micro_b, final_shape, recon_num_slices)
                            if recon_slab
                            else select_mra_prior_for_microbatch(case_mra_prior, micro_b, final_shape)
                        )
                        inp, tar, mas, mean, std = (
                            inp.to(device),
                            tar.to(device),
                            mas.to(device),
                            mean.to(device),
                            std.to(device),
                        )
                        sens = sens.to(device) if sens is not None else None
                        mra_prior = mra_prior.to(device) if mra_prior is not None else None

                        with autocast("cuda", torch.bfloat16, enabled=args.amp):
                            output = model(inp, mas.bool(), mask_type, acc_factor, acq_type, sensitivity_maps=sens, mra_prior=mra_prior)

                        if recon_slab:
                            center_s = recon_num_slices // 2
                            center_t = args.num_frames // 2
                            output = output[:, center_s, center_t]
                            tar = tar[:, center_s, center_t]
                            center_idx = window_idx[:, center_s, center_t]
                            inp = inp[:, center_s, center_t]
                            inp = inp * std[center_idx] + mean[center_idx]
                            output = output * std[center_idx] + mean[center_idx]
                        else:
                            output = output[:, args.num_frames // 2]
                            tar = tar[:, args.num_frames // 2]
                            inp = inp[:, args.num_frames // 2]
                            inp = inp * std[micro_b] + mean[micro_b]
                            output = output * std[micro_b] + mean[micro_b]  # [1, c/1, h, w, 2]
                        output = complex_abs(crop_k_space(output, (final_shape[-2], final_shape[-1])))  # [b, c/1, h, w]
                        tar = complex_abs(crop_k_space(tar, (final_shape[-2], final_shape[-1])))

                        outputs.append(output.data.cpu().numpy())
                        targets.append(tar.data.cpu().numpy())

                    outputs, targets = rearrange_mri_data(
                        [np.vstack(outputs), np.vstack(targets)],
                        args,
                        is_complex=False,
                        reverse=True,
                        num_slices=final_shape[-4],
                        num_coils=final_shape[-3],
                        temporal_shuffle=temporal_shuffle,
                    )  # (time), slice, coil, h, w

                    outputs_rss = np.sqrt(np.sum(outputs[0] ** 2, axis=-3))  # RSS: (time), slice, h, w
                    targets_rss = np.sqrt(np.sum(targets[0] ** 2, axis=-3))  # RSS: (time), slice, h, w

                    outputs_pp = postprocess_mri_recon(
                        outputs_rss,
                        args,
                        file_name,
                        is_training=False,
                        pp_z_score_norm=args.pp_z_score_norm,
                    )  # w, h, slice, (time)
                    targets_pp = postprocess_mri_recon(
                        targets_rss,
                        args,
                        file_name,
                        is_training=False,
                        pp_z_score_norm=args.pp_z_score_norm,
                    )  # w, h, slice, (time)

                    save_img4ranking(
                        outputs_pp,
                        os.path.join(outpath, f"val_img4ranking_epoch{str(epoch+1)}"),
                        file_name.replace(".json", ".mat"),
                    )

                    psnr_array, ssim_array, nmse_array = calmetric(outputs_pp, targets_pp, z_score=args.pp_z_score_norm)
                    val_ssim.append(np.mean(ssim_array))
                    val_nmse.append(np.mean(nmse_array))
                    val_psnr.append(np.mean(psnr_array))

                if args.ddp:
                    # wait for all processes to finish
                    dist.barrier()
                    val_ssim = gather_metric(val_ssim, device, world_size)
                    val_nmse = gather_metric(val_nmse, device, world_size)
                    val_psnr = gather_metric(val_psnr, device, world_size)
                else:
                    val_ssim, val_nmse, val_psnr = (
                        np.mean(val_ssim),
                        np.mean(val_nmse),
                        np.mean(val_psnr),
                    )

                metric = val_ssim

                # save the best checkpoint so far
                if (metric > best_metric) and not args.val:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if rank == 0:
                        save_checkpoint(
                            epoch,
                            global_step,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            outpath,
                            best_metric,
                            best_metric_epoch,
                            model_filename=args.model_filename[:-3] + "_best.pt",
                            epoch_finished=True,
                            is_ddp=args.ddp,
                            wandb_run_id=run.id,
                            python_rng_state=random.getstate(),
                            numpy_rng_state=np.random.get_state(),
                            torch_rng_state=torch.random.get_rng_state(),
                            cuda_rng_state=torch.cuda.get_rng_state(),
                        )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean ssim: {:.4f} best mean ssim: {:.4f} at epoch {} \
                     time elapsed: {:.2f} mins",
                    epoch + 1,
                    metric,
                    best_metric,
                    best_metric_epoch,
                    (time.time() - tic_val) / 60,
                )
                if rank == 0:
                    inp = (
                        inp
                        if args.model_type.lower() not in ["varnet", "kspace_mar"]
                        else ifftn_centered(inp, spatial_dims=2)
                    )
                    inp_vis = crop_k_space(inp, (final_shape[-2], final_shape[-1]))
                    sample_imgs = visualize(
                        inp_vis[:1, 0, ...].detach().cpu(),
                        outputs_rss[:1, 0, ...],
                        targets_rss[:1, 0, ...],
                        epoch + 1,
                        writer,
                    )
                    writer.add_scalar("val_mean_ssim", val_ssim, epoch + 1)
                    writer.add_scalar("val_mean_nmse", val_nmse, epoch + 1)
                    writer.add_scalar("val_mean_psnr", val_psnr, epoch + 1)
                    run.log(
                        {
                            "val/ssim": val_ssim,
                            "val/nmse": val_nmse,
                            "val/psnr": val_psnr,
                            "val/sample_image": [wandb.Image(img) for img in sample_imgs],
                        },
                        step=epoch + 1,
                    )
                if args.ddp:
                    # wait for all processes to finish
                    dist.barrier()
            torch.cuda.empty_cache()
        if args.val:
            break

    print(f"training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    if rank == 0:
        writer.close()

    if args.ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--val",
        default=False,
        action="store_true",
        help="Whether to run validation only",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    config.ddp = is_ddp_enabled()
    config.val = args.val
    config.debug = args.debug
    if config.ddp and ("MASTER_PORT" not in os.environ.keys()):
        port = str(find_free_network_port())
        print(f"using port {port}")
        os.environ["MASTER_PORT"] = port
    trainer(config)


if __name__ == "__main__":
    main()
