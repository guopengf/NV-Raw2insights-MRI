#!/usr/bin/env python3
"""Render an aligned before/after example from the production Windowed HDF5 dataset."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils import load_config  # noqa: E402
from windowed_4dflow import Windowed4DFlowDataset  # noqa: E402


def _coil_combine(image_ri: torch.Tensor, sensitivity_ri: torch.Tensor) -> torch.Tensor:
    image = torch.view_as_complex(image_ri.contiguous())
    sensitivity = torch.view_as_complex(sensitivity_ri.contiguous())
    return (image * sensitivity.conj()).sum(dim=1) / sensitivity.abs().square().sum(dim=1).clamp_min(1e-8)


def _extract(sample):
    input_image = sample["kspace_masked_ifft"] * sample["std"] + sample["mean"]
    target_image = sample["kspace_ifft"]
    sensitivity = sample["sensitivity_maps"]
    slice_idx = target_image.shape[2] // 2
    frame_idx = target_image.shape[3] // 2
    target = _coil_combine(target_image[0, :, slice_idx, frame_idx], sensitivity[0, :, slice_idx, frame_idx])
    masked = _coil_combine(input_image[0, :, slice_idx, frame_idx], sensitivity[0, :, slice_idx, frame_idx])
    phase_difference = torch.angle(target[1] * target[0].conj())
    segmask = sample["joint_segmask"][0, slice_idx]
    return {
        "target": target[0].abs().cpu().numpy(),
        "input": masked[0].abs().cpu().numpy(),
        "phase": phase_difference.cpu().numpy(),
        "segmask": segmask.cpu().numpy(),
    }


def _show_row(axes, values, label, target_vmax, input_vmax):
    axes[0].imshow(values["target"], cmap="gray", vmin=0, vmax=target_vmax, aspect="auto")
    axes[0].set_title(f"{label}: target magnitude")
    axes[1].imshow(values["input"], cmap="gray", vmin=0, vmax=input_vmax, aspect="auto")
    axes[1].set_title(f"{label}: input magnitude")
    axes[2].imshow(values["phase"], cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="auto")
    axes[2].set_title(f"{label}: phase E1-E0")
    axes[3].imshow(values["target"], cmap="gray", vmin=0, vmax=target_vmax, aspect="auto")
    axes[3].contour(values["segmask"] > 0.5, levels=[0.5], colors=["#ff334f"], linewidths=1.2)
    axes[3].set_title(f"{label}: segmentation alignment")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    augmented_config = load_config(args.config)
    augmented_config.num_samples_per_case = 1
    augmented_config.four_dflow_augmentation.flip.prob = 1.0
    augmented_config.four_dflow_augmentation.shift.prob = 1.0
    augmented_config.four_dflow_augmentation.contrast.prob = 1.0
    baseline_config = copy.deepcopy(augmented_config)
    baseline_config.data_aug = False

    manifest = json.loads(args.manifest.read_text())
    with h5py.File(manifest["windowed_h5"], "r", swmr=True) as store:
        frames, slices = store["hybrid/target"].shape[:2]
    center = (frames // 2) * slices + slices // 2
    selector = lambda total, count: np.asarray([center], dtype=np.int64)

    baseline = Windowed4DFlowDataset(
        [{"kspace": args.manifest}], baseline_config, center_selector=selector
    )[0]
    torch.manual_seed(args.seed)
    augmented = Windowed4DFlowDataset(
        [{"kspace": args.manifest}], augmented_config, center_selector=selector
    )[0]

    before = _extract(baseline)
    after = _extract(augmented)
    target_vmax = float(np.percentile(np.concatenate([before["target"].ravel(), after["target"].ravel()]), 99.5))
    input_vmax = float(np.percentile(np.concatenate([before["input"].ravel(), after["input"].ravel()]), 99.5))

    figure, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    _show_row(axes[0], before, "Before", target_vmax, input_vmax)
    _show_row(axes[1], after, "After", target_vmax, input_vmax)
    params = augmented["kspace_meta_dict"]["four_dflow_augmentation"]
    figure.suptitle(
        "Online 4D-flow augmentation | " + ", ".join(f"{key}={value}" for key, value in params.items()),
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "center": center,
                "params": params,
                "output": str(args.output),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "complete", "output": str(args.output), "params": params}, sort_keys=True))


if __name__ == "__main__":
    main()
