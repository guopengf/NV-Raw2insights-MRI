#!/usr/bin/env python3
"""Visualize and compare 4D Flow kt-Gaussian masks.

Expected challenge mask layout:

    (1, Nt, 1, SPE, PE, 1)

The script converts each mask to (Nt, SPE, PE), then saves:

    <label>_frames.png      binary mask frames
    <label>_time_sum.png    sum over time
    compare_diff_frames.png if two masks are provided
    compare_stats.csv       per-frame sample counts and overlap
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


MASK_KEYS = ("usmask_ktGaussian", "mask", "usmask", "sampling_mask")


def read_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        with h5py.File(path, "r", swmr=True) as f:
            data = {key: f[key][()] for key in f.keys()}
        print(f"{path}: read by h5py")
        return data
    except Exception:
        data = {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}
        print(f"{path}: read by scipy")
        return data


def pick_array(data: dict[str, np.ndarray], key: str | None) -> tuple[str, np.ndarray]:
    if key is not None:
        if key not in data:
            raise KeyError(f"key {key!r} not found. Available keys: {list(data)}")
        return key, data[key]

    for candidate in MASK_KEYS:
        if candidate in data:
            return candidate, data[candidate]
    if len(data) == 1:
        only_key = next(iter(data))
        return only_key, data[only_key]
    raise KeyError(f"could not infer mask key. Available keys: {list(data)}")


def mask_to_t_spe_pe(mask: np.ndarray, layout: str) -> np.ndarray:
    mask = np.asarray(mask)
    mask = np.abs(mask)

    if layout == "challenge":
        if mask.ndim != 6:
            raise ValueError(f"--layout challenge expects 6D (1,Nt,1,SPE,PE,1), got {mask.shape}")
        mask = mask[0, :, 0, :, :, 0]
    elif layout == "auto":
        squeezed = np.squeeze(mask)
        if mask.ndim == 6:
            mask = mask[0, :, 0, :, :, 0]
        elif squeezed.ndim == 2:
            mask = squeezed[None, :, :]
        elif squeezed.ndim == 3:
            # Most saved challenge masks become (Nt, SPE, PE) after squeezing.
            if squeezed.shape[0] <= 80:
                mask = squeezed
            else:
                mask = np.transpose(squeezed, (2, 0, 1))
        else:
            raise ValueError(f"could not auto-convert mask with shape {mask.shape}, squeezed={squeezed.shape}")
    elif layout == "t_spe_pe":
        mask = np.squeeze(mask)
    elif layout == "spe_pe_t":
        mask = np.transpose(np.squeeze(mask), (2, 0, 1))
    elif layout == "pe_spe_t":
        mask = np.transpose(np.squeeze(mask), (2, 1, 0))
    else:
        raise ValueError(f"unknown layout: {layout}")

    if mask.ndim != 3:
        raise ValueError(f"converted mask must be 3D (Nt,SPE,PE), got {mask.shape}")
    return (mask > 0).astype(np.float32)


def parse_frames(value: str, nt: int) -> list[int]:
    if value == "all":
        return list(range(nt))

    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(x) if x else None for x in part.split(":")]
            while len(pieces) < 3:
                pieces.append(None)
            start = 0 if pieces[0] is None else pieces[0]
            stop = nt if pieces[1] is None else pieces[1]
            step = 1 if pieces[2] is None else pieces[2]
            frames.extend(range(start, stop, step))
        else:
            frames.append(int(part))

    bad = [idx for idx in frames if idx < 0 or idx >= nt]
    if bad:
        raise ValueError(f"frame index out of range for Nt={nt}: {bad}")
    return sorted(dict.fromkeys(frames))


def save_frame_grid(mask: np.ndarray, frames: list[int], out_path: Path, title: str, max_cols: int, dpi: int) -> None:
    n_cols = min(max_cols, len(frames))
    n_rows = int(math.ceil(len(frames) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, frame_idx in zip(axes.ravel(), frames):
        ax.imshow(mask[frame_idx], cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"t={frame_idx} n={int(mask[frame_idx].sum())}", fontsize=8)
    fig.suptitle(f"{title} frames | shape={mask.shape}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_time_sum(mask: np.ndarray, out_path: Path, title: str, dpi: int) -> None:
    summed = mask.sum(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(summed, cmap="magma", origin="lower", interpolation="nearest")
    ax.set_title(f"{title} time sum | min={summed.min():.0f} max={summed.max():.0f}")
    ax.set_xlabel("PE")
    ax.set_ylabel("SPE")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_compare_grid(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    frames: list[int],
    label_a: str,
    label_b: str,
    out_path: Path,
    dpi: int,
) -> None:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"cannot compare masks with different shapes: {mask_a.shape} vs {mask_b.shape}")

    n_rows = len(frames)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 2.6 * n_rows), squeeze=False)
    for row, frame_idx in enumerate(frames):
        a = mask_a[frame_idx]
        b = mask_b[frame_idx]
        diff = b - a
        overlap = np.logical_and(a > 0, b > 0).sum()
        union = np.logical_or(a > 0, b > 0).sum()
        iou = overlap / union if union else 1.0

        axes[row, 0].imshow(a, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label_a} t={frame_idx} n={int(a.sum())}", fontsize=8)
        axes[row, 1].imshow(b, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{label_b} t={frame_idx} n={int(b.sum())}", fontsize=8)
        axes[row, 2].imshow(diff, cmap="bwr", origin="lower", interpolation="nearest", vmin=-1, vmax=1)
        axes[row, 2].set_title(f"{label_b}-{label_a} IoU={iou:.3f}", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_stats(mask_a: np.ndarray, mask_b: np.ndarray | None, labels: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if mask_b is None:
            writer.writerow(["frame", f"{labels[0]}_points", f"{labels[0]}_effective_R"])
            _, spe, pe = mask_a.shape
            for idx in range(mask_a.shape[0]):
                points = int(mask_a[idx].sum())
                writer.writerow([idx, points, (spe * pe) / max(points, 1)])
        else:
            writer.writerow(
                [
                    "frame",
                    f"{labels[0]}_points",
                    f"{labels[1]}_points",
                    "overlap",
                    "union",
                    "iou",
                    f"{labels[0]}_effective_R",
                    f"{labels[1]}_effective_R",
                ]
            )
            _, spe, pe = mask_a.shape
            for idx in range(mask_a.shape[0]):
                a = mask_a[idx] > 0
                b = mask_b[idx] > 0
                overlap = int(np.logical_and(a, b).sum())
                union = int(np.logical_or(a, b).sum())
                a_points = int(a.sum())
                b_points = int(b.sum())
                writer.writerow(
                    [
                        idx,
                        a_points,
                        b_points,
                        overlap,
                        union,
                        overlap / union if union else 1.0,
                        (spe * pe) / max(a_points, 1),
                        (spe * pe) / max(b_points, 1),
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("masks", nargs="+", type=Path, help="one or two mask .mat files")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("-k", "--key", default=None, help="MAT key; default auto-detect")
    parser.add_argument("--labels", default=None, help="comma-separated labels matching input masks")
    parser.add_argument(
        "--layout",
        choices=("auto", "challenge", "t_spe_pe", "spe_pe_t", "pe_spe_t"),
        default="auto",
    )
    parser.add_argument("--frames", default="all", help="all, comma list, or Python-like ranges, e.g. 0,5,10:19:2")
    parser.add_argument("--max-cols", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if not (1 <= len(args.masks) <= 2):
        raise ValueError("pass one mask for visualization or two masks for comparison")

    out_dir = assert_outputs_not_in_data([args.out_dir], args.masks)[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels.split(",") if args.labels else [path.stem for path in args.masks]
    if len(labels) != len(args.masks):
        raise ValueError("--labels must have the same count as input masks")

    loaded = []
    for path, label in zip(args.masks, labels):
        key, arr = pick_array(read_mat(path), args.key)
        mask = mask_to_t_spe_pe(arr, args.layout)
        loaded.append(mask)
        print(f"{label}: key={key}, raw_shape={np.asarray(arr).shape}, converted_shape={mask.shape}")

    frames = parse_frames(args.frames, loaded[0].shape[0])
    if len(frames) > 20:
        print(f"warning: plotting {len(frames)} frames; use --frames to select fewer if the PNG is too large")

    for label, mask in zip(labels, loaded):
        save_frame_grid(mask, frames, out_dir / f"{label}_frames.png", label, args.max_cols, args.dpi)
        save_time_sum(mask, out_dir / f"{label}_time_sum.png", label, args.dpi)

    if len(loaded) == 2:
        save_compare_grid(
            loaded[0],
            loaded[1],
            frames,
            labels[0],
            labels[1],
            out_dir / "compare_diff_frames.png",
            args.dpi,
        )
        save_stats(loaded[0], loaded[1], labels, out_dir / "compare_stats.csv")
    else:
        save_stats(loaded[0], None, labels, out_dir / f"{labels[0]}_stats.csv")

    print(f"saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
