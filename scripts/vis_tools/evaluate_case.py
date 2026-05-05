#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


GT_PATTERN = re.compile(r"^kdata_full_slice(\d+)_time(\d+)\.png$")
RECON_PATTERN = re.compile(
    r"^Center007__GE_30T_Architect__P076__ktGaussian10_slice(\d+)_time(\d+)\.png$"
)


def load_grayscale_float(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def index_gt_files(folder: Path) -> dict[tuple[int, int], Path]:
    mapping: dict[tuple[int, int], Path] = {}
    for p in sorted(folder.glob("*.png")):
        m = GT_PATTERN.match(p.name)
        if not m:
            continue
        slice_idx = int(m.group(1))
        time_idx = int(m.group(2))
        mapping[(slice_idx, time_idx)] = p
    return mapping


def index_recon_files(folder: Path) -> dict[tuple[int, int], Path]:
    mapping: dict[tuple[int, int], Path] = {}
    for p in sorted(folder.glob("*.png")):
        m = RECON_PATTERN.match(p.name)
        if not m:
            continue
        slice_idx = int(m.group(1))
        time_idx = int(m.group(2))
        mapping[(slice_idx, time_idx)] = p
    return mapping


def safe_psnr(gt: np.ndarray, recon: np.ndarray, data_range: float) -> float:
    if np.array_equal(gt, recon):
        return float("inf")
    return float(peak_signal_noise_ratio(gt, recon, data_range=data_range))


def safe_ssim(gt: np.ndarray, recon: np.ndarray, data_range: float) -> float:
    return float(structural_similarity(gt, recon, data_range=data_range))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Folder containing GT PNGs, e.g. kdata_full_slice000_time000.png",
    )
    parser.add_argument(
        "--recon",
        type=Path,
        required=True,
        help="Folder containing recon/ZF PNGs, e.g. Center007__GE_30T_Architect__P076__ktGaussian10_slice000_time000.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-frame metrics",
    )
    args = parser.parse_args()

    gt_map = index_gt_files(args.gt)
    recon_map = index_recon_files(args.recon)

    if not gt_map:
        raise RuntimeError(f"No GT PNGs matched expected pattern in: {args.gt}")
    if not recon_map:
        raise RuntimeError(f"No recon PNGs matched expected pattern in: {args.recon}")

    common_keys = sorted(set(gt_map) & set(recon_map))
    missing_in_recon = sorted(set(gt_map) - set(recon_map))
    missing_in_gt = sorted(set(recon_map) - set(gt_map))

    print(f"GT matched files: {len(gt_map)}")
    print(f"Recon matched files: {len(recon_map)}")
    print(f"Paired files: {len(common_keys)}")

    if missing_in_recon:
        print(f"Missing in recon: {len(missing_in_recon)}")
        print("First few missing keys in recon:", missing_in_recon[:10])

    if missing_in_gt:
        print(f"Missing in GT: {len(missing_in_gt)}")
        print("First few missing keys in GT:", missing_in_gt[:10])

    if not common_keys:
        raise RuntimeError("No matched (slice, time) pairs found.")

    rows: list[dict[str, object]] = []
    psnrs: list[float] = []
    ssims: list[float] = []

    for key in common_keys:
        gt_path = gt_map[key]
        recon_path = recon_map[key]

        gt = load_grayscale_float(gt_path)
        recon = load_grayscale_float(recon_path)

        if gt.shape != recon.shape:
            raise RuntimeError(
                f"Shape mismatch for key {key}: GT {gt.shape} vs recon {recon.shape}\n"
                f"GT file: {gt_path}\nRecon file: {recon_path}"
            )

        data_range = max(float(gt.max()), float(recon.max())) - min(float(gt.min()), float(recon.min()))
        if data_range <= 0:
            data_range = 255.0

        psnr_val = safe_psnr(gt, recon, data_range=data_range)
        ssim_val = safe_ssim(gt, recon, data_range=data_range)

        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        rows.append(
            {
                "slice": key[0],
                "time": key[1],
                "gt_file": gt_path.name,
                "recon_file": recon_path.name,
                "psnr": psnr_val,
                "ssim": ssim_val,
            }
        )

    finite_psnrs = [x for x in psnrs if math.isfinite(x)]
    mean_psnr = float(np.mean(finite_psnrs)) if finite_psnrs else float("inf")
    mean_ssim = float(np.mean(ssims))

    print("\n=== Results ===")
    print(f"Mean PSNR: {mean_psnr:.6f}")
    print(f"Mean SSIM: {mean_ssim:.6f}")

    if args.csv is not None:
        csv_path = assert_outputs_not_in_data([args.csv], [args.gt, args.recon])[0]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["slice", "time", "gt_file", "recon_file", "psnr", "ssim"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-frame CSV to: {csv_path}")

    print(
        "\nNote: these metrics are computed on the exported PNGs as-is. "
        "If the PNGs were individually normalized during visualization export, "
        "the numbers reflect the rendered images, not the original raw volumes."
    )


if __name__ == "__main__":
    main()
