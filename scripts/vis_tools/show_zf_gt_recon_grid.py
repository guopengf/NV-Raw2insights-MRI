#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


ZF_PATTERN = re.compile(r".*?(?:slice|dim2)_(\d+).*(?:time|dim3)_(\d+)\.png$")
GT_PATTERN = re.compile(r".*?(?:slice|dim2)_(\d+).*(?:time|dim3)_(\d+)\.png$")
RECON_PATTERN = re.compile(r".*?(?:slice|dim2)_(\d+).*(?:time|dim3)_(\d+)\.png$")


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def build_index(folder: Path, pattern: re.Pattern) -> dict[tuple[int, int], Path]:
    out: dict[tuple[int, int], Path] = {}
    for p in sorted(folder.glob("*.png")):
        m = pattern.match(p.name)
        if not m:
            continue
        slice_idx = int(m.group(1))
        time_idx = int(m.group(2))
        out[(slice_idx, time_idx)] = p
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zf", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["0,0", "1,0"],
        help='Pairs to show as "slice,time", default: 0,0 1,0',
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    out = assert_outputs_not_in_data([args.out], [args.zf, args.gt, args.recon])[0]

    zf_map = build_index(args.zf, ZF_PATTERN)
    gt_map = build_index(args.gt, GT_PATTERN)
    recon_map = build_index(args.recon, RECON_PATTERN)

    pairs: list[tuple[int, int]] = []
    for s in args.pairs:
        a, b = s.split(",")
        pairs.append((int(a), int(b)))

    if len(pairs) != 2:
        raise ValueError("Please provide exactly 2 pairs so the figure is 2 rows x 3 cols.")

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    col_titles = ["ZF", "GT", "Recon"]

    for r, key in enumerate(pairs):
        if key not in zf_map:
            raise FileNotFoundError(f"ZF pair not found: {key}")
        if key not in gt_map:
            raise FileNotFoundError(f"GT pair not found: {key}")
        if key not in recon_map:
            raise FileNotFoundError(f"Recon pair not found: {key}")

        imgs = [
            load_gray(zf_map[key]),
            load_gray(gt_map[key]),
            load_gray(recon_map[key]),
        ]

        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray")
            ax.axis("off")

            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)

            if c == 0:
                ax.set_ylabel(f"slice {key[0]}, time {key[1]}", fontsize=10)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
