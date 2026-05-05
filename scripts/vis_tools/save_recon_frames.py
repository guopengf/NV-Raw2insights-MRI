#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_not_in_known_raw_data_path


def read_mat(mat_file):
    try:
        with h5py.File(mat_file, "r", swmr=True) as f:
            data_kv = [(key, f[key][()]) for key in f]
        print("read by h5py")
    except Exception:
        data = scipy.io.loadmat(mat_file)
        data_kv = [(key, data[key]) for key in data if not key.startswith("__")]
        print("read by scipy")
    return dict(data_kv)


def normalize_01(arr):
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin == 0:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def save_frames(mat_path, out_dir, key="img4ranking", normalize=True, cmap="gray", dpi=150):
    out_dir = assert_not_in_known_raw_data_path(out_dir, what="output directory")
    d = read_mat(mat_path)
    if key not in d:
        raise KeyError(f"Key '{key}' not found. Available keys: {list(d.keys())}")

    img = np.asarray(d[key])
    img = np.squeeze(img)

    print("raw shape:", img.shape, img.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(mat_path).stem

    # Assumed 4D saved layout:
    # (dim0, dim1, dim2, dim3) = (y, z, x, time)
    # Visualization target:
    # image plane = (x, y), slice = z
    if img.ndim == 4:
        ny, nz, nx, nt = img.shape
        print(f"interpreted as (y, z, x, time)=({ny}, {nz}, {nx}, {nt})")

        for z in range(nz):
            for t in range(nt):
                frame = img[:, z, :, t]   # (y, x)

                if normalize:
                    frame = normalize_01(frame)

                out_path = out_dir / f"{stem}_slice{z:03d}_time{t:03d}.png"
                plt.figure(figsize=(5, 5))
                plt.imshow(frame, cmap=cmap, origin="lower")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
                plt.close()

    # Assumed 3D saved layout:
    # (dim0, dim1, dim2) = (y, z, x)
    # Visualization target:
    # image plane = (x, y), slice = z
    elif img.ndim == 3:
        ny, nz, nx = img.shape
        print(f"interpreted as (y, z, x)=({ny}, {nz}, {nx})")

        for z in range(nz):
            frame = img[:, z, :]   # (y, x)

            if normalize:
                frame = normalize_01(frame)

            out_path = out_dir / f"{stem}_slice{z:03d}.png"
            plt.figure(figsize=(5, 5))
            plt.imshow(frame, cmap=cmap, origin="lower")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close()

    elif img.ndim == 2:
        print(f"interpreted as (y, x)={img.shape}")
        frame = normalize_01(img) if normalize else img
        out_path = out_dir / f"{stem}.png"
        plt.figure(figsize=(5, 5))
        plt.imshow(frame, cmap=cmap, origin="lower")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close()

    else:
        raise ValueError(f"Unsupported img ndim: {img.ndim}, shape={img.shape}")

    print("saved to:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path")
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("-k", "--key", default="img4ranking")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cmap", default="gray")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    save_frames(
        mat_path=args.mat_path,
        out_dir=args.out_dir,
        key=args.key,
        normalize=not args.no_normalize,
        cmap=args.cmap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
