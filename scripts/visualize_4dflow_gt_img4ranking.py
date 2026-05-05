#!/usr/bin/env python3
import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from path_safety import assert_outputs_not_in_data


def read_mat(mat_file):
    try:
        with h5py.File(mat_file, "r", swmr=True) as f:
            data_kv = [(key, f[key][()]) for key in f]
        print("read by h5py", file=sys.stderr)
    except Exception:
        data = scipy.io.loadmat(mat_file)
        data_kv = [(key, data[key]) for key in data if not key.startswith("__")]
        print("read by scipy", file=sys.stderr)
    return dict(data_kv)


def to_complex_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    raise ValueError(f"Unsupported complex dtype: {arr.dtype}")


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.fftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def rss(img: np.ndarray, coil_axis: int) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=coil_axis))


def normalize_01(arr):
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - vmin) / (vmax - vmin)


def center_crop_4d(img, target_shape):
    """
    img: (x, y, z, t)
    target_shape: (cx, cy, cz, ct)
    """
    sx, sy, sz, st = img.shape
    cx, cy, cz, ct = target_shape

    x0 = max((sx - cx) // 2, 0)
    y0 = max((sy - cy) // 2, 0)
    z0 = max((sz - cz) // 2, 0)
    t0 = 0  # repo keeps first frames

    return img[x0:x0 + cx, y0:y0 + cy, z0:z0 + cz, t0:t0 + ct]


def run4ranking_like_gt(img_xyzt):
    """
    Mimic repo's img4ranking style for a dynamic non-mapping sequence:
    - keep central 2 slices if z >= 3, else keep all slices
    - keep first 3 time frames
    - center crop spatial size to (round(x/3), round(y/2))
    Input/Output layout: (x, y, z, t)
    """
    sx, sy, sz, st = img_xyzt.shape

    cx = int(round(sx / 3.0))
    cy = int(round(sy / 2.0))
    ct = min(3, st)

    if sz < 3:
        cz = sz
        sub = img_xyzt[:, :, :cz, :ct]
    else:
        # central 2 slices, same spirit as repo
        end_z = int(round(sz / 2.0 + 1e-5))
        start_z = max(end_z - 2, 0)
        sub = img_xyzt[:, :, start_z:end_z, :ct]
        cz = sub.shape[2]

    out = center_crop_4d(sub, (cx, cy, cz, ct))
    return out


def visualize_gt_img4ranking_style(
    mat_path,
    encoding_idx=0,
    out_path=None,
    normalize=True,
    cmap="gray",
    show_colorbar=True,
    figsize_per_slice=4.0,
    dpi=120,
):
    d = read_mat(mat_path)
    if "kdata_full" not in d:
        raise KeyError(f"'kdata_full' not found. Keys: {list(d.keys())}")

    raw = to_complex_array(d["kdata_full"])

    # expected: (enc, t, coil, z, ky, kx)
    if raw.ndim != 6:
        raise ValueError(f"Expected shape (enc, t, coil, z, ky, kx), got {raw.shape}")

    if not (0 <= encoding_idx < raw.shape[0]):
        raise ValueError(f"encoding_idx={encoding_idx} out of range for shape {raw.shape}")

    raw = raw[encoding_idx]                 # (t, coil, z, ky, kx)
    raw = np.transpose(raw, (0, 2, 1, 3, 4))  # (t, z, coil, ky, kx)

    img = ifft2c(raw)                      # (t, z, coil, y, x)
    gt = rss(img, coil_axis=2).astype(np.float32)  # (t, z, y, x)

    # repo-style visualization layout: (x, y, z, t)
    vis = np.transpose(gt, (3, 2, 1, 0))  # (x, y, z, t)

    vis = run4ranking_like_gt(vis)

    if normalize:
        vis = vis.astype(np.float64)
        for s in range(vis.shape[2]):
            for t in range(vis.shape[3]):
                vis[:, :, s, t] = normalize_01(vis[:, :, s, t])

    n_slices = vis.shape[2]
    n_times = vis.shape[3]

    fig, axes = plt.subplots(
        n_slices,
        n_times,
        figsize=(figsize_per_slice * n_times, figsize_per_slice * n_slices),
        squeeze=False,
    )

    for s in range(n_slices):
        for t in range(n_times):
            ax = axes[s, t]
            im = ax.imshow(vis[:, :, s, t], cmap=cmap)
            ax.set_title(f"t={t}, slice={s}", fontsize=8)
            ax.axis("off")
            if show_colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"{os.path.basename(mat_path)} | encoding={encoding_idx} | img4ranking-style GT | shape={vis.shape}",
        fontsize=10,
    )
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print("Saved:", out_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path", help="Path to kdata_full.mat")
    parser.add_argument("--encoding-idx", type=int, default=0)
    parser.add_argument("-o", "--output", required=True, help="Output PNG path")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cmap", default="gray")
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--figsize", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    output = assert_outputs_not_in_data([args.output], [args.mat_path])[0]

    visualize_gt_img4ranking_style(
        mat_path=args.mat_path,
        encoding_idx=args.encoding_idx,
        out_path=output,
        normalize=not args.no_normalize,
        cmap=args.cmap,
        show_colorbar=not args.no_colorbar,
        figsize_per_slice=args.figsize,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
