#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


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


def to_complex_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    raise ValueError(f"Unsupported complex dtype: {arr.dtype}")


def ifft1c_kx(kspace: np.ndarray) -> np.ndarray:
    """
    (enc, t, coil, kz, ky, kx) -> (enc, t, coil, kz, ky, x)
    """
    return np.fft.ifftshift(
        np.fft.ifft(
            np.fft.fftshift(kspace, axes=(-1,)),
            axis=-1,
            norm="ortho",
        ),
        axes=(-1,),
    )


def ifft2c_kzky(hybrid: np.ndarray) -> np.ndarray:
    """
    (enc, t, coil, kz, ky, x) -> (enc, t, coil, z, y, x)
    """
    return np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.fftshift(hybrid, axes=(-3, -2)),
            axes=(-3, -2),
            norm="ortho",
        ),
        axes=(-3, -2),
    )


def normalize_01(arr):
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin == 0:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def save_gt_frames(mat_path, out_dir, key=None, normalize=True, cmap="gray", dpi=150):
    out_dir = assert_outputs_not_in_data([out_dir], [mat_path])[0]
    d = read_mat(mat_path)

    if key is None:
        for candidate in ["kdata_full", "kdata", "kspace", "kus", "kdata_ktGaussian"]:
            if candidate in d:
                key = candidate
                break
    if key is None or key not in d:
        raise KeyError(f"Could not find k-space key. Available keys: {list(d.keys())}")

    raw = to_complex_array(d[key])

    if raw.ndim != 6:
        raise ValueError(f"Expected raw shape (enc, t, coil, kz, ky, kx), got {raw.shape}")

    print("raw shape:", raw.shape, raw.dtype)

    # Step 1: kx -> x
    hybrid = ifft1c_kx(raw)  # (enc, t, coil, kz, ky, x)

    # Step 2: kz, ky -> z, y
    img = ifft2c_kzky(hybrid)  # (enc, t, coil, z, y, x)

    # Step 3: magnitude + RSS over coil
    img_mag = np.abs(img)
    img_rss = np.sqrt(np.sum(img_mag ** 2, axis=2)).astype(np.float32)  # (enc, t, z, y, x)

    # Step 4: merge enc and t -> (enc*t, z, y, x)
    n_enc, nt, nz, ny, nx = img_rss.shape
    img_rss = img_rss.reshape(n_enc * nt, nz, ny, nx)

    print("img_rss shape:", img_rss.shape, "interpreted as (time, z, y, x)")

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(mat_path).stem

    # save one image per (z, time), image plane = (y, x)
    for t_idx in range(img_rss.shape[0]):
        for z_idx in range(img_rss.shape[1]):
            frame = img_rss[t_idx, z_idx, :, :]  # (y, x)

            if normalize:
                frame = normalize_01(frame)

            out_path = out_dir / f"{stem}_slice{z_idx:03d}_time{t_idx:03d}.png"
            plt.figure(figsize=(5, 5))
            plt.imshow(frame, cmap=cmap, origin="lower")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close()

    print("saved to:", out_dir)
    print(f"saved {img_rss.shape[0] * img_rss.shape[1]} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path")
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("-k", "--key", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--cmap", default="gray")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    save_gt_frames(
        mat_path=args.mat_path,
        out_dir=args.out_dir,
        key=args.key,
        normalize=not args.no_normalize,
        cmap=args.cmap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
