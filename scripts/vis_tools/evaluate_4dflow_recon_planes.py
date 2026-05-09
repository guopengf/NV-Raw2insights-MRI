#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.metrics import structural_similarity

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


ACC_RE = re.compile(r"(ktGaussian\d+)")


def read_mat_dict(path: Path) -> dict:
    try:
        with h5py.File(path, "r", swmr=True) as f:
            return {key: f[key][()] for key in f}
    except Exception:
        return {key: val for key, val in scipy.io.loadmat(path).items() if not key.startswith("__")}


def first_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    dat = read_mat_dict(path)
    for key in preferred_keys:
        if key in dat:
            return dat[key]
    for key, val in dat.items():
        if not key.startswith("__"):
            return val
    raise ValueError(f"No array found in {path}")


def to_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(np.complex64)


def ifft1c(x: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(x, axes=axis), axis=axis, norm="ortho"),
        axes=axis,
    )


def ifft2c(x: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(x, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def gt_to_yzxt(gt_mat: Path, enc_idx: int) -> np.ndarray:
    raw = to_complex(first_array(gt_mat, ("kdata_full", "kdata", "kspace_full", "kspace")))
    if raw.ndim != 6:
        raise ValueError(f"Expected GT shape (enc,t,coil,kz,ky,kx), got {raw.shape}")
    if not (0 <= enc_idx < raw.shape[0]):
        raise ValueError(f"enc_idx={enc_idx} out of range for GT shape {raw.shape}")

    raw = raw[enc_idx : enc_idx + 1]
    hybrid = ifft1c(raw, axis=-1)  # (1,t,coil,kz,ky,x)
    img = ifft2c(hybrid, axes=(-3, -2))  # (1,t,coil,z,y,x)
    rss = np.sqrt(np.sum(np.abs(img[0]) ** 2, axis=1)).astype(np.float32)  # (t,z,y,x)
    return np.transpose(rss, (2, 1, 3, 0)).astype(np.float32)  # (y,z,x,t)


def recon_to_yzxt(recon_mat: Path) -> np.ndarray:
    recon = np.asarray(first_array(recon_mat, ("img4ranking", "recon", "reconstruction"))).squeeze()
    if recon.ndim != 4:
        raise ValueError(f"Expected recon shape (y,z,x,t), got {recon.shape} from {recon_mat}")
    return recon.astype(np.float32)


def normalize_for_display(x: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    if vmax is None:
        vmax = float(np.percentile(x, 99.5))
    vmax = max(vmax, 1e-8)
    return np.clip(x / vmax, 0.0, 1.0), vmax


def plane_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    data_range = max(float(gt.max()), float(recon.max())) - min(float(gt.min()), float(recon.min()))
    if data_range <= 0:
        data_range = 1.0
    return float(structural_similarity(gt, recon, data_range=data_range))


def save_comparison(gt: np.ndarray, recon: np.ndarray, out_path: Path, title: str) -> None:
    vmax = float(np.percentile(gt, 99.5))
    gt_vis, vmax = normalize_for_display(gt, vmax)
    recon_vis, _ = normalize_for_display(recon, vmax)
    err_vis, _ = normalize_for_display(np.abs(recon - gt))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, img, name in zip(axes, [gt_vis, recon_vis, err_vis], ["GT", "Recon", "Abs error"]):
        ax.imshow(img, cmap="gray", origin="lower")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def evaluate_recon(
    *,
    gt_yzxt: np.ndarray,
    recon_yzxt: np.ndarray,
    out_case_dir: Path,
    acc_tag: str,
    save_images: bool,
    save_every_xy: int,
    save_every_zy: int,
) -> dict[str, float]:
    if gt_yzxt.shape != recon_yzxt.shape:
        raise ValueError(f"Shape mismatch GT {gt_yzxt.shape} vs recon {recon_yzxt.shape}")

    ny, nz, nx, nt = gt_yzxt.shape
    acc_dir = out_case_dir / acc_tag
    xy_dir = acc_dir / "xy"
    zy_dir = acc_dir / "zy"

    xy_rows = []
    zy_rows = []
    xy_ssims = []
    zy_ssims = []

    for t in range(nt):
        for z in range(nz):
            gt_plane = gt_yzxt[:, z, :, t]      # (y, x)
            recon_plane = recon_yzxt[:, z, :, t]
            ssim_val = plane_ssim(gt_plane, recon_plane)
            xy_ssims.append(ssim_val)
            xy_rows.append({"plane": "xy", "time": t, "z": z, "ssim": ssim_val})
            if save_images and (t % save_every_xy == 0) and (z % save_every_xy == 0):
                save_comparison(
                    gt_plane,
                    recon_plane,
                    xy_dir / f"t{t:03d}_z{z:03d}.png",
                    f"{acc_tag} xy t={t} z={z} SSIM={ssim_val:.4f}",
                )

    for t in range(nt):
        for x in range(nx):
            gt_plane = gt_yzxt[:, :, x, t].T    # (z, y)
            recon_plane = recon_yzxt[:, :, x, t].T
            ssim_val = plane_ssim(gt_plane, recon_plane)
            zy_ssims.append(ssim_val)
            zy_rows.append({"plane": "zy", "time": t, "x": x, "ssim": ssim_val})
            if save_images and (t % save_every_zy == 0) and (x % save_every_zy == 0):
                save_comparison(
                    gt_plane,
                    recon_plane,
                    zy_dir / f"t{t:03d}_x{x:03d}.png",
                    f"{acc_tag} zy t={t} x={x} SSIM={ssim_val:.4f}",
                )

    acc_dir.mkdir(parents=True, exist_ok=True)
    with (acc_dir / "metrics_xy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plane", "time", "z", "ssim"])
        writer.writeheader()
        writer.writerows(xy_rows)
    with (acc_dir / "metrics_zy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plane", "time", "x", "ssim"])
        writer.writeheader()
        writer.writerows(zy_rows)

    return {
        "xy_ssim_mean": float(np.mean(xy_ssims)),
        "xy_ssim_std": float(np.std(xy_ssims)),
        "zy_ssim_mean": float(np.mean(zy_ssims)),
        "zy_ssim_std": float(np.std(zy_ssims)),
    }


def case_name_from_recon(path: Path) -> str:
    parts = path.stem.split("__")
    if len(parts) >= 4:
        return "__".join(parts[:3])
    return path.stem


def acc_tag_from_recon(path: Path) -> str:
    match = ACC_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer ktGaussian tag from filename: {path.name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mat", type=Path, required=True)
    parser.add_argument("--recon-mats", type=Path, nargs="+", required=True)
    parser.add_argument("--enc-idx", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/4dflow_eval_vis"))
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--save-every-xy", type=int, default=1)
    parser.add_argument("--save-every-zy", type=int, default=1)
    args = parser.parse_args()

    out_root = assert_outputs_not_in_data([args.out_root], [args.gt_mat, *args.recon_mats])[0]
    gt_yzxt = gt_to_yzxt(args.gt_mat, args.enc_idx)

    summary_rows = []
    for recon_mat in args.recon_mats:
        recon_yzxt = recon_to_yzxt(recon_mat)
        case_name = case_name_from_recon(recon_mat)
        acc_tag = acc_tag_from_recon(recon_mat)
        metrics = evaluate_recon(
            gt_yzxt=gt_yzxt,
            recon_yzxt=recon_yzxt,
            out_case_dir=out_root / case_name,
            acc_tag=acc_tag,
            save_images=not args.no_images,
            save_every_xy=max(args.save_every_xy, 1),
            save_every_zy=max(args.save_every_zy, 1),
        )
        row = {"case": case_name, "acc": acc_tag, **metrics}
        summary_rows.append(row)
        print(
            f"{case_name} {acc_tag}: "
            f"xy_ssim={metrics['xy_ssim_mean']:.6f}±{metrics['xy_ssim_std']:.6f}, "
            f"zy_ssim={metrics['zy_ssim_mean']:.6f}±{metrics['zy_ssim_std']:.6f}"
        )

    summary_path = out_root / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "acc", "xy_ssim_mean", "xy_ssim_std", "zy_ssim_mean", "zy_ssim_std"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved visualizations under: {out_root}")


if __name__ == "__main__":
    main()
