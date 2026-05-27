#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from scipy.ndimage import gaussian_filter


ENCODING_LABELS = {
    0: "reference",
    1: "RL",
    2: "AP",
    3: "HF",
}


def read_mat_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            for key in preferred_keys:
                if key in f:
                    return to_complex_np(f[key][()])
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    return to_complex_np(f[key][()])
    except OSError:
        dat = scipy.io.loadmat(path)
        for key in preferred_keys:
            if key in dat:
                return to_complex_np(dat[key])
        for key, value in dat.items():
            if not key.startswith("__"):
                return to_complex_np(value)
    raise ValueError(f"No array found in {path}")


def to_complex_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    return arr


def h5_to_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.fields is not None and "real" in x.dtype.fields and "imag" in x.dtype.fields:
        return (x["real"] + 1j * x["imag"]).astype(np.complex64, copy=False)
    if np.iscomplexobj(x):
        return x.astype(np.complex64, copy=False)
    if x.ndim > 0 and x.shape[-1] == 2 and np.issubdtype(x.dtype, np.floating):
        return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported complex dtype: {x.dtype}")


def temporal_mean_kspace(path: Path, key: str, *, device: torch.device, nonzero: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Missing key '{key}' in {path}")
        dset = f[key]
        if len(dset.shape) != 6:
            raise ValueError(f"Expected k-space shape (enc,t,coil,kz,ky,kx), got {dset.shape} in {path}")
        nv, nt, nc, nz, ny, nx = dset.shape
        out = torch.zeros((nv, nc, nz, ny, nx), dtype=torch.complex64, device=device)
        cnt = torch.zeros((nv, nc, nz, ny, nx), dtype=torch.int32, device=device) if nonzero else None
        for t in range(nt):
            frame = torch.as_tensor(h5_to_complex(dset[:, t]), dtype=torch.complex64, device=device)
            out += frame
            if nonzero:
                cnt += frame != 0
        if nonzero:
            mask = cnt > 0
            out = torch.where(mask, out / cnt.clamp_min(1), torch.zeros_like(out))
        else:
            mask = torch.ones_like(out, dtype=torch.bool)
            out /= nt
    return out, mask


def k2i_torch(x: torch.Tensor, dim: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def i2k_torch(x: torch.Tensor, dim: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def direct_recon(kmean: torch.Tensor, coilmap: torch.Tensor) -> torch.Tensor:
    imgc = k2i_torch(kmean, dim=(-1, -2, -3))  # (enc, coil, z, y, x)
    num = torch.sum(imgc * torch.conj(coilmap).unsqueeze(0), dim=1)
    den = torch.sum(torch.abs(coilmap) ** 2, dim=0) + 1e-8
    return (num / den.unsqueeze(0)).to(torch.complex64)  # (enc, z, y, x)


def sense_recon(
    kdata: torch.Tensor,
    mask: torch.Tensor,
    coilmap: torch.Tensor,
    *,
    lam: float,
    niter: int,
) -> torch.Tensor:
    def A(x: torch.Tensor) -> torch.Tensor:
        return mask * i2k_torch(coilmap * x.unsqueeze(0), dim=(-1, -2, -3))

    def AH(y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.conj(coilmap) * k2i_torch(mask * y, dim=(-1, -2, -3)), dim=0)

    b = AH(kdata)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.real(torch.vdot(r.reshape(-1), r.reshape(-1)))
    for _ in range(niter):
        Ap = AH(A(p)) + lam * p
        denom = torch.real(torch.vdot(p.reshape(-1), Ap.reshape(-1))).clamp_min(1e-12)
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.real(torch.vdot(r.reshape(-1), r.reshape(-1)))
        if rs_new < 1e-12:
            break
        p = r + (rs_new / rs.clamp_min(1e-12)) * p
        rs = rs_new
    return x.to(torch.complex64)


def pcmra_from_img(img: torch.Tensor) -> dict[str, torch.Tensor]:
    if img.shape[0] < 4:
        raise ValueError(f"Expected enc dimension >=4 for reference + RL/AP/HF, got {img.shape}")
    mag = torch.mean(torch.abs(img), dim=0)
    phase_diff = torch.angle(img[1:4] * torch.conj(img[0:1]))
    flow = torch.sqrt(torch.sum(phase_diff**2, dim=0))
    pc_mra = mag * flow
    return {
        "pc_mra": pc_mra.float(),
        "mag": mag.float(),
        "flow": flow.float(),
        "phase_diff": phase_diff.float(),
    }


def make_vessel_map(
    pc_mra: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    smooth_sigma: float,
    threshold: float | None,
) -> np.ndarray:
    lo = float(np.percentile(pc_mra, lower_percentile))
    hi = float(np.percentile(pc_mra, upper_percentile))
    vessel = np.clip((pc_mra.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if smooth_sigma > 0:
        vessel = gaussian_filter(vessel, sigma=smooth_sigma)
        vessel = np.clip(vessel, 0.0, 1.0)
    if threshold is not None:
        vessel = (vessel >= threshold).astype(np.float32)
    return vessel.astype(np.float32)


def to_numpy_dict(prefix: str, result: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {f"{name}_{prefix}": value.detach().cpu().numpy().astype(np.float32) for name, value in result.items()}


def save_slice_grid(path: Path, images: dict[str, np.ndarray], acceleration: int, vessel_map: np.ndarray) -> None:
    first = next(iter(images.values()))
    nz = first.shape[0]
    mid = max(nz // 2 - 3, 0)
    indices = sorted(set([max(mid - 4, 0), mid, min(mid + 4, nz - 1)]))
    vmax = float(np.percentile(np.stack(list(images.values())), 99.5))
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(len(indices) + 1, len(images) + 1, figsize=(4.2 * (len(images) + 1), 4 * (len(indices) + 1)))
    if axes.ndim == 1:
        axes = axes[None, :]

    for row, z in enumerate(indices):
        for col, (name, img) in enumerate(images.items()):
            axes[row, col].imshow(img[z].T, cmap="gray", vmax=vmax, origin="lower")
            axes[row, col].set_title(f"{name} R={acceleration} z={z}")
            axes[row, col].axis("off")
        axes[row, len(images)].imshow(vessel_map[z].T, cmap="magma", vmin=0, vmax=1, origin="lower")
        axes[row, len(images)].set_title(f"vessel_map z={z}")
        axes[row, len(images)].axis("off")

    row = len(indices)
    for col, (name, img) in enumerate(images.items()):
        axes[row, col].imshow(np.max(img, axis=2), cmap="gray", vmax=vmax, origin="lower")
        axes[row, col].set_title(f"{name} x-MIP (zy)")
        axes[row, col].axis("off")
    vessel_2d = np.max(vessel_map, axis=2) if vessel_map.ndim == 3 else vessel_map
    axes[row, len(images)].imshow(vessel_2d, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[row, len(images)].set_title("vessel_map x-MIP (zy)")
    axes[row, len(images)].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    case_dir = args.case_dir
    gt_kspace = args.gt_kspace or (case_dir / "kdata_full.mat")
    us_kspace = args.us_kspace or (case_dir / f"kdata_ktGaussian{args.acceleration}.mat")
    coilmap = args.coilmap or (case_dir / "coilmap.mat")
    return gt_kspace, us_kspace, coilmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-case PC-MRA and vessel map from 4D Flow GT/US k-space.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("/SSDHome/share/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P088"),
    )
    parser.add_argument("--gt-kspace", type=Path)
    parser.add_argument("--us-kspace", type=Path)
    parser.add_argument("--coilmap", type=Path)
    parser.add_argument("--source", choices=("gt", "zf"), default="gt")
    parser.add_argument("--acceleration", type=int, default=50)
    parser.add_argument("--sense-niter", type=int, default=5)
    parser.add_argument("--sense-lam", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("/SSDHome/share/haosen/4dflow/MRAtest"))
    parser.add_argument("--vessel-lower-percentile", type=float, default=1.0)
    parser.add_argument("--vessel-upper-percentile", type=float, default=99.5)
    parser.add_argument("--vessel-smooth-sigma", type=float, default=0.75)
    parser.add_argument("--vessel-threshold", type=float, default=None)
    parser.add_argument("--save-debug", action="store_true", help="Also save GT/ZF/SENSE comparison regardless of source.")
    args = parser.parse_args()

    gt_kspace, us_kspace, coilmap_path = resolve_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"case_dir: {args.case_dir}")
    print(f"source: {args.source}")
    print("reference enc = 0")
    print("velocity encodings = [RL, AP, HF]")
    print(f"acceleration = {args.acceleration}x")
    print(f"device: {device}")

    coilmap_np = read_mat_array(coilmap_path, ("coilmap", "csm", "sensitivity_maps", "sens_maps"))
    coilmap = torch.as_tensor(np.asarray(coilmap_np, dtype=np.complex64), device=device)
    print(f"coilmap shape: {tuple(coilmap.shape)}")

    results_np: dict[str, np.ndarray] = {}
    display_images: dict[str, np.ndarray] = {}

    if args.source == "gt" or args.save_debug:
        kmean_full, _ = temporal_mean_kspace(gt_kspace, "kdata_full", device=device, nonzero=False)
        print(f"kmean_full shape: {tuple(kmean_full.shape)}")
        full = pcmra_from_img(direct_recon(kmean_full, coilmap))
        results_np.update(to_numpy_dict("full_ifft", full))
        results_np["pc_mra_full_ifft"] = results_np.pop("pc_mra_full_ifft")
        display_images["GT full IFFT"] = results_np["pc_mra_full_ifft"]

    if args.source == "zf" or args.save_debug:
        kmean_us, mask_us = temporal_mean_kspace(us_kspace, "kdata_ktGaussian", device=device, nonzero=True)
        print(f"kmean_us shape: {tuple(kmean_us.shape)}")
        us_ifft = pcmra_from_img(direct_recon(kmean_us, coilmap))
        results_np.update(to_numpy_dict("us_ifft", us_ifft))
        results_np["pc_mra_us_ifft"] = results_np.pop("pc_mra_us_ifft")
        display_images["ZF direct IFFT"] = results_np["pc_mra_us_ifft"]

        img_us_sense = torch.stack(
            [
                sense_recon(kmean_us[enc], mask_us[enc], coilmap, lam=args.sense_lam, niter=args.sense_niter)
                for enc in range(kmean_us.shape[0])
            ],
            dim=0,
        )
        us_sense = pcmra_from_img(img_us_sense)
        results_np.update(to_numpy_dict("us_sense", us_sense))
        results_np["pc_mra_us_sense"] = results_np.pop("pc_mra_us_sense")
        display_images["CG-SENSE"] = results_np["pc_mra_us_sense"]

    if args.source == "gt":
        pc_mra_for_prior = results_np["pc_mra_full_ifft"]
        prior_name = "pc_mra_full_ifft"
    else:
        pc_mra_for_prior = results_np["pc_mra_us_sense"]
        prior_name = "pc_mra_us_sense"

    vessel_map = make_vessel_map(
        pc_mra_for_prior,
        lower_percentile=args.vessel_lower_percentile,
        upper_percentile=args.vessel_upper_percentile,
        smooth_sigma=args.vessel_smooth_sigma,
        threshold=args.vessel_threshold,
    )
    results_np["vessel_map"] = vessel_map
    results_np["mra_x_mip"] = np.max(pc_mra_for_prior, axis=2).astype(np.float32)
    results_np["vessel_map_2d"] = np.max(vessel_map, axis=2).astype(np.float32)

    metadata = {
        "case_dir": str(args.case_dir),
        "source": args.source,
        "prior_from": prior_name,
        "projection": "x-MIP; raw x is the slice axis in the current 4D Flow pipeline, leaving a zy prior",
        "acceleration": args.acceleration,
        "reference_enc": 0,
        "velocity_encodings": {"enc1": "RL", "enc2": "AP", "enc3": "HF"},
        "sense_niter": args.sense_niter,
        "sense_lam": args.sense_lam,
        "vessel_lower_percentile": args.vessel_lower_percentile,
        "vessel_upper_percentile": args.vessel_upper_percentile,
        "vessel_smooth_sigma": args.vessel_smooth_sigma,
        "vessel_threshold": args.vessel_threshold,
        "phase3_config_note": {
            "phase3.enable_vaa": "If false, future model should bypass VAA and should not generate/load MRA.",
            "phase3.mra.source": "gt uses GT PC-MRA; zf uses undersampled CG-SENSE PC-MRA by default.",
        },
    }

    for key, value in results_np.items():
        print(f"{key}: shape={value.shape}, min={float(value.min()):.6g}, max={float(value.max()):.6g}")

    scipy.io.savemat(args.out_dir / "pc_mra_single_case.mat", {**results_np, "metadata_json": json.dumps(metadata)})
    np.savez_compressed(args.out_dir / "pc_mra_single_case.npz", **results_np)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_slice_grid(args.out_dir / "pc_mra_comparison.png", display_images, args.acceleration, vessel_map)

    print(f"Saved MAT: {args.out_dir / 'pc_mra_single_case.mat'}")
    print(f"Saved NPZ: {args.out_dir / 'pc_mra_single_case.npz'}")
    print(f"Saved PNG: {args.out_dir / 'pc_mra_comparison.png'}")
    print(f"Saved metadata: {args.out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
