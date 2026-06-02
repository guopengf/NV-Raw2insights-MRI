#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from skimage.metrics import structural_similarity

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data


RECON_RE = re.compile(r"kdata_(ktGaussian\d+)_enc(\d+)_recon\.mat$")


def read_mat_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    try:
        with h5py.File(path, "r", swmr=True) as f:
            for key in preferred_keys:
                if key in f:
                    return f[key][()]
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    return f[key][()]
    except OSError:
        dat = scipy.io.loadmat(path)
        for key in preferred_keys:
            if key in dat:
                return dat[key]
        for key, value in dat.items():
            if not key.startswith("__"):
                return value
    raise ValueError(f"No array found in {path}")


def to_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    return arr.astype(np.float32, copy=False)


def ifftnc(x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def temporal_mean_kspace(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as f:
        dset = f[key]
        if len(dset.shape) != 6:
            raise ValueError(f"Expected 6D k-space in {path}, got {dset.shape}")
        nv, nt, nc, nz, ny, nx = dset.shape
        out = np.zeros((nv, nc, nz, ny, nx), dtype=np.complex64)
        for t in range(nt):
            out += to_complex(dset[:, t])
        out /= nt
    return out


def gt_pcmra_and_mag(gt_mat: Path, coilmap_mat: Path) -> tuple[np.ndarray, np.ndarray]:
    coilmap = to_complex(read_mat_array(coilmap_mat, ("coilmap", "csm", "sensitivity_maps", "sens_maps")))
    kmean = temporal_mean_kspace(gt_mat, "kdata_full")
    imgc = ifftnc(kmean, axes=(-1, -2, -3))  # (enc, coil, z, y, x)
    num = np.sum(imgc * np.conj(coilmap)[None, ...], axis=1)
    den = np.sum(np.abs(coilmap) ** 2, axis=0) + 1e-8
    img = (num / den[None, ...]).astype(np.complex64, copy=False)  # (enc, z, y, x)
    mag = np.mean(np.abs(img), axis=0).astype(np.float32)
    flow = np.angle(img[1:] * np.conj(img[0:1]))
    pcmra = (mag * np.sqrt(np.sum(flow**2, axis=0))).astype(np.float32)  # (z, y, x)
    return pcmra, mag


def gt_pcmra(gt_mat: Path, coilmap_mat: Path) -> np.ndarray:
    pcmra, _ = gt_pcmra_and_mag(gt_mat, coilmap_mat)
    return pcmra


LAYOUT_TO_ZYX_PERM = {
    "zyx": (0, 1, 2),
    "xyz": (2, 1, 0),
    "yzx": (1, 0, 2),
    "xzy": (1, 2, 0),
    "zxy": (0, 2, 1),
    "yxz": (2, 0, 1),
}


def orient_3d(vol: np.ndarray, target_shape: tuple[int, int, int], layout: str) -> np.ndarray:
    vol = np.asarray(vol)
    if layout in LAYOUT_TO_ZYX_PERM:
        out = np.transpose(vol, LAYOUT_TO_ZYX_PERM[layout])
    elif layout == "auto":
        matches = []
        for perm in itertools.permutations(range(3)):
            candidate = np.transpose(vol, perm)
            if candidate.shape == target_shape:
                matches.append(candidate)
        if not matches:
            raise ValueError(f"Cannot orient volume shape {vol.shape} to target {target_shape}")
        if len(matches) > 1:
            print(
                f"[warn] Ambiguous orientation for volume shape {vol.shape} to target {target_shape}; "
                "using the first shape-compatible permutation. Pass --recon-layout explicitly."
            )
        out = matches[0]
    else:
        raise ValueError(f"Unsupported 3D layout: {layout}")
    if out.shape != target_shape:
        raise ValueError(f"Oriented volume has shape {out.shape}, expected {target_shape}")
    return out.astype(np.float32, copy=False)


def recon_volume_mean(path: Path, target_shape: tuple[int, int, int], layout: str) -> tuple[np.ndarray, bool]:
    arr = to_complex(read_mat_array(path, ("img4ranking", "recon", "reconstruction", "image")))
    is_complex = np.iscomplexobj(arr)
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 4:
        if layout.endswith("t") and len(layout) == 4:
            vol = np.mean(arr, axis=3)
            return orient_3d(vol, target_shape, layout[:3]), is_complex
        candidates = []
        for time_axis in range(4):
            vol = np.mean(arr, axis=time_axis)
            try:
                candidates.append(orient_3d(vol, target_shape, "auto"))
            except ValueError:
                pass
        if not candidates:
            raise ValueError(f"Cannot orient recon shape {arr.shape} to target {target_shape}: {path}")
        return candidates[0], is_complex
    if arr.ndim == 3:
        return orient_3d(arr, target_shape, layout if layout in LAYOUT_TO_ZYX_PERM else "auto"), is_complex
    raise ValueError(f"Expected 3D/4D recon array, got {arr.shape}: {path}")


def pcmra_from_recon_encs(enc_volumes: list[np.ndarray], complex_input: bool) -> tuple[np.ndarray, str]:
    stack = np.stack(enc_volumes, axis=0)
    if complex_input:
        mag = np.mean(np.abs(stack), axis=0)
        flow = np.angle(stack[1:] * np.conj(stack[0:1]))
        return (mag * np.sqrt(np.sum(flow**2, axis=0))).astype(np.float32), "true_complex_pcmra"

    # Current inference output is magnitude-only, so true phase-based PCMRA is impossible.
    # This fallback is only a 3D magnitude proxy for quick sanity checks.
    return np.mean(np.abs(stack), axis=0).astype(np.float32), "magnitude_mean_fallback"


def safe_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    data_range = max(float(gt.max()), float(recon.max())) - min(float(gt.min()), float(recon.min()))
    if data_range <= 0:
        data_range = 1.0
    return float(structural_similarity(gt, recon, data_range=data_range))


def xy_slice_ssim(gt_zyx: np.ndarray, recon_zyx: np.ndarray) -> tuple[float, float]:
    vals = [safe_ssim(gt_zyx[z], recon_zyx[z]) for z in range(gt_zyx.shape[0])]
    return float(np.mean(vals)), float(np.std(vals))


def case_from_path(recon_file: Path, recon_root: Path) -> tuple[str, str, str]:
    rel = recon_file.relative_to(recon_root)
    if len(rel.parts) < 4:
        raise ValueError(f"Expected recon path Center/Scanner/Patient/file.mat, got {recon_file}")
    return rel.parts[0], rel.parts[1], rel.parts[2]


def discover_recons(recon_root: Path) -> dict[tuple[str, str, str, str], dict[int, Path]]:
    groups: dict[tuple[str, str, str, str], dict[int, Path]] = defaultdict(dict)
    for path in sorted(recon_root.rglob("kdata_ktGaussian*_enc*_recon.mat")):
        match = RECON_RE.match(path.name)
        if not match:
            continue
        acc, enc = match.group(1), int(match.group(2))
        center, scanner, patient = case_from_path(path, recon_root)
        groups[(center, scanner, patient, acc)][enc] = path
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-root", type=Path, required=True, help="Folder like outputs/.../final")
    parser.add_argument("--data-root", type=Path, required=True, help="Aorta root containing Center/Scanner/Patient GT files")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/4dflow_pcmra_metrics.csv"))
    parser.add_argument("--encodings", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--mag-only",
        action="store_true",
        help="Only compute GT magnitude vs recon magnitude metrics; skip PCMRA/fallback metrics.",
    )
    parser.add_argument(
        "--recon-layout",
        choices=(
            "xyzt",
            "yzxt",
            "zyxt",
            "xzyt",
            "zxyt",
            "yxzt",
            "xyz",
            "yzx",
            "zyx",
            "xzy",
            "zxy",
            "yxz",
            "auto",
        ),
        default="yzxt",
        help=(
            "Axis order of recon img4ranking. Current 4D Flow inference outputs are "
            "(y,z,x,t), so the default is yzxt."
        ),
    )
    args = parser.parse_args()

    out_csv = assert_outputs_not_in_data([args.out_csv], [args.recon_root, args.data_root])[0]
    groups = discover_recons(args.recon_root)
    rows = []

    for (center, scanner, patient, acc), enc_map in groups.items():
        missing = [enc for enc in args.encodings if enc not in enc_map]
        if missing:
            print(f"[skip] {center}/{scanner}/{patient} {acc}: missing enc {missing}")
            continue

        patient_dir = args.data_root / center / scanner / patient
        gt_mat = patient_dir / "kdata_full.mat"
        coilmap_mat = patient_dir / "coilmap.mat"
        if not gt_mat.exists() or not coilmap_mat.exists():
            print(f"[skip] missing GT/coilmap: {patient_dir}")
            continue

        gt, gt_mag = gt_pcmra_and_mag(gt_mat, coilmap_mat)
        enc_volumes = []
        complex_flags = []
        for enc in args.encodings:
            vol, is_complex = recon_volume_mean(enc_map[enc], gt.shape, args.recon_layout)
            enc_volumes.append(vol)
            complex_flags.append(is_complex)
        recon_mag = np.mean(np.abs(np.stack(enc_volumes, axis=0)), axis=0).astype(np.float32)

        row = {
            "center": center,
            "scanner": scanner,
            "patient": patient,
            "acc": acc,
            "mag_ssim_3d": safe_ssim(gt_mag, recon_mag),
        }
        row["mag_ssim_xy_mean"], row["mag_ssim_xy_std"] = xy_slice_ssim(gt_mag, recon_mag)
        if args.mag_only:
            row["mode"] = "magnitude_only"
        else:
            recon, mode = pcmra_from_recon_encs(enc_volumes, complex_input=all(complex_flags))
            row["mode"] = mode
            row["ssim_3d"] = safe_ssim(gt, recon)
            row["ssim_xy_mean"], row["ssim_xy_std"] = xy_slice_ssim(gt, recon)
        rows.append(row)
        if args.mag_only:
            print(
                f"{center}/{scanner}/{patient} {acc}: "
                f"mode={row['mode']}, mag_ssim_3d={row['mag_ssim_3d']:.6f}, "
                f"mag_xy_mean={row['mag_ssim_xy_mean']:.6f}±{row['mag_ssim_xy_std']:.6f}"
            )
        else:
            print(
                f"{center}/{scanner}/{patient} {acc}: "
                f"mode={row['mode']}, ssim_3d={row['ssim_3d']:.6f}, "
                f"xy_mean={row['ssim_xy_mean']:.6f}±{row['ssim_xy_std']:.6f}, "
                f"mag_ssim_3d={row['mag_ssim_3d']:.6f}, "
                f"mag_xy_mean={row['mag_ssim_xy_mean']:.6f}±{row['mag_ssim_xy_std']:.6f}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["center", "scanner", "patient", "acc", "mode"]
        if not args.mag_only:
            fieldnames += ["ssim_3d", "ssim_xy_mean", "ssim_xy_std"]
        fieldnames += ["mag_ssim_3d", "mag_ssim_xy_mean", "mag_ssim_xy_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
