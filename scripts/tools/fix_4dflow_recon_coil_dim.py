#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_safety import assert_outputs_not_in_data  # noqa: E402


RECON_RE = re.compile(r"^kdata_ktGaussian(?P<acc>\d+)_enc(?P<enc>\d+)_recon\.mat$")
RECON_KEYS = ("img4ranking", "recon", "reconstruction", "image")
KSPACE_KEYS = ("kdata_ktGaussian", "kdata", "kspace", "kdata_full", "kspace_full")
COILMAP_KEYS = ("coilmap", "csm", "sensitivity_maps", "sens_maps")


def read_mat_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    try:
        with h5py.File(path, "r", swmr=True) as f:
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


def read_mat_shape(path: Path, preferred_keys: tuple[str, ...]) -> tuple[int, ...]:
    try:
        with h5py.File(path, "r", swmr=True) as f:
            for key in preferred_keys:
                if key in f:
                    return tuple(int(v) for v in f[key].shape)
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    return tuple(int(v) for v in f[key].shape)
    except OSError:
        return tuple(int(v) for v in read_mat_array(path, preferred_keys).shape)
    raise ValueError(f"No dataset found in {path}")


def to_complex_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    return arr


def complex_to_ri(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if not np.iscomplexobj(arr):
        raise ValueError(f"Expected complex array, got dtype={arr.dtype}")
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32, copy=False)


def parse_case_from_parts(parts: tuple[str, ...], default_anatomy: str, label: str) -> tuple[str, str, str, str]:
    if len(parts) < 4:
        raise ValueError(f"Expected path like [Aorta/]Center/Scanner/Patient/file, got {label}")
    patient = parts[-2]
    scanner = parts[-3]
    center = parts[-4]
    anatomy = parts[-5] if len(parts) >= 5 else default_anatomy
    if not center.startswith("Center") or not patient.startswith("P"):
        raise ValueError(f"Could not infer Center/Scanner/Patient from {label}")
    return anatomy, center, scanner, patient


def parse_case(src: Path, rel: Path, default_anatomy: str) -> tuple[str, str, str, str]:
    if len(rel.parts) >= 4:
        return parse_case_from_parts(rel.parts, default_anatomy, str(rel))
    return parse_case_from_parts(src.parts, default_anatomy, str(src))


def patient_dir_from_data_root(data_root: Path, anatomy: str, center: str, scanner: str, patient: str) -> Path:
    if data_root.name == anatomy:
        return data_root / center / scanner / patient
    if (data_root / anatomy).is_dir():
        return data_root / anatomy / center / scanner / patient
    return data_root / center / scanner / patient


def load_case_meta(data_root: Path, anatomy: str, center: str, scanner: str, patient: str, acc: int):
    patient_dir = patient_dir_from_data_root(data_root, anatomy, center, scanner, patient)
    kspace_path = patient_dir / f"kdata_ktGaussian{acc}.mat"
    coilmap_path = patient_dir / "coilmap.mat"
    if not kspace_path.exists():
        raise FileNotFoundError(f"Missing source k-space: {kspace_path}")
    if not coilmap_path.exists():
        raise FileNotFoundError(f"Missing coilmap: {coilmap_path}")

    kshape = read_mat_shape(kspace_path, KSPACE_KEYS)
    if len(kshape) != 6:
        raise ValueError(f"Expected k-space shape (Nv,Nt,Nc,SPE,PE,FE), got {kshape}: {kspace_path}")
    nv, nt, nc, spe, pe, fe = [int(v) for v in kshape]
    coilmap = read_mat_array(coilmap_path, COILMAP_KEYS)
    if coilmap.shape != (nc, spe, pe, fe):
        raise ValueError(
            f"coilmap shape mismatch for {coilmap_path}: got {coilmap.shape}, expected {(nc, spe, pe, fe)}"
        )
    return (nv, nt, nc, spe, pe, fe), coilmap.astype(np.complex64, copy=False)


def combine_yzxtc_to_yzxt(img_yzxtc: np.ndarray, coilmap_czyx: np.ndarray, eps: float) -> np.ndarray:
    if img_yzxtc.ndim != 5:
        raise ValueError(f"Expected image shape (y,z,x,t,c), got {img_yzxtc.shape}")

    pe, spe, fe, nt, nc = img_yzxtc.shape
    if coilmap_czyx.shape != (nc, spe, pe, fe):
        raise ValueError(
            f"coilmap shape mismatch: got {coilmap_czyx.shape}, expected {(nc, spe, pe, fe)} "
            f"for image shape {img_yzxtc.shape}"
        )

    img_tzyxc = np.transpose(img_yzxtc, (3, 1, 0, 2, 4)).astype(np.complex64, copy=False)
    coil_zyxc = np.transpose(coilmap_czyx, (1, 2, 3, 0)).astype(np.complex64, copy=False)
    numerator = np.sum(img_tzyxc * np.conj(coil_zyxc)[None, ...], axis=-1)
    denominator = np.sum(np.abs(coil_zyxc) ** 2, axis=-1)[None, ...] + eps
    combined_tzyx = numerator / denominator
    return np.transpose(combined_tzyx, (2, 1, 3, 0)).astype(np.complex64, copy=False)


def fix_one_file(
    src: Path,
    dst: Path,
    *,
    final_root: Path,
    data_root: Path,
    default_anatomy: str,
    eps: float,
    overwrite: bool,
    dry_run: bool,
) -> str:
    match = RECON_RE.match(src.name)
    if not match:
        return "skip-name"

    rel = src.relative_to(final_root)
    anatomy, center, scanner, patient = parse_case(src, rel, default_anatomy)
    acc = int(match.group("acc"))
    _enc = int(match.group("enc"))

    (nv, nt, nc, spe, pe, fe), coilmap = load_case_meta(data_root, anatomy, center, scanner, patient, acc)
    arr = read_mat_array(src, RECON_KEYS)

    if arr.shape == (pe, spe, fe, nt, nc):
        fixed = combine_yzxtc_to_yzxt(arr, coilmap, eps=eps)
        action = "coil-combine"
    elif arr.shape == (pe, spe, fe, nt):
        fixed = arr.astype(np.complex64, copy=False)
        action = "already-fixed"
    else:
        raise ValueError(
            f"Unexpected recon shape for {src}: got {arr.shape}. "
            f"Expected {(pe, spe, fe, nt, nc)} or {(pe, spe, fe, nt)} based on k-space {(nv, nt, nc, spe, pe, fe)}."
        )

    if fixed.shape != (pe, spe, fe, nt):
        raise ValueError(f"Fixed shape mismatch for {src}: got {fixed.shape}, expected {(pe, spe, fe, nt)}")

    if dry_run:
        print(f"[DRY] {action}: {src} shape={arr.shape} -> {dst} shape={fixed.shape}")
        return action

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(dst, {"img4ranking": complex_to_ri(fixed)}, do_compression=True)
    print(f"[OK] {action}: {src} -> {dst}")
    return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fix 4D Flow raw complex inference .mat files that still contain a coil dimension. "
            "Input is expected as (PE,SPE,FE,Nt,Nc,2); output is coil-combined (PE,SPE,FE,Nt,2)."
        )
    )
    parser.add_argument("--final-root", type=Path, required=True, help="Input root containing final/Aorta/Center/Scanner/PXXX/*.mat")
    parser.add_argument("--fixed-root", type=Path, required=True, help="Output root that mirrors final-root structure")
    parser.add_argument("--data-root", type=Path, required=True, help="Official Aorta root, or parent containing Aorta")
    parser.add_argument("--anatomy", type=str, default="Aorta")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    final_root = args.final_root.expanduser().resolve(strict=False)
    fixed_root = assert_outputs_not_in_data([args.fixed_root], [args.data_root])[0]
    data_root = args.data_root.expanduser().resolve(strict=False)

    files = sorted(final_root.rglob("kdata_ktGaussian*_enc*_recon.mat"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No recon files found under {final_root}")

    counts: dict[str, int] = {}
    for src in files:
        rel = src.relative_to(final_root)
        dst = fixed_root / rel
        action = fix_one_file(
            src,
            dst,
            final_root=final_root,
            data_root=data_root,
            default_anatomy=args.anatomy,
            eps=args.eps,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        counts[action] = counts.get(action, 0) + 1

    print(f"done. files={len(files)}, counts={counts}, fixed_root={fixed_root}")


if __name__ == "__main__":
    main()
