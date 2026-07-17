#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from path_safety import assert_outputs_not_in_data


FLAT_RE = re.compile(r"^(?P<center>Center[^_]+)__(?P<scanner>.+)__(?P<patient>P\d+)__ktGaussian(?P<acc>\d+)__enc(?P<enc>\d+)\.mat$")
ORG_RE = re.compile(r"^kdata_ktGaussian(?P<acc>\d+)_enc(?P<enc>\d+)_recon\.mat$")
KSPACE_KEYS = ("kdata_ktGaussian", "kdata", "kspace", "kdata_full", "kspace_full")
SEGMASK_KEYS = ("segmask", "mask")
RECON_KEYS = ("img4ranking", "recon", "reconstruction", "image")
AXIS_ORDER = "tzyx"


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


def to_complex(arr: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    raise ValueError(
        f"{path} does not contain complex data. Official 4D Flow submission needs phase; "
        f"got dtype={arr.dtype}, shape={arr.shape}."
    )


def orient_to_tzyx(arr: np.ndarray, layout: str, target_shape: tuple[int, int, int, int], path: Path) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D per-encoding recon before submission export, got {arr.shape}: {path}")
    if sorted(layout) != sorted(AXIS_ORDER) or len(layout) != 4:
        raise ValueError(f"--recon-layout must be a permutation of '{AXIS_ORDER}', got {layout!r}")

    perm = tuple(layout.index(axis) for axis in AXIS_ORDER)
    out = np.transpose(arr, perm)
    if out.shape != target_shape:
        raise ValueError(
            f"Oriented recon shape mismatch for {path}: got {out.shape}, expected {target_shape}. "
            f"Input shape={arr.shape}, layout={layout}. If axes are swapped, pass the correct --recon-layout."
        )
    return out.astype(np.complex64, copy=False)


def save_coo_npz(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr).astype(np.complex64, copy=False)
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]
    np.savez_compressed(
        path,
        coords=coords,
        data=data.astype(np.complex64, copy=False),
        shape=np.array(arr.shape, dtype=np.int64),
    )


def load_coo_npz(path: Path) -> np.ndarray:
    z = np.load(path)
    coords = z["coords"]
    data = z["data"]
    shape = tuple(int(v) for v in z["shape"])
    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out


def discover_recon_files(recon_root: Path) -> dict[tuple[str, str, str, int], dict[int, Path]]:
    groups: dict[tuple[str, str, str, int], dict[int, Path]] = defaultdict(dict)
    roots = [recon_root]
    if (recon_root / "val_img4ranking").is_dir():
        roots.append(recon_root / "val_img4ranking")

    for root in roots:
        for path in sorted(root.rglob("*.mat")):
            flat = FLAT_RE.match(path.name)
            if flat:
                center = flat.group("center")
                scanner = flat.group("scanner")
                patient = flat.group("patient")
                acc = int(flat.group("acc"))
                enc = int(flat.group("enc"))
                groups[(center, scanner, patient, acc)][enc] = path
                continue

            org = ORG_RE.match(path.name)
            if org:
                rel = path.relative_to(recon_root)
                if len(rel.parts) < 4:
                    print(f"[skip] organized output needs Center/Scanner/Patient/file.mat: {path}")
                    continue
                center, scanner, patient = rel.parts[-4], rel.parts[-3], rel.parts[-2]
                acc = int(org.group("acc"))
                enc = int(org.group("enc"))
                groups[(center, scanner, patient, acc)][enc] = path

    return groups


def official_shape(data_root: Path, center: str, scanner: str, patient: str, acc: int) -> tuple[int, int, int, int, int]:
    kspace_path = data_root / center / scanner / patient / f"kdata_ktGaussian{acc}.mat"
    if not kspace_path.exists():
        raise FileNotFoundError(f"Missing source k-space for shape check: {kspace_path}")
    shape = read_mat_shape(kspace_path, KSPACE_KEYS)
    if len(shape) != 6:
        raise ValueError(f"Expected source k-space shape (Nv,Nt,Nc,SPE,PE,FE), got {shape}: {kspace_path}")
    nv, nt, _nc, spe, pe, fe = shape
    return int(nv), int(nt), int(spe), int(pe), int(fe)


def read_segmask(data_root: Path, center: str, scanner: str, patient: str, target_shape: tuple[int, int, int]) -> np.ndarray:
    seg_path = data_root / center / scanner / patient / "segmask.mat"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmask: {seg_path}")
    segmask = np.asarray(read_mat_array(seg_path, SEGMASK_KEYS)).squeeze()
    if segmask.shape != target_shape:
        raise ValueError(f"segmask shape mismatch: got {segmask.shape}, expected {target_shape}: {seg_path}")
    return segmask.astype(bool, copy=False)


def export_case(
    enc_map: dict[int, Path],
    *,
    out_root: Path,
    data_root: Path,
    task: str,
    split: str,
    anatomy: str,
    center: str,
    scanner: str,
    patient: str,
    acc: int,
    encodings: list[int],
    recon_layout: str,
    overwrite: bool,
    verify: bool,
) -> Path:
    missing = [enc for enc in encodings if enc not in enc_map]
    if missing:
        raise ValueError(f"{center}/{scanner}/{patient} ktGaussian{acc}: missing encodings {missing}")

    nv, nt, spe, pe, fe = official_shape(data_root, center, scanner, patient, acc)
    if nv < len(encodings):
        raise ValueError(f"Source k-space has Nv={nv}, but requested encodings={encodings}")
    target_per_enc = (nt, spe, pe, fe)
    segmask = read_segmask(data_root, center, scanner, patient, (spe, pe, fe))

    per_enc = []
    for enc in encodings:
        raw = read_mat_array(enc_map[enc], RECON_KEYS)
        img = to_complex(raw, enc_map[enc])
        per_enc.append(orient_to_tzyx(img, recon_layout, target_per_enc, enc_map[enc]))

    stacked = np.stack(per_enc, axis=0).astype(np.complex64, copy=False)
    stacked *= segmask[None, None, :, :, :]

    out_path = out_root / task / split / anatomy / center / scanner / patient / f"img_ktGaussian{acc}.npz"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")

    save_coo_npz(out_path, stacked)
    if verify:
        reloaded = load_coo_npz(out_path)
        if reloaded.shape != stacked.shape:
            raise RuntimeError(f"Reloaded shape mismatch for {out_path}: {reloaded.shape} vs {stacked.shape}")
        max_err = float(np.max(np.abs(reloaded - stacked))) if stacked.size else 0.0
        if max_err != 0.0:
            raise RuntimeError(f"Reloaded data mismatch for {out_path}: max_abs_err={max_err}")
        outside = reloaded[:, :, ~segmask]
        outside_max = float(np.max(np.abs(outside))) if outside.size else 0.0
        if outside_max != 0.0:
            raise RuntimeError(f"Nonzero values outside segmask for {out_path}: max={outside_max}")

    return out_path


def zip_submission(out_root: Path, zip_path: Path, task: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    base = out_root
    task_root = out_root / task
    if not task_root.exists():
        raise FileNotFoundError(f"Cannot zip; missing task output: {task_root}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(task_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(base))


def parse_encodings(value: str) -> list[int]:
    encs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not encs:
        raise argparse.ArgumentTypeError("At least one encoding is required.")
    return encs


def parse_accelerations(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NV-Raw2Insights 4D Flow per-encoding .mat outputs to official CMRx4DFlow2026 sparse NPZ submission format."
    )
    parser.add_argument("--recon-root", type=Path, required=True, help="Root containing val_img4ranking/*.mat or Center/Scanner/Patient/*.mat")
    parser.add_argument("--data-root", type=Path, required=True, help="Official Aorta data root used for shape and segmask lookup")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for official submission tree")
    parser.add_argument("--task", type=str, default="TaskR1R2")
    parser.add_argument("--split", type=str, default="ValidationSet", help="ValidationSet or TestSet")
    parser.add_argument("--anatomy", type=str, default="Aorta")
    parser.add_argument("--recon-layout", type=str, default="yzxt", help="Axis order of per-encoding recon before real/imag, e.g. yzxt -> tzyx")
    parser.add_argument("--encodings", type=parse_encodings, default=[0, 1, 2, 3], help="Comma-separated encodings to export")
    parser.add_argument("--accelerations", type=parse_accelerations, default=None, help="Optional comma-separated acceleration filter")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true", help="Skip read-back verification")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip cases missing requested encodings instead of raising")
    parser.add_argument("--zip", action="store_true", help="Create Submission.zip under out-root")
    parser.add_argument("--zip-name", type=str, default="Submission.zip")
    args = parser.parse_args()

    out_root = assert_outputs_not_in_data([args.out_root], [args.data_root])[0]
    out_root.mkdir(parents=True, exist_ok=True)

    groups = discover_recon_files(args.recon_root)
    if args.accelerations is not None:
        groups = {key: value for key, value in groups.items() if key[3] in args.accelerations}
    if not groups:
        raise RuntimeError(f"No reconstruction .mat files found under {args.recon_root}")

    written = []
    failed = 0
    for (center, scanner, patient, acc), enc_map in sorted(groups.items()):
        try:
            out_path = export_case(
                enc_map,
                out_root=out_root,
                data_root=args.data_root,
                task=args.task,
                split=args.split,
                anatomy=args.anatomy,
                center=center,
                scanner=scanner,
                patient=patient,
                acc=acc,
                encodings=args.encodings,
                recon_layout=args.recon_layout,
                overwrite=args.overwrite,
                verify=not args.no_verify,
            )
            written.append(out_path)
            print(f"[OK] {center}/{scanner}/{patient} ktGaussian{acc} -> {out_path}")
        except Exception as exc:
            failed += 1
            msg = f"[ERROR] {center}/{scanner}/{patient} ktGaussian{acc}: {exc}"
            if args.skip_incomplete:
                print(msg)
                continue
            raise RuntimeError(msg) from exc

    if args.zip:
        zip_path = out_root / args.zip_name
        zip_submission(out_root, zip_path, args.task)
        print(f"[ZIP] {zip_path}")

    print(f"done. written={len(written)}, failed={failed}, out_root={out_root}")


if __name__ == "__main__":
    main()
