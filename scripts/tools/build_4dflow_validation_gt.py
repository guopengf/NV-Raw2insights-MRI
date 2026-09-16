#!/usr/bin/env python3
"""Build a bounded-memory image-domain cache for CMRx4DFlow validation GT.

The challenge stores full k-space as MATLAB v7.3/HDF5 arrays with shape
``(encoding, time, coil, z, y, x)``.  This tool applies the organizer's
centered, orthonormal 3-D inverse FFT and sensitivity-map coil combination,
but only holds one ``(coil, z, y, x)`` slab in memory at a time.

Raw challenge data is never modified.  Each output case contains a dense
``img_gt.npy`` plus exact copies of ``segmask.mat`` and ``params.csv``.
Dense NPY is intentional: GT is not sparse, and constructing COO coordinates
for every voxel has a much larger peak-memory cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.fft


REQUIRED_FILES = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compound_to_complex(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    fields = value.dtype.fields
    if fields is not None and "real" in fields and "imag" in fields:
        return value["real"].astype(np.float32) + 1j * value["imag"].astype(np.float32)
    if np.iscomplexobj(value):
        return value.astype(np.complex64, copy=False)
    raise TypeError(f"Expected complex or compound real/imag data, got dtype={value.dtype}")


def discover_cases(data_root: Path) -> list[dict]:
    cases = []
    for kspace_path in sorted(data_root.glob("Task*/ValidationSet/*/*/*/*/kdata_full.mat")):
        case_dir = kspace_path.parent
        missing = [name for name in REQUIRED_FILES if not (case_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {case_dir}")
        rel = case_dir.relative_to(data_root)
        parts = rel.parts
        if len(parts) != 6:
            raise ValueError(f"Unexpected validation case path: {rel}")
        cases.append(
            {
                "index": len(cases),
                "rel_dir": rel.as_posix(),
                "task": parts[0],
                "settype": parts[1],
                "anatomy": parts[2],
                "center": parts[3],
                "vendor": parts[4],
                "patient": parts[5],
                "source_sizes": {
                    name: (case_dir / name).stat().st_size for name in REQUIRED_FILES
                },
            }
        )
    if not cases:
        raise RuntimeError(f"No validation kdata_full.mat cases found under {data_root}")
    return cases


def prepare_plan(data_root: Path, plan_path: Path, expected_cases: int | None) -> None:
    data_root = data_root.expanduser().resolve(strict=True)
    cases = discover_cases(data_root)
    if expected_cases is not None and len(cases) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} cases, found {len(cases)}")
    task_counts: dict[str, int] = {}
    for case in cases:
        task_counts[case["task"]] = task_counts.get(case["task"], 0) + 1
    payload = {
        "schema_version": 1,
        "data_root": str(data_root),
        "case_count": len(cases),
        "task_counts": task_counts,
        "cases": cases,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_path.with_suffix(plan_path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, plan_path)
    print(json.dumps({"plan": str(plan_path), "case_count": len(cases), "task_counts": task_counts}))


def centered_ifft3(x: np.ndarray, workers: int) -> np.ndarray:
    axes = (-3, -2, -1)
    return scipy.fft.fftshift(
        scipy.fft.ifftn(
            scipy.fft.ifftshift(x, axes=axes),
            axes=axes,
            norm="ortho",
            workers=workers,
        ),
        axes=axes,
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _validate_existing(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        return cached.shape == shape and cached.dtype == np.dtype(np.complex64)
    except Exception:
        return False


def build_case(
    plan_path: Path,
    case_index: int,
    gt_root: Path,
    workers: int,
    overwrite: bool,
    hash_output: bool,
) -> None:
    plan = json.loads(plan_path.read_text())
    cases = plan["cases"]
    if not 0 <= case_index < len(cases):
        raise IndexError(f"case-index {case_index} is outside [0, {len(cases)})")
    case = cases[case_index]
    source_dir = Path(plan["data_root"]) / case["rel_dir"]
    output_dir = gt_root / case["rel_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "img_gt.npy"
    metadata_path = output_dir / "img_gt.metadata.json"

    started = time.time()
    with h5py.File(source_dir / "kdata_full.mat", "r", swmr=True) as kspace_file, h5py.File(
        source_dir / "coilmap.mat", "r", swmr=True
    ) as coil_file:
        kspace = kspace_file["kdata_full"]
        coil_dset = coil_file["coilmap"]
        if kspace.ndim != 6 or coil_dset.ndim != 4:
            raise ValueError(f"Unexpected shapes kspace={kspace.shape}, coilmap={coil_dset.shape}")
        nv, nt, nc, nz, ny, nx = (int(v) for v in kspace.shape)
        expected_coil_shape = (nc, nz, ny, nx)
        if tuple(coil_dset.shape) != expected_coil_shape:
            raise ValueError(
                f"Coilmap shape {coil_dset.shape} does not match kspace {expected_coil_shape}"
            )
        output_shape = (nv, nt, nz, ny, nx)
        if output_path.exists() and not overwrite:
            if not _validate_existing(output_path, output_shape):
                raise RuntimeError(f"Existing cache is invalid; rerun with --overwrite: {output_path}")
            print(json.dumps({"status": "already-valid", "case_index": case_index, "path": str(output_path)}))
            return

        coilmap = compound_to_complex(coil_dset[()]).astype(np.complex64, copy=False)
        coilmap_conj = np.conj(coilmap)
        temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
        try:
            output = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.complex64, shape=output_shape
            )
            for encoding in range(nv):
                for frame in range(nt):
                    slab = compound_to_complex(kspace[encoding, frame, ...]).astype(
                        np.complex64, copy=False
                    )
                    coil_images = centered_ifft3(slab, workers=workers)
                    output[encoding, frame] = np.sum(
                        coil_images * coilmap_conj, axis=0, dtype=np.complex64
                    )
                    del slab, coil_images
                output.flush()
                print(
                    f"case={case_index}/{len(cases)} encoding={encoding + 1}/{nv} "
                    f"elapsed_s={time.time() - started:.1f}",
                    flush=True,
                )
            del output
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    _atomic_copy(source_dir / "segmask.mat", output_dir / "segmask.mat")
    _atomic_copy(source_dir / "params.csv", output_dir / "params.csv")
    metadata = {
        **case,
        "source_dir": str(source_dir),
        "output_path": str(output_path),
        "shape": list(output_shape),
        "dtype": "complex64",
        "formula": "sum(ifftn_centered(kdata_full, axes=(-3,-2,-1), norm=ortho) * conj(coilmap), axis=coil)",
        "elapsed_seconds": time.time() - started,
        "output_size": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path) if hash_output else None,
        "segmask_sha256": sha256_file(output_dir / "segmask.mat"),
        "params_sha256": sha256_file(output_dir / "params.csv"),
    }
    temporary_meta = metadata_path.with_suffix(metadata_path.suffix + f".tmp-{os.getpid()}")
    temporary_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_meta, metadata_path)
    print(json.dumps({"status": "complete", "case_index": case_index, "path": str(output_path)}))


def finalize(plan_path: Path, gt_root: Path, receipt_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    rows = []
    for case in plan["cases"]:
        case_dir = gt_root / case["rel_dir"]
        metadata_path = case_dir / "img_gt.metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        image_path = case_dir / "img_gt.npy"
        if not _validate_existing(image_path, tuple(metadata["shape"])):
            raise RuntimeError(f"Invalid image cache: {image_path}")
        if not (case_dir / "segmask.mat").is_file() or not (case_dir / "params.csv").is_file():
            raise RuntimeError(f"Missing copied metadata files: {case_dir}")
        rows.append(metadata)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "plan": str(plan_path),
        "gt_root": str(gt_root),
        "case_count": len(rows),
        "task_counts": plan["task_counts"],
        "total_output_bytes": sum(int(row["output_size"]) for row in rows),
        "cases": rows,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps({"status": "complete", "case_count": len(rows), "receipt": str(receipt_path)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-plan", action="store_true")
    modes.add_argument("--run-case", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-cases", type=int, default=112)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-output", action="store_true")
    args = parser.parse_args()

    if args.prepare_plan:
        if args.data_root is None:
            parser.error("--prepare-plan requires --data-root")
        prepare_plan(args.data_root, args.plan, args.expected_cases)
    elif args.run_case:
        if args.gt_root is None or args.case_index is None:
            parser.error("--run-case requires --gt-root and --case-index")
        build_case(
            args.plan,
            args.case_index,
            args.gt_root,
            max(1, args.workers),
            args.overwrite,
            args.hash_output,
        )
    else:
        if args.gt_root is None or args.receipt is None:
            parser.error("--finalize requires --gt-root and --receipt")
        finalize(args.plan, args.gt_root, args.receipt)


if __name__ == "__main__":
    main()
