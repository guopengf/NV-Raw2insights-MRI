#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import h5py
import scipy.io


DEFAULT_EVAL_CODE_DIR = Path("/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode")
NPZ_RE = re.compile(r"^img_ktGaussian(?P<R>\d+)\.npz$")


def _load_official_module(eval_code_dir: Path, module_name: str):
    """Load one evaluator module without executing the package ``__init__``."""

    package_digest = hashlib.sha256(str(eval_code_dir).encode("utf-8")).hexdigest()[:16]
    package_name = f"_cmrx4dflow_eval_{package_digest}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(eval_code_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    module_path = eval_code_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official evaluator module not found: {module_path}")
    qualified_name = f"{package_name}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for official evaluator module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def import_official_eval(eval_code_dir: Path):
    eval_code_dir = eval_code_dir.expanduser().resolve(strict=False)
    if not eval_code_dir.is_dir():
        raise FileNotFoundError(f"EvaluationCode directory not found: {eval_code_dir}")
    eval_code_path = str(eval_code_dir)
    if eval_code_path not in sys.path:
        # Preserve support for older flat evaluator copies that use imports such
        # as ``from pytorch_ssim import ...`` instead of package-relative imports.
        sys.path.insert(0, eval_code_path)
    utils_bgc = _load_official_module(eval_code_dir, "utils_bgc")
    utils_flow = _load_official_module(eval_code_dir, "utils_flow")
    utils_metrics = _load_official_module(eval_code_dir, "utils_metrics")

    return {
        "execute_MSAC": utils_bgc.execute_MSAC,
        "load_coo_npz": load_coo_npz,
        "load_mat_array": load_mat_array,
        "save_coo_npz": save_coo_npz,
        "complex2magflow": utils_flow.complex2magflow,
        "SSIM": utils_metrics.SSIM,
        "nRMSE": utils_metrics.nRMSE,
        "RelErr": utils_metrics.RelErr,
        "AngErr": utils_metrics.AngErr,
        "ComplexDiffErr": utils_metrics.ComplexDiffErr,
    }


def load_coo_npz(path: str | Path, as_dense: bool = True):
    z = np.load(path)
    coords = z["coords"]
    data = z["data"]
    shape = tuple(int(v) for v in z["shape"])
    if not as_dense:
        return coords, data, shape
    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out


def save_coo_npz(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]
    np.savez_compressed(
        path,
        coords=coords,
        data=data,
        shape=np.array(arr.shape, dtype=np.int64),
    )


def load_mat_array(path: str | Path, key: str) -> np.ndarray:
    path = Path(path)
    try:
        with h5py.File(path, "r", swmr=True) as f:
            if key not in f:
                raise KeyError(f"Missing key '{key}' in {path}; keys={list(f.keys())}")
            arr = f[key][()]
    except OSError:
        dat = scipy.io.loadmat(path)
        if key not in dat:
            raise KeyError(f"Missing key '{key}' in {path}; keys={[k for k in dat if not k.startswith('__')]}")
        arr = dat[key]
    return np.asarray(arr)


def as_float32_if_real(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return x.astype(np.complex64, copy=False)
    return x.astype(np.float32, copy=False)


def get_nonzero_bbox(mask: np.ndarray):
    mask_nz = np.asarray(mask) != 0
    if not np.any(mask_nz):
        return None
    coords = np.nonzero(mask_nz)
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def crop_to_seg_bbox(gt: np.ndarray, pred: np.ndarray, segmask: np.ndarray, corr_maps: np.ndarray | None):
    bbox = get_nonzero_bbox(segmask)
    if bbox is None:
        return gt, pred, segmask, corr_maps
    prefix = (slice(None),) * (gt.ndim - segmask.ndim)
    full_slice = prefix + bbox
    gt_c = gt[full_slice]
    pred_c = pred[full_slice]
    seg_c = segmask[bbox]
    corr_c = corr_maps[full_slice] if corr_maps is not None else None
    return gt_c, pred_c, seg_c, corr_c


def phase_correct(gt: np.ndarray, pred: np.ndarray, corr_maps: np.ndarray):
    factor = np.exp(-1j * corr_maps).astype(np.complex64, copy=False)
    gt_c = np.array(gt, copy=True)
    pred_c = np.array(pred, copy=True)
    gt_c[1:] *= factor
    pred_c[1:] *= factor
    return gt_c, pred_c


def load_segmask(load_mat_array_fn, path: Path) -> np.ndarray:
    seg = load_mat_array_fn(str(path), key="segmask")
    return np.asarray(seg, dtype=np.float32)


def load_dense(load_coo_npz, path: Path) -> np.ndarray:
    return as_float32_if_real(load_coo_npz(str(path), as_dense=True))


def corrmap_path_for(gt_case_dir: Path, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        return gt_case_dir / "corrmap.npz"
    return cache_dir / gt_case_dir.relative_to(gt_case_dir.anchor) / "corrmap.npz"


def get_corrmap(funcs: dict, gt: np.ndarray, segmask: np.ndarray, gt_case_dir: Path, cache_dir: Path | None):
    path = corrmap_path_for(gt_case_dir, cache_dir)
    if path.exists():
        return as_float32_if_real(funcs["load_coo_npz"](str(path), as_dense=True))
    corr = funcs["execute_MSAC"](gt, corr_fit_order=3, th=0.1)
    corr = np.asarray(corr, dtype=np.float32) * np.asarray(segmask, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    funcs["save_coo_npz"](str(path), corr)
    return corr


def compute_corrmap(funcs: dict, gt: np.ndarray, segmask: np.ndarray) -> np.ndarray:
    """Compute the same MSAC background correction used by the wrapper scorer."""

    corr = funcs["execute_MSAC"](gt, corr_fit_order=3, th=0.1)
    return np.asarray(corr, dtype=np.float32) * np.asarray(segmask, dtype=np.float32)


def strip_submission_prefix(path: Path, submission_root: Path, task: str | None) -> Path:
    rel = path.relative_to(submission_root)
    parts = rel.parts
    if task is not None:
        if len(parts) < 7 or parts[0] != task:
            raise ValueError(f"Unexpected submission path for task={task}: {rel}")
        return rel
    if len(parts) >= 7 and parts[0].startswith("Task"):
        return rel
    raise ValueError(f"Submission file must live under Task*/Set/Anatomy/Center/Vendor/Patient: {rel}")


def evaluate_arrays(
    funcs: dict,
    pred: np.ndarray,
    gt: np.ndarray,
    segmask: np.ndarray,
    *,
    corr_maps: np.ndarray | None = None,
    include_complex_diff: bool = False,
):
    """Evaluate dense arrays with the same crop/conversion/metrics as submission evaluation."""

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    segmask = np.asarray(segmask, dtype=np.float32)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt{gt.shape} vs pred{pred.shape}")
    if gt.shape[0] != 4 or gt.ndim != 5:
        raise ValueError(f"Expected dense complex [4,time,z,y,x], got {gt.shape}")
    if segmask.shape != gt.shape[-3:]:
        raise ValueError(f"Segmentation shape {segmask.shape} does not match complex volume {gt.shape[-3:]}")
    if not np.any(segmask):
        raise ValueError("Official flow metrics require a non-empty segmentation mask")

    corr_maps = compute_corrmap(funcs, gt, segmask) if corr_maps is None else np.asarray(corr_maps)
    gt, pred, segmask, corr_maps = crop_to_seg_bbox(gt, pred, segmask, corr_maps)
    gt_c, pred_c = phase_correct(gt, pred, corr_maps)

    mag_gt, flow_gt = funcs["complex2magflow"](gt_c)
    mag_pred, flow_pred = funcs["complex2magflow"](pred_c)
    mag_gt = as_float32_if_real(mag_gt)
    mag_pred = as_float32_if_real(mag_pred)
    flow_gt = as_float32_if_real(flow_gt)
    flow_pred = as_float32_if_real(flow_pred)
    segmask = np.asarray(segmask, dtype=np.float32)

    row = {
        "SSIM": float(funcs["SSIM"](mag_pred, mag_gt, segmask)),
        "nRMSE": float(funcs["nRMSE"](mag_pred, mag_gt, segmask)),
        "RelErr": float(funcs["RelErr"](flow_pred, flow_gt, segmask)),
        "AngErr": float(funcs["AngErr"](flow_pred, flow_gt, segmask)),
    }
    if include_complex_diff:
        row["ComplexDiffErr"] = float(funcs["ComplexDiffErr"](pred_c, gt_c, segmask))
    return row


def evaluate_one(funcs: dict, pred_path: Path, gt_path: Path, seg_path: Path, cache_dir: Path | None, include_complex_diff: bool):
    gt = load_dense(funcs["load_coo_npz"], gt_path)
    pred = load_dense(funcs["load_coo_npz"], pred_path)
    segmask = load_segmask(funcs["load_mat_array"], seg_path)
    corr = get_corrmap(funcs, gt, segmask, gt_path.parent, cache_dir)
    return evaluate_arrays(
        funcs,
        pred,
        gt,
        segmask,
        corr_maps=corr,
        include_complex_diff=include_complex_diff,
    )


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CMRx4DFlow2026-style submission directory against GT using official EvaluationCode utilities."
    )
    parser.add_argument("--submission-root", type=Path, required=True, help="Directory containing TaskR1R2/Set/Aorta/.../img_ktGaussian*.npz")
    parser.add_argument("--gt-root", type=Path, default=Path("/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT"))
    parser.add_argument("--eval-code-dir", type=Path, default=DEFAULT_EVAL_CODE_DIR)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--task", type=str, default=None, help="Optional task filter, e.g. TaskR1R2")
    parser.add_argument("--include-complex-diff", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional corrmap cache directory. Default writes beside GT, matching official code.")
    parser.add_argument("--skip-errors", action="store_true", help="Keep going and record error comments instead of failing.")
    args = parser.parse_args()

    funcs = import_official_eval(args.eval_code_dir)
    submission_root = args.submission_root.expanduser().resolve(strict=False)
    gt_root = args.gt_root.expanduser().resolve(strict=False)

    pred_files = sorted(submission_root.rglob("img_ktGaussian*.npz"))
    if not pred_files:
        raise RuntimeError(f"No img_ktGaussian*.npz found under {submission_root}")

    rows = []
    metric_keys = ["SSIM", "nRMSE", "RelErr", "AngErr"]
    if args.include_complex_diff:
        metric_keys.append("ComplexDiffErr")

    for pred_path in pred_files:
        rel = strip_submission_prefix(pred_path, submission_root, args.task)
        match = NPZ_RE.match(pred_path.name)
        if match is None:
            continue
        gt_case_dir = gt_root / rel.parent
        gt_path = gt_case_dir / "img_gt.npz"
        seg_path = gt_case_dir / "segmask.mat"

        row = {
            "rel_path": str(rel),
            "task": rel.parts[0],
            "settype": rel.parts[1],
            "anatomy": rel.parts[2],
            "center": rel.parts[3],
            "vendor": rel.parts[4],
            "patient": rel.parts[5],
            "R": match.group("R"),
            "comments": "",
        }

        try:
            if not gt_path.exists():
                raise FileNotFoundError(f"missing GT: {gt_path}")
            if not seg_path.exists():
                raise FileNotFoundError(f"missing segmask: {seg_path}")
            row.update(evaluate_one(funcs, pred_path, gt_path, seg_path, args.cache_dir, args.include_complex_diff))
        except Exception as exc:
            if not args.skip_errors:
                raise
            row["comments"] = f"{type(exc).__name__}: {exc}"
            for key in metric_keys:
                row[key] = np.nan
        rows.append(row)
        print(f"[{row['comments'] or 'OK'}] {rel}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rel_path", "task", "settype", "anatomy", "center", "vendor", "patient", "R", *metric_keys, "comments"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if not r.get("comments")]
    summary = {
        "num_valid": len(valid),
        "num_total": len(rows),
        "metrics_mean": {
            key: mean_or_nan([float(r[key]) for r in valid if key in r and np.isfinite(float(r[key]))])
            for key in metric_keys
        },
    }
    if args.out_json is None:
        args.out_json = args.out_csv.with_suffix(".summary.json")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote csv: {args.out_csv}")
    print(f"wrote summary: {args.out_json}")


if __name__ == "__main__":
    main()
