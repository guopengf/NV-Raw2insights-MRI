"""Batch-score final-manifest cases through the existing official-wrapper metric core."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common.case_data import FullCase, read_pipeline, select_cases
from .common.prediction_io import discover, load_array, load_prediction, read_index
from .common.safety import ROOT, OutputGuard, sha256


METRICS = ("SSIM", "nRMSE", "RelErr", "AngErr", "ComplexDiffErr")


def official_core(eval_dir):
    # Importing external evaluation code must not create __pycache__ beside datasets.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from scripts.tools.evaluate_4dflow_submission import evaluate_arrays, import_official_eval

        funcs = import_official_eval(Path(eval_dir))
    finally:
        sys.dont_write_bytecode = previous
    return evaluate_arrays, funcs


def score(data, prediction, settings, evaluate_arrays, funcs):
    if prediction.shape != data.gt.shape:
        raise ValueError(f"Full-case shape mismatch: prediction={prediction.shape}, GT={data.gt.shape}; train windows cannot be scored as full cases")
    seg = data.segmentation()
    gt = np.asarray(data.gt)
    corr_path = data.case["files"].get("corrmap.npz")
    if not corr_path or not Path(corr_path).is_file():
        raise FileNotFoundError("A supplied corrmap.npz is required; this entry never writes a corrmap into GT folders")
    corr = np.asarray(load_array(corr_path))
    if np.iscomplexobj(corr):
        if not np.all(corr.imag == 0):
            raise ValueError("corrmap must contain real phase offsets; nonzero/non-finite imaginary values cannot be discarded")
        corr = corr.real
    corr = np.asarray(corr, np.float32)
    if corr.shape not in {(3, 1, *seg.shape), (3, gt.shape[1], *seg.shape)}:
        raise ValueError(f"corrmap shape not aligned to [3,1/T,Z,Y,X]: {corr.shape}")
    if not np.isfinite(corr).all():
        raise ValueError("Non-finite supplied background phase correction")
    if settings["mask_images"]:
        gt = gt * seg[None, None]
        prediction = prediction * seg[None, None]
    venc = np.array([float(v) for v in data.params["VENC"].split(";")], np.float32) if settings["use_venc"] else None
    return evaluate_arrays(funcs, prediction, gt, seg, corr_maps=corr,
                           include_complex_diff=settings["include_complex_diff"],
                           spatial_mode=settings["spatial_mode"], venc=venc)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["dataset_group"], str(row["R"]))].append(row)
    output = []
    for (method, group, r), entries in sorted(grouped.items()):
        valid = [row for row in entries if not row["comments"]]
        output.append({"experiment": method, "dataset_group": group, "R": int(r),
                       "num_valid": len(valid), "num_failed": len(entries) - len(valid),
                       "metrics_mean": {key: float(np.mean([v[key] for v in valid if v.get(key) is not None]))
                                        if any(v.get(key) is not None for v in valid) else None for key in METRICS}})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--recon-root", type=Path, help="Full-case outputs from experiments.paper.infer; may contain multiple experiments")
    inputs.add_argument("--prediction-index", type=Path, help="Explicit CSV for our/external MAT/NPY/NPZ outputs")
    p.add_argument("--pipeline", type=Path, default=ROOT / "configs/paper/evaluation.json")
    p.add_argument("--source-mapping", type=Path, help="Override pipeline.source_mapping without editing the original manifest")
    p.add_argument("--groups", nargs="+", choices=("train", "ordinary_test", "focused_test"), default=["train"])
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--skip-errors", action="store_true")
    args = p.parse_args()
    cfg = read_pipeline(args.pipeline)
    source_mapping = args.source_mapping or cfg.get("source_mapping")
    source_mapping = ROOT / source_mapping if source_mapping else None
    cases, base_guard = select_cases(ROOT / cfg["manifest"], args.groups, args.case_ids, source_mapping=source_mapping)
    lookup = {case["case_id"]: case for case in cases}
    entries = discover(args.recon_root) if args.recon_root else read_index(args.prediction_index)
    selected = [e for e in entries if e["case_id"] in lookup]
    if not selected:
        raise ValueError("No complete predictions for the explicitly selected original dataset groups")
    protected = [*base_guard.protected, cfg["metrics"]["eval_code_dir"],
                 *(p["path"] for entry in entries for p in entry["parts"])]
    protected.extend(p for p in (args.pipeline, args.prediction_index) if p is not None)
    guard = OutputGuard(protected)
    csv_path, json_path = guard.check(args.out_csv), guard.check(args.out_json)
    guard.directory(csv_path.parent)
    guard.directory(json_path.parent)
    evaluate_arrays, funcs = official_core(cfg["metrics"]["eval_code_dir"])
    rows, sampling, seen = [], {}, set()
    # Group by case to retain just one case's GT memmap and metadata in memory.
    selected.sort(key=lambda e: (e["case_id"], e["R"], e["experiment"]))
    data = None
    for entry in selected:
        case = lookup[entry["case_id"]]
        key = (entry["experiment"], entry["case_id"], entry["R"])
        if key in seen:
            raise ValueError(f"Duplicate experiment/case/R prediction: {key}")
        seen.add(key)
        row = {"experiment": entry["experiment"], "case_id": case["case_id"],
               "dataset_group": case["dataset_group"], "anatomy": case["case_id"].split("_", 2)[-1],
               "R": entry["R"], "sampling_verified": False, "comments": ""}
        try:
            if data is None or data.case["case_id"] != case["case_id"]:
                data = FullCase(case, ROOT / cfg["cache_root"], guard)
            meta = entry["metadata"]
            if meta and (meta["dataset_group"] != case["dataset_group"] or meta["source_signature"] != data.signature):
                raise ValueError("Prediction dataset label/source signature does not match the original manifest/case")
            known_mask = meta.get("mask", {}).get("sha256") or entry["parts"][0].get("mask_sha256")
            sample_key = (case["case_id"], entry["R"])
            if known_mask:
                _, expected = data.fixed_mask(entry["R"], cfg["mask"]["seed"])
                if known_mask != expected["sha256"] or sample_key in sampling and sampling[sample_key] != known_mask:
                    raise ValueError("Methods are using different masks for this case/R/seed")
                sampling[sample_key], row["sampling_verified"] = known_mask, True
            row["venc_order"] = data.params["venc_order"]
            result = score(data, load_prediction(entry), cfg["metrics"], evaluate_arrays, funcs)
            if not all(np.isfinite(value) for value in result.values()):
                raise FloatingPointError(f"Non-finite evaluation metrics: {result}")
            row.update(result)
            data.assert_unchanged()
        except Exception as exc:
            if not args.skip_errors:
                raise
            row["comments"] = f"{type(exc).__name__}: {exc}"
            row.update({key: None for key in METRICS})
        rows.append(row)
        print(f"{row['comments'] or 'OK'} {entry['experiment']} {case['case_id']} R={entry['R']}", flush=True)
    fields = ["experiment", "case_id", "dataset_group", "anatomy", "R", "sampling_verified", "venc_order", *METRICS, "comments"]
    temp = guard.check(csv_path.with_suffix(f".tmp-{uuid.uuid4().hex}"))
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)
    guard.json(json_path, {"num_total": len(rows), "num_valid": sum(not r["comments"] for r in rows),
                           "num_failed": sum(bool(r["comments"]) for r in rows), "configuration": cfg["metrics"],
                           "metric_wrapper_sha256": sha256(ROOT / "scripts/tools/evaluate_4dflow_submission.py"),
                           "official_code_sha256": {p.name: sha256(p) for p in Path(cfg["metrics"]["eval_code_dir"]).glob("*.py")},
                           "mask_seed": cfg["mask"]["seed"], "groups": summarize(rows),
                           "source_mapping_sha256": sha256(source_mapping) if source_mapping else None,
                           "sampling_unverified_count": sum(not r["sampling_verified"] for r in rows),
                           "note": "External CSV results without a mask SHA are scoreable but sampling fairness is unverified."})
    print(f"WROTE {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
