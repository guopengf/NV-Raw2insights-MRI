"""Synthetic-only end-to-end check; never substitute phantom scores for MRI results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from .common.case_data import FullCase, read_pipeline, select_cases
from .common.prediction_io import discover, load_prediction
from .common.safety import ROOT, OutputGuard, sha256


def prepare(root, guard):
    marker = root / "SYNTHETIC_ONLY.json"
    if marker.exists():
        return root / "manifests/pipeline.json"
    source = guard.directory(root / "inputs/SYNTHETIC_Case001_Aorta")
    manifests = guard.directory(root / "manifests")
    e, nt, nc, z, y, x = 4, 3, 2, 16, 16, 4
    tg, zg, yg, xg = np.indices((nt, z, y, x))
    vessel = np.exp(-((zg - 7.5) ** 2 + (yg - 7.5) ** 2) / 12)
    magnitude = 0.2 + vessel * (1 + 0.1 * tg) + 0.02 * xg
    truth = np.stack([magnitude * np.exp(1j * (0.03 * tg + 0.02 * xg + ei * (0.15 + 0.3 * vessel)))
                      for ei in range(e)]).astype(np.complex64)
    coils = np.ones((nc, z, y, x), np.complex64) / np.sqrt(nc)
    coils[1] *= np.exp(0.7j)
    coil_images = truth[:, :, None] * coils[None, None]
    axes = (-3, -2, -1)
    raw = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(coil_images, axes=axes), axes=axes, norm="ortho"), axes=axes).astype(np.complex64)
    seg = (((zg[0] - 7.5) ** 2 + (yg[0] - 7.5) ** 2) < 20).astype(np.uint8)
    arrays = {"kdata_full.mat": ("kdata_full", raw), "coilmap.mat": ("coilmap", coils),
              "segmask.mat": ("segmask", seg), "segmask_new.mat": ("segmask", np.repeat(seg[None], nt, axis=0))}
    for name, (key, array) in arrays.items():
        with h5py.File(guard.check(source / name), "w") as f:
            f[key] = array
    with guard.check(source / "params.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("VENC", "venc_order", "spatial_order", "matrix_size"))
        writer.writeheader()
        writer.writerow({"VENC": "100;150;200", "venc_order": "FH;AP;RL", "spatial_order": "FH;AP;RL",
                         "matrix_size": ";".join(map(str, raw.shape))})
    np.savez_compressed(guard.check(source / "corrmap.npz"), coords=np.zeros((0, 5), np.int32),
                        data=np.array([], np.float32), shape=np.array((3, 1, z, y, x)))
    manifest = guard.check(manifests / "synthetic_manifest.csv")
    with manifest.open("w", newline="") as f:
        fields = ("target_file", "size_bytes", "shape", "dataset_group", "disk", "destination_path")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in (*arrays, "params.csv", "corrmap.npz"):
            writer.writerow({"target_file": f"/{source.name}/{name}", "size_bytes": (source / name).stat().st_size,
                             "shape": json.dumps(arrays[name][1].shape) if name in arrays else "[]",
                             "dataset_group": "train", "disk": "SYNTHETIC_NOT_REAL_TRAIN", "destination_path": str(source / name)})
    pipeline = read_pipeline(ROOT / "configs/paper/evaluation.json")
    pipeline.update(manifest=str(manifest), cache_root=str(root / "cache"), output_root=str(root / "reconstructions"))
    pipeline["mask"]["accelerations"] = [10]
    pipeline["synthetic_only"] = True
    guard.json(manifests / "pipeline.json", pipeline)
    guard.json(marker, {"synthetic_only": True, "native_shape": list(raw.shape),
                        "real_train_or_test_data_read": False, "manifest_sha256": sha256(manifest),
                        "purpose": "Full time/slice/encoding integration, not reconstruction quality assessment"})
    return manifests / "pipeline.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "outputs/paper/full_case_check")
    p.add_argument("--run", action="store_true", help="Run all selected networks/solvers using previous local smoke checkpoints")
    p.add_argument("--experiments", nargs="+")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    guard = OutputGuard()
    root = guard.directory(args.root)
    pipeline = prepare(root, guard)
    print(f"SYNTHETIC ONLY: {pipeline}", flush=True)
    if not args.run:
        return
    registry = json.loads((ROOT / "experiments/paper/experiment_registry.json").read_text())["experiments"]
    requested = set(args.experiments or [e["name"] for e in registry])
    if requested - {e["name"] for e in registry}:
        p.error(f"Unknown experiments: {requested - {e['name'] for e in registry}}")
    report = {"synthetic_only": True, "real_train_or_test_data_read": False, "experiments": []}
    logs = guard.directory(root / "logs")
    env = {**os.environ, "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for entry in registry:
        if entry["name"] not in requested:
            continue
        config = ROOT / entry["config"]
        cfg = json.loads(config.read_text())
        command = [sys.executable, "-m", "experiments.paper.infer", "--config", str(config),
                   "--pipeline", str(pipeline), "--device", args.device]
        if cfg["paper"]["method"] not in {"cg_sense", "cs_tv"}:
            command += ["--checkpoint", str(ROOT / entry["output_dir"] / "smoke/latest.pt")]
        if args.overwrite:
            command += ["--overwrite"]
        log = guard.check(logs / f"{entry['name']}.log")
        with log.open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        row = {"experiment": entry["name"], "exit_code": result.returncode, "command": command, "log": str(log)}
        report["experiments"].append(row)
        guard.json(root / "report.json", report)
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {entry['name']} {log}", flush=True)
    predictions = discover(root / "reconstructions")
    report["predictions"] = [{"experiment": e["experiment"], "shape": list(load_prediction(e).shape),
                               "coverage": e["metadata"]["coverage"], "mask_sha256": e["metadata"]["mask"]["sha256"]}
                              for e in predictions]
    if predictions:
        command = [sys.executable, "-m", "experiments.paper.evaluate", "--pipeline", str(pipeline),
                   "--recon-root", str(root / "reconstructions"), "--out-csv", str(root / "metrics/metrics.csv"),
                   "--out-json", str(root / "metrics/metrics.json")]
        with guard.check(logs / "evaluate.log").open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        report["evaluation_exit_code"] = result.returncode
        if result.returncode == 0:
            from .evaluate import official_core, score

            cfg = read_pipeline(pipeline)
            cases, raw_guard = select_cases(cfg["manifest"])
            data = FullCase(cases[0], cfg["cache_root"], raw_guard)
            evaluate_arrays, funcs = official_core(cfg["metrics"]["eval_code_dir"])
            exact = score(data, np.array(data.gt), cfg["metrics"], evaluate_arrays, funcs)
            report["known_truth_self_metrics"] = exact
            if abs(exact["SSIM"] - 1) > 1e-5 or any(abs(exact[k]) > 1e-6 for k in ("nRMSE", "RelErr", "ComplexDiffErr")):
                raise AssertionError(f"Known-truth metric self-check failed: {exact}")
            data.assert_unchanged()
    else:
        report["evaluation_exit_code"] = None
    report["passed"] = (all(e["exit_code"] == 0 for e in report["experiments"]) and
                        report["evaluation_exit_code"] == 0 and len(predictions) >= len(requested))
    guard.json(root / "report.json", report)
    print(f"SYNTHETIC INTEGRATION {'PASS' if report['passed'] else 'FAIL'}: {root / 'report.json'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
