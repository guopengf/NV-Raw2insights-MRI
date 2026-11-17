"""Reconstruct complete cases for ours/baselines, keeping all time, x and VENC indices."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .common.case_data import FullCase, read_pipeline, select_cases
from .common.safety import ROOT, sha256
from .common.windows import Assembly, windows
from .predictor import LEARNED, Predictor


@contextmanager
def output_lock(directory, guard):
    lock = guard.check(directory / "inference.lock")
    try:
        with lock.open("x") as stream:
            json.dump({"pid": os.getpid(), "started": time.time()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Output is locked; check the running/unfinished inference before removing {lock}") from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def reconstruct(data, predictor, directory, guard, request, mask, *, overwrite=False):
    metadata = guard.check(directory / "reconstruction.json")
    destination = guard.check(directory / "reconstruction.npy")
    shape = tuple(request["shape"])
    with output_lock(directory, guard):
        if metadata.exists() and not overwrite:
            previous = json.loads(metadata.read_text())
            if any(previous.get(key) != value for key, value in request.items()):
                raise ValueError(f"Existing output has different weights/mask/config/window/sources; use another root or --overwrite: {directory}")
            if not previous.get("coverage", {}).get("complete") or not destination.is_file():
                raise ValueError(f"Incomplete output; rerun with --overwrite: {directory}")
            saved = np.load(destination, mmap_mode="r", allow_pickle=False)
            if saved.shape != shape or not np.iscomplexobj(saved):
                raise ValueError(f"Invalid existing reconstruction array: {destination}")
            print(f"SKIP {directory}", flush=True)
            return
        # The marker is published last; failed overwrites must not expose old metadata.
        metadata.unlink(missing_ok=True)
        start = time.monotonic()
        temp = guard.check(directory / f"reconstruction.{uuid.uuid4().hex}.tmp.npy")
        try:
            output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.complex64, shape=shape)
            coverage = np.zeros((shape[1], shape[-1]), np.uint32)
            assembly = Assembly(output, coverage)
            count, unconverged = 0, 0
            window_settings = request["window"]
            for window in windows(shape[1], shape[-1], window_settings["frames"], window_settings["slices"],
                                  center_time=request["method"] == "ours", merge=window_settings["merge"]):
                batch = data.batch(window.frames, window.slices, request["R"], mask)
                prediction = predictor(batch)
                assembly.add(prediction, window, float(batch["scale"][0]))
                details = predictor.provenance.get("last_solver_status", {})
                unconverged += details.get("cg_solves", 0) - details.get("cg_converged", 0)
                count += 1
                if count == 1 or count % 25 == 0:
                    print(f"INFER {request['experiment']} {request['case_id']} R={request['R']} windows={count} covered={np.count_nonzero(coverage)}/{coverage.size}", flush=True)
            completed = assembly.finish()
            data.assert_unchanged()
            output.flush()
            del output, assembly
            temp.replace(destination)
            guard.json(metadata, {**predictor.provenance, **request, "schema": 1, "scope": "full_case",
                                  "amplitude_domain": "unnormalized; unit-RSS external coil combination",
                                  "coverage": completed, "venc_order": data.params["venc_order"],
                                  "reference_encoding": 0, "windows": count, "seconds": time.monotonic() - start,
                                  "cg_unconverged_solves": unconverged})
            print(f"COMPLETE {metadata} shape={shape}", flush=True)
        finally:
            temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Experiment/model config")
    p.add_argument("--pipeline", type=Path, default=ROOT / "configs/paper/evaluation.json")
    p.add_argument("--source-mapping", type=Path, help="Override pipeline.source_mapping without editing the original manifest")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--groups", nargs="+", choices=("train", "ordinary_test", "focused_test"), default=["train"])
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--limit-cases", type=int)
    p.add_argument("--accelerations", nargs="+", type=int)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--merge", choices=("center", "tile", "overlap"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Run one process per experiment/output; this entry is not DDP")
    cfg, pipeline = json.loads(args.config.read_text()), read_pipeline(args.pipeline)
    source_mapping = args.source_mapping or pipeline.get("source_mapping")
    source_mapping = ROOT / source_mapping if source_mapping else None
    selected, guard = select_cases(ROOT / pipeline["manifest"], args.groups, args.case_ids, args.limit_cases,
                                   source_mapping=source_mapping)
    root = guard.directory(args.output_root or ROOT / pipeline["output_root"])
    cache = guard.directory(ROOT / pipeline["cache_root"])
    method, name = cfg["paper"]["method"], cfg["paper"]["name"]
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Experiment name must be a single path component")
    checkpoint = args.checkpoint or cfg["paper"].get("checkpoint")
    if method in LEARNED and checkpoint is None:
        checkpoint = ROOT / cfg["paper"]["output_dir"] / "train/latest.pt"
    predictor = Predictor(cfg, args.device, checkpoint)
    config_sha, pipeline_sha = sha256(args.config), sha256(args.pipeline)
    accelerations = args.accelerations or pipeline["mask"]["accelerations"]
    for case in selected:
        data = FullCase(case, cache, guard)
        _, nt, _, nz, ny, nx = data.shape
        for r in accelerations:
            mask, mask_info = data.fixed_mask(r, pipeline["mask"]["seed"])
            directory = guard.directory(root / name / case["dataset_group"] / case["case_id"] / f"R{r}")
            if method in {"cg_sense", "cs_tv"}:
                ns, nf, merge = nx, 1, "tile"
            else:
                ns, nf = cfg["paper"]["num_slices"], cfg["paper"]["num_frames"]
                merge = args.merge or pipeline["inference"]["ours_merge" if method == "ours" else "neural_merge"]
            request = {"experiment": name, "method": method, "case_id": case["case_id"],
                       "dataset_group": case["dataset_group"], "case_directory": case["directory"],
                       "R": int(r), "seed": pipeline["mask"]["seed"], "shape": [4, nt, nz, ny, nx],
                       "axes": "etzyx", "complex": True, "source_signature": data.signature,
                       "config_sha256": config_sha, "pipeline_sha256": pipeline_sha,
                       "checkpoint_sha256": predictor.provenance.get("checkpoint_sha256"),
                       "mask": mask_info, "window": {"slices": ns, "frames": nf, "merge": merge}}
            if source_mapping is not None:
                request["source_mapping_sha256"] = sha256(source_mapping)
            reconstruct(data, predictor, directory, guard, request, mask, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
