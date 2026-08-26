#!/usr/bin/env python3
"""Compare raw-MAT and windowed-HDF5 preprocessing for one real patient."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from monai.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from joint_encoding import joint_encoding_spec
from train_utils import get_train_transforms
from utils import load_config
from windowed_4dflow import Windowed4DFlowDataset, build_windowed_4dflow_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-manifests", type=int, default=5)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Measure only the windowed store when the raw baseline is already available.",
    )
    return parser.parse_args()


def _raw_manifest(patient: dict, acceleration: int, output: Path, *, joint: bool) -> Path:
    sources = patient["source_manifest"]
    payload = {
        "kspace": sources["inputs"][str(acceleration)]["path"],
        "target_kspace": sources["target"]["path"],
        "mask": [sources["masks"][str(acceleration)]["path"]],
        "mask_type": f"ktGaussian{acceleration}",
        "acquisition": "Flow4d",
        "is_4dflow": True,
    }
    if sources.get("coilmap"):
        payload["coilmap"] = sources["coilmap"]["path"]
    if sources.get("segmask"):
        payload["segmask"] = sources["segmask"]["path"]
    if joint:
        payload["joint_encodings"] = True
        payload["encoding_indices"] = [0, 1, 2, 3]
        suffix = "joint4"
    else:
        payload["joint_encodings"] = False
        payload["encoding_idx"] = 0
        suffix = "enc0"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"acc{acceleration}_{suffix}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _measure(dataset, count: int, seed: int) -> tuple[list[float], list[dict]]:
    timings = []
    shapes = []
    for index in range(count):
        np.random.seed(seed + index)
        started = time.perf_counter()
        sample = dataset[index]
        timings.append((time.perf_counter() - started) * 1000.0)
        shapes.append(
            {
                "input": list(sample["kspace_masked_ifft"].shape),
                "target": list(sample["kspace_ifft"].shape),
                "mask": list(sample["mask"].shape),
            }
        )
    return timings, shapes


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    cli.work_dir.mkdir(parents=True, exist_ok=True)
    with cli.index_path.open() as stream:
        index = json.load(stream)
    patient = index["patients"][0]
    accelerations = [
        value for value in config.four_dflow_accelerations if int(value) in patient["accelerations"]
    ][: cli.max_manifests]
    if not accelerations:
        raise RuntimeError("No requested acceleration is available in the converted store")
    joint = joint_encoding_spec(config).enabled

    raw_manifests = []
    if not cli.skip_raw:
        raw_manifests = [
            _raw_manifest(patient, int(acceleration), cli.work_dir / "raw", joint=joint)
            for acceleration in accelerations
        ]
    h5_manifests = build_windowed_4dflow_manifests(
        index_path=cli.index_path,
        data_roots=config.data_path_train,
        out_dir=cli.work_dir / "windowed",
        accelerations=accelerations,
        encodings=[0] if not joint else config.four_dflow_encodings,
        joint_encodings=joint,
        allow_profile_mismatch=True,
    )
    h5_dataset = Windowed4DFlowDataset(
        [{"kspace": path} for path in h5_manifests], config
    )

    raw_ms = []
    raw_shapes = []
    if raw_manifests:
        raw_dataset = Dataset(
            data=[{"kspace": path} for path in raw_manifests],
            transform=get_train_transforms(config),
        )
        raw_ms, raw_shapes = _measure(raw_dataset, len(raw_manifests), seed=100)
    h5_ms, h5_shapes = _measure(h5_dataset, len(h5_manifests), seed=100)
    raw_total_ms = float(sum(raw_ms)) if raw_ms else None
    speedup = raw_total_ms / sum(h5_ms) if raw_total_ms is not None else None
    payload = {
        "config": str(cli.config),
        "index_path": str(cli.index_path),
        "patient_key": patient["patient_key"],
        "joint_encodings": joint,
        "storage_profile": patient["storage_profile"],
        "encoding_chunk_size": int(patient["encoding_chunk_size"]),
        "batch_size": int(config.batch_size),
        "num_samples_per_case": int(config.num_samples_per_case),
        "accelerations": [int(value) for value in accelerations],
        "raw_ms": raw_ms,
        "windowed_ms": h5_ms,
        "raw_total_ms": raw_total_ms,
        "windowed_total_ms": float(sum(h5_ms)),
        "speedup": speedup,
        "raw_shapes": raw_shapes,
        "windowed_shapes": h5_shapes,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
