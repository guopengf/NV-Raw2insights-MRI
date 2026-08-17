#!/usr/bin/env python3
"""Benchmark repeated full-target reads for one patient's acceleration group."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from readers import CMRxReconReader  # noqa: E402
from utils import load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=5)
    return parser.parse_args()


def load_groups(manifest_dir: Path):
    groups = defaultdict(list)
    payloads = {}
    for path in sorted(manifest_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        target_path = payload.get("target_kspace", payload.get("gt_kspace", payload.get("full_kspace")))
        if payload.get("is_4dflow") and payload.get("joint_encodings") and target_path:
            groups[str(target_path)].append(path)
            payloads[path] = payload
    return groups, payloads


def choose_median_group(groups, group_size: int):
    candidates = []
    for target_path, manifests in groups.items():
        if len(manifests) == group_size and Path(target_path).is_file():
            candidates.append((Path(target_path).stat().st_size, target_path, manifests))
    if not candidates:
        raise RuntimeError(f"No complete {group_size}-manifest target group was found")
    candidates.sort(key=lambda item: item[0])
    return candidates[len(candidates) // 2]


def run_trial(args, manifests, target_cache_entries: int, coilmap_cache_entries: int):
    args.reader_target_cache_entries = target_cache_entries
    args.reader_coilmap_cache_entries = coilmap_cache_entries
    args.performance_timing.worker_timing_enabled = True
    reader = CMRxReconReader(fixed_mask_types=args.train_mask_types, args=args)
    rows = []
    started = time.perf_counter()
    for manifest in manifests:
        sample = reader.read(manifest)
        timing = dict(sample["worker_timing"])
        rows.append(
            {
                "manifest": manifest.name,
                "target_read_ms": timing.get("target_read_ms", 0.0),
                "target_cache_hit": bool(timing.get("target_cache_hit", 0.0)),
                "coilmap_read_ms": timing.get("coilmap_read_ms", 0.0),
                "coilmap_cache_hit": bool(timing.get("coilmap_cache_hit", 0.0)),
                "input_read_ms": timing.get("input_read_ms", 0.0),
            }
        )
        del sample
        gc.collect()
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "cache": {
            "target_entries": target_cache_entries,
            "coilmap_entries": coilmap_cache_entries,
        },
        "wall_ms": wall_ms,
        "target_reads": sum(not row["target_cache_hit"] for row in rows),
        "target_cache_hits": sum(row["target_cache_hit"] for row in rows),
        "coilmap_reads": sum(not row["coilmap_cache_hit"] for row in rows),
        "coilmap_cache_hits": sum(row["coilmap_cache_hit"] for row in rows),
        "target_read_ms_total": sum(row["target_read_ms"] for row in rows),
        "input_read_ms_median": statistics.median(row["input_read_ms"] for row in rows),
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0),
        "rows": rows,
    }


def main():
    cli = parse_args()
    args = load_config(cli.config)
    if args is None:
        raise RuntimeError(f"Could not load config: {cli.config}")
    groups, payloads = load_groups(cli.manifest_dir)
    target_size, target_path, manifests = choose_median_group(groups, cli.group_size)
    manifests = sorted(
        manifests,
        key=lambda path: (
            float(payloads[path].get("acceleration", 0.0)),
            str(path),
        ),
    )
    selected = {
        "target_path": target_path,
        "target_size_gib": target_size / (1024.0**3),
        "manifests": [path.name for path in manifests],
    }
    print(json.dumps({"selected_group": selected}), flush=True)

    uncached = run_trial(args, manifests, target_cache_entries=0, coilmap_cache_entries=0)
    print(json.dumps({"uncached": uncached}), flush=True)
    cached = run_trial(args, manifests, target_cache_entries=1, coilmap_cache_entries=1)
    print(json.dumps({"cached": cached}), flush=True)

    summary = {
        "selected_group": selected,
        "uncached_target_reads": uncached["target_reads"],
        "cached_target_reads": cached["target_reads"],
        "uncached_coilmap_reads": uncached["coilmap_reads"],
        "cached_coilmap_reads": cached["coilmap_reads"],
        "target_read_time_reduction_percent": 100.0
        * (1.0 - cached["target_read_ms_total"] / max(uncached["target_read_ms_total"], 1e-9)),
        "wall_time_reduction_percent": 100.0
        * (1.0 - cached["wall_ms"] / max(uncached["wall_ms"], 1e-9)),
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
