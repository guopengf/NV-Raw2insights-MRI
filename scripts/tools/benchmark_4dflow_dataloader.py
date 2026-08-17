#!/usr/bin/env python3
"""Build, summarize, rank, and report isolated 4D-flow dataloader trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FIXED_BATCH_SIZE = 8
FIXED_SAMPLES_PER_CASE = 8
EXPECTED_WORLD_SIZE = 16
FULL_EPOCH_BATCHES = 173


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setting_key(payload: dict[str, Any]) -> tuple[int, int, bool, int]:
    return (
        int(payload["num_workers"]),
        int(payload["train_prefetch_factor"]),
        bool(payload["train_pin_memory"]),
        int(payload["omp_num_threads"]),
    )


def make_config(args: argparse.Namespace) -> None:
    base_path = Path(args.base_config).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root)
    if run_root.is_absolute():
        raise ValueError("--run-root must be relative to --repo-root for container portability")

    with base_path.open(encoding="utf-8") as source:
        config = json.load(source)

    if int(config.get("batch_size", -1)) != FIXED_BATCH_SIZE:
        raise ValueError(
            f"Base config batch_size must stay fixed at {FIXED_BATCH_SIZE}; "
            f"found {config.get('batch_size')!r}"
        )
    if int(config.get("num_samples_per_case", -1)) != FIXED_SAMPLES_PER_CASE:
        raise ValueError(
            f"Base config num_samples_per_case must stay fixed at {FIXED_SAMPLES_PER_CASE}; "
            f"found {config.get('num_samples_per_case')!r}"
        )

    trial_dir = repo_root / run_root / "trials" / args.tag
    trial_dir.mkdir(parents=True, exist_ok=False)
    relative_trial_parent = (run_root / "trials").as_posix().rstrip("/") + "/"

    timing = dict(config.get("performance_timing", {}))
    timing.update(
        {
            "enabled": bool(args.detailed),
            "sample_interval": 1 if args.detailed else 10,
            "cuda_synchronize": False,
            "log_rank_details": bool(args.detailed),
            "worker_timing_enabled": bool(args.detailed),
            "post_worker_timing_enabled": bool(args.detailed),
            "lightweight_enabled": True,
            "worker_loader_max_events_per_rank": max(300, int(args.max_batches)),
            "slow_loader_threshold_ms": 500.0,
            "slow_loader_max_events_per_rank": max(100, int(args.max_batches)),
            "debug_max_train_batches": int(args.max_batches),
            "run_tag": args.tag,
        }
    )

    config.update(
        {
            "batch_size": FIXED_BATCH_SIZE,
            "num_samples_per_case": FIXED_SAMPLES_PER_CASE,
            "num_workers": int(args.workers),
            "train_prefetch_factor": max(1, int(args.prefetch)),
            "train_pin_memory": bool(args.pin_memory),
            "train_singleton_view_collate": True,
            "use_multi_epochs_train_loader": True,
            "cache_rate": 0.0,
            "seed": int(args.seed),
            "val": False,
            "val_interval": 1000000,
            "resume_rng_state": False,
            "reinit_wandb": True,
            "enable_onelogger": False,
            "exp_dir": relative_trial_parent,
            "exp": args.tag,
            "model_filename": "benchmark_no_checkpoint.pt",
            "performance_timing": timing,
        }
    )

    config_path = trial_dir / "effective_config.json"
    with config_path.open("w", encoding="utf-8") as destination:
        json.dump(config, destination, indent=2, sort_keys=True)
        destination.write("\n")

    manifest = {
        "tag": args.tag,
        "phase": args.phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_path),
        "base_config_sha256": sha256(base_path),
        "effective_config": str(config_path),
        "num_workers": int(args.workers),
        "train_prefetch_factor": max(1, int(args.prefetch)),
        "train_pin_memory": bool(args.pin_memory),
        "omp_num_threads": int(args.omp_threads),
        "batch_size": FIXED_BATCH_SIZE,
        "num_samples_per_case": FIXED_SAMPLES_PER_CASE,
        "max_batches": int(args.max_batches),
        "warmup_batches": int(args.warmup_batches),
        "seed": int(args.seed),
        "detailed": bool(args.detailed),
        "expected_world_size": EXPECTED_WORLD_SIZE,
        "full_epoch_batches": FULL_EPOCH_BATCHES,
    }
    with (trial_dir / "trial_manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")

    print(config_path.relative_to(repo_root))


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_worker_events(trial_dir: Path) -> dict[str, Any]:
    events = read_jsonl(trial_dir / "performance" / "worker_loader_events.jsonl")
    if not events:
        return {}
    fields: dict[str, list[float]] = defaultdict(list)
    for event in events:
        for section_name in ("worker_timing", "post_worker_timing", "worker_resource"):
            section = event.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for name, value in section.items():
                if isinstance(value, (int, float)) and not name.startswith("_"):
                    fields[f"{section_name}.{name}"].append(float(value))
    return {
        "events": len(events),
        "fields": {name: numeric_summary(values) for name, values in sorted(fields.items())},
    }


def summarize_telemetry(trial_dir: Path) -> dict[str, Any]:
    gpu_utilization: list[float] = []
    gpu_memory_mib: list[float] = []
    gpu_power_w: list[float] = []
    for path in sorted((trial_dir / "telemetry").glob("gpu_*.csv")):
        with path.open(encoding="utf-8") as source:
            for row in csv.DictReader(source):
                try:
                    gpu_utilization.append(float(row["utilization_gpu_percent"]))
                    gpu_memory_mib.append(float(row["memory_used_mib"]))
                    gpu_power_w.append(float(row["power_draw_w"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return {
        "gpu_utilization_percent": numeric_summary(gpu_utilization),
        "gpu_memory_used_mib": numeric_summary(gpu_memory_mib),
        "gpu_power_draw_w": numeric_summary(gpu_power_w),
    }


def summarize(args: argparse.Namespace) -> None:
    trial_dir = Path(args.trial_dir).resolve()
    with (trial_dir / "trial_manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)

    exit_codes = []
    for path in sorted(trial_dir.glob("exit_code_node*.txt")):
        try:
            exit_codes.append(int(path.read_text(encoding="utf-8").strip()))
        except ValueError:
            exit_codes.append(999)

    events = read_jsonl(trial_dir / "performance" / "lightweight_events.jsonl")
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_batch[int(event["batch_index"])].append(event)

    expected_world_size = int(manifest.get("expected_world_size", EXPECTED_WORLD_SIZE))
    complete_batches = {
        batch_index: batch_events
        for batch_index, batch_events in by_batch.items()
        if len({int(event["rank"]) for event in batch_events}) == expected_world_size
    }
    warmup_batches = int(manifest["warmup_batches"])
    measured = {
        batch_index: batch_events
        for batch_index, batch_events in complete_batches.items()
        if batch_index >= warmup_batches
    }

    critical_intervals = [
        max(float(event["batch_interval_ms"]) for event in batch_events)
        for _, batch_events in sorted(measured.items())
    ]
    critical_loader_waits = [
        max(float(event["loader_wait_ms"]) for event in batch_events)
        for _, batch_events in sorted(measured.items())
    ]
    event_waits = [
        float(event["loader_wait_ms"])
        for batch_events in measured.values()
        for event in batch_events
    ]
    event_computes = [
        float(event["compute_ms"])
        for batch_events in measured.values()
        for event in batch_events
    ]
    all_critical_intervals = [
        max(float(event["batch_interval_ms"]) for event in batch_events)
        for _, batch_events in sorted(complete_batches.items())
    ]

    required_measured_batches = max(1, int(manifest["max_batches"]) - warmup_batches)
    valid = (
        len(exit_codes) == 2
        and all(code == 0 for code in exit_codes)
        and len(complete_batches) == int(manifest["max_batches"])
        and len(measured) == required_measured_batches
        and bool(critical_intervals)
    )
    critical_mean = statistics.fmean(critical_intervals) if critical_intervals else None

    summary = {
        **manifest,
        "valid": valid,
        "exit_codes": exit_codes,
        "event_count": len(events),
        "complete_batches": len(complete_batches),
        "measured_batches": len(measured),
        "critical_interval_ms": numeric_summary(critical_intervals),
        "critical_loader_wait_ms": numeric_summary(critical_loader_waits),
        "event_loader_wait_ms": numeric_summary(event_waits),
        "event_compute_ms": numeric_summary(event_computes),
        "slow_loader_events_over_500ms": sum(wait > 500.0 for wait in event_waits),
        "observed_all_batches_minutes": sum(all_critical_intervals) / 60000.0,
        "projected_epoch_minutes": (
            critical_mean * FULL_EPOCH_BATCHES / 60000.0 if critical_mean is not None else None
        ),
        "worker_profile": summarize_worker_events(trial_dir),
        "telemetry": summarize_telemetry(trial_dir),
    }
    with (trial_dir / "benchmark_summary.json").open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")

    score = summary["critical_interval_ms"].get("mean")
    score_text = f"{score:.1f}" if score is not None else "unavailable"
    epoch_value = summary["projected_epoch_minutes"]
    epoch_text = f"{epoch_value:.2f}" if epoch_value is not None else "unavailable"
    print(
        f"[summary] tag={manifest['tag']} valid={valid} batches={len(complete_batches)} "
        f"score_ms={score_text} projected_epoch_min={epoch_text}"
    )


def load_summaries(run_root: Path, phases: set[str] | None = None) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((run_root / "trials").glob("*/benchmark_summary.json")):
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        if not payload.get("valid"):
            continue
        if phases is not None and payload.get("phase") not in phases:
            continue
        summaries.append(payload)
    return summaries


def rank_settings(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    phases = set(filter(None, (args.phases or "").split(","))) or None
    summaries = load_summaries(run_root, phases)
    grouped: dict[tuple[int, int, bool, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[setting_key(summary)].append(summary)

    ranked = []
    for key, runs in grouped.items():
        if len(runs) < int(args.min_runs):
            continue
        scores = [float(run["critical_interval_ms"]["mean"]) for run in runs]
        ranked.append((statistics.median(scores), key, runs))
    ranked.sort(key=lambda item: item[0])

    if args.unique_workers:
        distinct = []
        seen_workers = set()
        for item in ranked:
            workers = item[1][0]
            if workers <= 0 or workers in seen_workers:
                continue
            distinct.append(item)
            seen_workers.add(workers)
        ranked = distinct

    selected = ranked[: int(args.top)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for score, key, runs in selected:
            workers, prefetch, pin_memory, omp_threads = key
            destination.write(
                f"{workers}\t{prefetch}\t{int(pin_memory)}\t{omp_threads}\t{score:.6f}\t{runs[0]['tag']}\n"
            )
    print(f"[rank] candidates={len(ranked)} selected={len(selected)} output={output_path}")


def render_report(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    summaries = load_summaries(run_root)
    summaries.sort(key=lambda item: float(item["critical_interval_ms"]["mean"]))
    confirmations = [summary for summary in summaries if summary.get("phase") == "confirm"]
    recommendation_pool = confirmations or summaries
    recommendation = recommendation_pool[0] if recommendation_pool else None

    lines = [
        "# 4D-flow dataloader tuning report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Fixed training semantics",
        "",
        f"- `batch_size={FIXED_BATCH_SIZE}`",
        f"- `num_samples_per_case={FIXED_SAMPLES_PER_CASE}`",
        "- Two nodes, eight GPUs per node, sixteen DDP ranks",
        "- Model, optimizer, loss, dataset, and checkpoint selection unchanged",
        "",
        "## Recommendation",
        "",
    ]
    if recommendation is None:
        lines.append("No valid trial completed.")
    else:
        lines.extend(
            [
                f"- `num_workers={recommendation['num_workers']}`",
                f"- `train_prefetch_factor={recommendation['train_prefetch_factor']}`",
                f"- `train_pin_memory={str(recommendation['train_pin_memory']).lower()}`",
                f"- `OMP_NUM_THREADS={recommendation['omp_num_threads']}`",
                f"- Projected epoch time: {recommendation['projected_epoch_minutes']:.2f} minutes",
            ]
        )

    lines.extend(
        [
            "",
            "## Valid trials",
            "",
            "| Phase | Tag | Workers | Prefetch | Pin | OMP | Batches | Critical mean (ms) | Critical p95 (ms) | Loader p99 (ms) | Projected epoch (min) | GPU util mean (%) |",
            "|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        gpu_mean = summary.get("telemetry", {}).get("gpu_utilization_percent", {}).get("mean", math.nan)
        lines.append(
            "| {phase} | `{tag}` | {workers} | {prefetch} | {pin} | {omp} | {batches} | "
            "{mean:.1f} | {p95:.1f} | {loader_p99:.1f} | {epoch:.2f} | {gpu:.1f} |".format(
                phase=summary["phase"],
                tag=summary["tag"],
                workers=summary["num_workers"],
                prefetch=summary["train_prefetch_factor"],
                pin="yes" if summary["train_pin_memory"] else "no",
                omp=summary["omp_num_threads"],
                batches=summary["complete_batches"],
                mean=summary["critical_interval_ms"]["mean"],
                p95=summary["critical_interval_ms"]["p95"],
                loader_p99=summary["critical_loader_wait_ms"]["p99"],
                epoch=summary["projected_epoch_minutes"],
                gpu=gpu_mean,
            )
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "Short trials discard initial warm-up batches. Rankings use the maximum batch interval across all sixteen ranks at each DDP step. Finalists are repeated with a second seed and the top two are confirmed over 173 batches. Benchmark runs use the normal forward/backward path and exit before checkpoint saving.",
            "",
            "## Artifacts",
            "",
            f"- Run root: `{run_root}`",
            "- Each trial contains its effective config, manifest, per-rank timing events, telemetry, logs, and machine-readable summary.",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] output={output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-config")
    make.add_argument("--base-config", required=True)
    make.add_argument("--repo-root", required=True)
    make.add_argument("--run-root", required=True)
    make.add_argument("--tag", required=True)
    make.add_argument("--phase", required=True)
    make.add_argument("--workers", type=int, required=True)
    make.add_argument("--prefetch", type=int, required=True)
    make.add_argument("--pin-memory", type=parse_bool, required=True)
    make.add_argument("--omp-threads", type=int, required=True)
    make.add_argument("--max-batches", type=int, required=True)
    make.add_argument("--warmup-batches", type=int, required=True)
    make.add_argument("--seed", type=int, required=True)
    make.add_argument("--detailed", action="store_true")
    make.set_defaults(func=make_config)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--trial-dir", required=True)
    summary_parser.set_defaults(func=summarize)

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--run-root", required=True)
    rank_parser.add_argument("--phases", default="")
    rank_parser.add_argument("--top", type=int, required=True)
    rank_parser.add_argument("--min-runs", type=int, default=1)
    rank_parser.add_argument("--unique-workers", action="store_true")
    rank_parser.add_argument("--output", required=True)
    rank_parser.set_defaults(func=rank_settings)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-root", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=render_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
