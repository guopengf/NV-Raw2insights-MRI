#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


PHASE_FIELDS = (
    "pre_worker_start_ms",
    "worker_on_critical_path_ms",
    "pre_collate_gap_ms",
    "collate_on_critical_path_ms",
    "post_collate_on_critical_path_ms",
    "critical_path_unattributed_ms",
)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def summarize(values):
    return {
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def load_events(path, run_tag):
    events = []
    with path.open("r", encoding="utf-8") as event_file:
        for line in event_file:
            event = json.loads(line)
            if run_tag and event.get("run_tag") != run_tag:
                continue
            if not event.get("loader_critical_path"):
                continue
            events.append(event)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--run-tag")
    parser.add_argument("--slow-threshold-ms", type=float, default=500.0)
    args = parser.parse_args()

    events = load_events(args.events, args.run_tag)
    if not events:
        raise SystemExit("No matching IPC timing events found.")

    warm_events = [event for event in events if int(event.get("batch_index", 0)) > 0]
    slow_events = [
        event
        for event in warm_events
        if float(event.get("consumer_loader_wait_ms", 0.0)) >= args.slow_threshold_ms
    ]
    normal_events = [event for event in warm_events if event not in slow_events]

    print(
        f"events={len(events)} warm={len(warm_events)} slow={len(slow_events)} "
        f"slow_fraction={len(slow_events) / max(1, len(warm_events)):.3f}"
    )
    print(f"dominant_slow_phases={dict(Counter(event['loader_critical_path']['dominant_phase'] for event in slow_events))}")
    print("metric                              slow median/p95/max      normal median/p95/max")

    metric_extractors = {
        "loader_wait_ms": lambda event: float(event["consumer_loader_wait_ms"]),
        "collate_ms": lambda event: float(event["post_worker_timing"].get("collate_ms", 0.0)),
        "payload_mib": lambda event: float(event["post_worker_timing"].get("output_payload_bytes", 0.0)) / (1024**2),
        **{
            field: lambda event, field=field: float(event["loader_critical_path"].get(field, 0.0))
            for field in PHASE_FIELDS
        },
    }
    for name, extractor in metric_extractors.items():
        slow_summary = summarize([extractor(event) for event in slow_events])
        normal_summary = summarize([extractor(event) for event in normal_events])
        print(
            f"{name:34s} "
            f"{slow_summary['median']:8.1f}/{slow_summary['p95']:8.1f}/{slow_summary['max']:8.1f}  "
            f"{normal_summary['median']:8.1f}/{normal_summary['p95']:8.1f}/{normal_summary['max']:8.1f}"
        )

    closure_errors = []
    for event in warm_events:
        critical_path = event["loader_critical_path"]
        attributed = sum(float(critical_path.get(field, 0.0)) for field in PHASE_FIELDS)
        closure_errors.append(abs(float(event["consumer_loader_wait_ms"]) - attributed))
    print(f"decomposition_error_ms={summarize(closure_errors)}")
    print(f"slow_ranks={dict(sorted(Counter(event['rank'] for event in slow_events).items()))}")
    print(f"slow_workers={dict(sorted(Counter(event['worker_context'].get('worker_id', -1) for event in slow_events).items()))}")


if __name__ == "__main__":
    main()
