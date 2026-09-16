#!/usr/bin/env python3
"""Select the local evaluation variant that reproduces accepted Synapse scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPECTED = {
    "TaskR1R2": {"AngErr": 32.125552, "RelErr": 0.339351, "SSIM": 0.947082, "nRMSE": 0.047038},
    "TaskS1": {"AngErr": 33.746172, "RelErr": 0.528381, "SSIM": 0.952121, "nRMSE": 0.082605},
    "TaskS2": {"AngErr": 44.806296, "RelErr": 0.536178, "SSIM": 0.949960, "nRMSE": 0.070394},
}

SYNAPSE_ROWS = {
    "TaskR1R2": {"row_id": 9780517, "entity_id": "syn77416865"},
    "TaskS1": {"row_id": 9780518, "entity_id": "syn77416863"},
    "TaskS2": {"row_id": 9780519, "entity_id": "syn77416864"},
}


def task_means(summary: dict, task: str) -> dict:
    entry = summary["by_task"][task]
    return entry["metrics_mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-5,
        help="Maximum absolute difference from six-decimal public scores.",
    )
    args = parser.parse_args()

    variants = []
    for path in args.summaries:
        summary = json.loads(path.read_text())
        differences = {}
        all_abs = []
        for task, expected_metrics in EXPECTED.items():
            observed = task_means(summary, task)
            differences[task] = {}
            for metric, expected in expected_metrics.items():
                delta = float(observed[metric]) - expected
                differences[task][metric] = {
                    "observed": float(observed[metric]),
                    "expected": expected,
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
                all_abs.append(abs(delta))
        variants.append(
            {
                "summary": str(path),
                "configuration": summary.get("configuration", {}),
                "max_abs_delta": max(all_abs),
                "mean_abs_delta": sum(all_abs) / len(all_abs),
                "differences": differences,
            }
        )

    variants.sort(key=lambda item: (item["max_abs_delta"], item["mean_abs_delta"]))
    selected = variants[0]
    passed = selected["max_abs_delta"] <= args.tolerance
    receipt = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "tolerance": args.tolerance,
        "selected_summary": selected["summary"] if passed else None,
        "selected_configuration": selected["configuration"] if passed else None,
        "expected": EXPECTED,
        "synapse_rows": SYNAPSE_ROWS,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"status": receipt["status"], "selected": receipt["selected_configuration"], "max_abs_delta": selected["max_abs_delta"]}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
