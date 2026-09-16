#!/usr/bin/env python3
"""Compare Regular Task 2 ComplexDiffErr against the official FlowVN baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def read_task_rows(path: Path, task: str) -> dict[str, dict]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["task"] == task]
    if not rows:
        raise RuntimeError(f"No {task} rows found in {path}")
    result = {}
    for row in rows:
        if row.get("comments"):
            raise RuntimeError(f"Invalid row in {path}: {row['rel_path']}: {row['comments']}")
        if row["rel_path"] in result:
            raise RuntimeError(f"Duplicate rel_path in {path}: {row['rel_path']}")
        result[row["rel_path"]] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-csv", type=Path, required=True)
    parser.add_argument("--flowvn-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--task", default="TaskR1R2")
    parser.add_argument("--expected-cases", type=int, default=32)
    args = parser.parse_args()

    submission = read_task_rows(args.submission_csv, args.task)
    flowvn = read_task_rows(args.flowvn_csv, args.task)
    if set(submission) != set(flowvn):
        raise RuntimeError(
            f"Path mismatch: submission_only={sorted(set(submission) - set(flowvn))[:5]} "
            f"flowvn_only={sorted(set(flowvn) - set(submission))[:5]}"
        )
    if len(submission) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} paired cases, found {len(submission)}")

    paired = []
    for rel_path in sorted(submission):
        submitted = float(submission[rel_path]["ComplexDiffErr"])
        baseline = float(flowvn[rel_path]["ComplexDiffErr"])
        paired.append(
            {
                "rel_path": rel_path,
                "R": submission[rel_path]["R"],
                "submission_ComplexDiffErr_adj": submitted,
                "flowvn_ComplexDiffErr_adj": baseline,
                "ComplexDiffErr_Diff_adj": submitted - baseline,
                "better_than_flowvn": submitted < baseline,
            }
        )

    submitted_mean = float(np.mean([row["submission_ComplexDiffErr_adj"] for row in paired]))
    baseline_mean = float(np.mean([row["flowvn_ComplexDiffErr_adj"] for row in paired]))
    better = sum(bool(row["better_than_flowvn"]) for row in paired)
    summary = {
        "schema_version": 1,
        "task": args.task,
        "num_cases": len(paired),
        "ComplexDiffErr_adj": submitted_mean,
        "FlowVN_ComplexDiffErr_adj": baseline_mean,
        "ComplexDiffErr_Diff_adj": submitted_mean - baseline_mean,
        "BetterThanFlowVN": f"{better}/{len(paired)}",
        "better_cases": better,
        "passes_accuracy_gate": submitted_mean - baseline_mean < 0,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = args.out_csv.with_suffix(args.out_csv.suffix + f".tmp-{os.getpid()}")
    with csv_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    os.replace(csv_temporary, args.out_csv)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = args.out_json.with_suffix(args.out_json.suffix + f".tmp-{os.getpid()}")
    json_temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(json_temporary, args.out_json)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

