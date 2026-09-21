#!/usr/bin/env python3
"""Build a deterministic one-patient-per-acceleration-profile canary plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TOOLS_ROOT = SCRIPTS_ROOT / "tools"
sys.path[:0] = [str(SCRIPTS_ROOT), str(TOOLS_ROOT)]

from build_4dflow_windowed_h5 import (
    _discover_from_config,
    _source_bytes,
    build_conversion_plan_payload,
    write_conversion_plan,
)
from windowed_4dflow import build_patient_source_manifest


EXPECTED_PROFILES = {
    (10,),
    (20,),
    (30,),
    (40,),
    (50,),
    (10, 20, 30, 40, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--e1-output-root", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--selection-out", required=True, type=Path)
    return parser.parse_args()


def _patient_source_bytes(patient: dict) -> int:
    return _source_bytes(
        build_patient_source_manifest(
            target_path=patient["target_path"],
            acceleration_inputs=patient["inputs"],
            acceleration_masks=patient["masks"],
            coilmap_path=patient["coilmap_path"],
            segmask_path=patient["segmask_path"],
        )
    )


def main() -> None:
    cli = parse_args()
    _, patients = _discover_from_config(cli.config, 0, data_split="train")
    grouped = defaultdict(list)
    for patient in patients:
        grouped[tuple(sorted(int(value) for value in patient["inputs"]))].append(patient)
    if set(grouped) != EXPECTED_PROFILES:
        raise RuntimeError(
            f"Unexpected acceleration profiles: {sorted(grouped)}, expected {sorted(EXPECTED_PROFILES)}"
        )

    selected = []
    selection_records = []
    for profile in sorted(grouped):
        candidates = sorted(
            ((_patient_source_bytes(patient), patient["patient_key"], patient) for patient in grouped[profile]),
            key=lambda item: (item[0], item[1]),
        )
        source_bytes, _, patient = candidates[len(candidates) // 2]
        selected.append(patient)
        selection_records.append(
            {
                "profile": list(profile),
                "patient_key": patient["patient_key"],
                "target_path": str(patient["target_path"]),
                "source_bytes": source_bytes,
                "candidate_count": len(candidates),
            }
        )

    plan = build_conversion_plan_payload(
        config_path=cli.config,
        output_root=cli.e1_output_root,
        patients=selected,
        num_shards=1,
        expected_patients=len(EXPECTED_PROFILES),
    )
    write_conversion_plan(plan, cli.plan_out)
    selection_payload = {
        "status": "selected",
        "patient_count": len(selected),
        "plan_sha256": plan["plan_sha256"],
        "patients": selection_records,
    }
    cli.selection_out.parent.mkdir(parents=True, exist_ok=True)
    cli.selection_out.write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection_payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
