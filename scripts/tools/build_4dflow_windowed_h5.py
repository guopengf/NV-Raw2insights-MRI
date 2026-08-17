#!/usr/bin/env python3
"""Convert raw 4D-flow MAT files into restart-safe windowed HDF5 stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils import load_config
from windowed_4dflow import (
    convert_patient_to_windowed_hdf5,
    discover_4dflow_patients,
    rebuild_windowed_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit-patients", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--deep-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.verify_only:
        index = rebuild_windowed_index(cli.output_root, deep=cli.deep_verify)
        print(json.dumps({"status": "verified", "patients": len(index["patients"])}, sort_keys=True))
        return

    config = load_config(cli.config)
    if config is None:
        raise RuntimeError(f"Could not load config: {cli.config}")
    roots = list(dict.fromkeys([*config.data_path_train, *config.data_path_val]))
    accelerations = [int(value) for value in config.four_dflow_accelerations]
    patients = discover_4dflow_patients(roots, accelerations)
    if cli.limit_patients > 0:
        patients = patients[: cli.limit_patients]
    if not patients:
        raise RuntimeError("No fully populated 4D-flow patients were discovered")

    for patient_index, patient in enumerate(patients, start=1):
        output_path = cli.output_root / "patients" / f"{patient['patient_key']}.h5"
        record = convert_patient_to_windowed_hdf5(
            patient_key=patient["patient_key"],
            target_path=patient["target_path"],
            acceleration_inputs=patient["inputs"],
            acceleration_masks=patient["masks"],
            output_path=output_path,
            coilmap_path=patient["coilmap_path"],
            segmask_path=patient["segmask_path"],
            mask_args=config,
            overwrite=cli.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": "converted",
                    "patient_index": patient_index,
                    "patient_count": len(patients),
                    **record,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    index = rebuild_windowed_index(cli.output_root, deep=cli.deep_verify)
    print(json.dumps({"status": "complete", "patients": len(index["patients"])}, sort_keys=True))


if __name__ == "__main__":
    main()
