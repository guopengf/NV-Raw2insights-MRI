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
    ENCODING_CHUNK_1,
    ENCODING_CHUNK_ALL,
    convert_patient_to_windowed_hdf5_profiles,
    discover_4dflow_patients,
    rebuild_windowed_index,
    validate_windowed_profile_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--e1-output-root", required=True, type=Path)
    parser.add_argument("--e4-output-root", required=True, type=Path)
    parser.add_argument("--limit-patients", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--deep-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.verify_only:
        result = validate_windowed_profile_pair(
            cli.e1_output_root,
            cli.e4_output_root,
            deep=cli.deep_verify,
        )
        print(json.dumps({"status": "verified", **result}, sort_keys=True))
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
        records = convert_patient_to_windowed_hdf5_profiles(
            patient_key=patient["patient_key"],
            target_path=patient["target_path"],
            acceleration_inputs=patient["inputs"],
            acceleration_masks=patient["masks"],
            output_paths={
                ENCODING_CHUNK_1: cli.e1_output_root / "patients" / f"{patient['patient_key']}.h5",
                ENCODING_CHUNK_ALL: cli.e4_output_root / "patients" / f"{patient['patient_key']}.h5",
            },
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
                    "patient_key": patient["patient_key"],
                    "profiles": records,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    e1_index = rebuild_windowed_index(cli.e1_output_root, deep=cli.deep_verify)
    e4_index = rebuild_windowed_index(cli.e4_output_root, deep=cli.deep_verify)
    result = validate_windowed_profile_pair(
        cli.e1_output_root,
        cli.e4_output_root,
        deep=cli.deep_verify,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "patients": len(e1_index["patients"]),
                "e1_profile": e1_index["storage_profile"],
                "e4_profile": e4_index["storage_profile"],
                "pair": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
