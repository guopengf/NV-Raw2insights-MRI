#!/usr/bin/env python3
"""Validate real-patient windowed HDF5 stores against their raw MAT sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from mra_utils import read_mat_array, read_real_mat_array
from transforms import raw_4dflow_to_joint_hybrid
from utils import load_config
from windowed_4dflow import (
    Windowed4DFlowDataset,
    _mask_to_compact,
    build_windowed_4dflow_manifests,
    discover_4dflow_patients,
    validate_patient_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-patients", required=True, type=int)
    parser.add_argument("--expected-manifests", required=True, type=int)
    parser.add_argument("--expected-val-patients", type=int, default=0)
    parser.add_argument("--expected-val-manifests", type=int, default=0)
    parser.add_argument(
        "--raw-parity-patients",
        type=int,
        default=-1,
        help="Number of indexed patients to compare to raw MAT sources; -1 validates all.",
    )
    return parser.parse_args()


def _assert_hybrid_probes_equal(dataset: h5py.Dataset, expected: np.ndarray) -> None:
    if tuple(dataset.shape) != tuple(expected.shape):
        raise AssertionError(f"Shape mismatch: H5={dataset.shape}, raw={expected.shape}")
    probes = {
        (0, 0),
        (int(dataset.shape[0]) // 2, int(dataset.shape[1]) // 2),
        (int(dataset.shape[0]) - 1, int(dataset.shape[1]) - 1),
    }
    for frame_idx, slice_idx in sorted(probes):
        np.testing.assert_array_equal(
            dataset[frame_idx, slice_idx], expected[frame_idx, slice_idx]
        )


def _validate_raw_parity(patient: dict) -> None:
    record = validate_patient_store(patient["path"], deep=True)
    sources = record["source_manifest"]
    with h5py.File(patient["path"], "r", swmr=True) as store:
        target_raw = read_mat_array(
            sources["target"]["path"], ("kdata_full", "kdata", "kspace_full", "kspace")
        )
        target_hybrid = raw_4dflow_to_joint_hybrid(target_raw)
        del target_raw
        _assert_hybrid_probes_equal(store["hybrid/target"], target_hybrid)
        del target_hybrid

        for acceleration in patient["accelerations"]:
            acceleration_key = str(int(acceleration))
            input_raw = read_mat_array(
                sources["inputs"][acceleration_key]["path"],
                ("kdata", "kdata_ktGaussian", "kus", "kspace", "kspace_full"),
            )
            input_hybrid = raw_4dflow_to_joint_hybrid(input_raw)
            del input_raw
            _assert_hybrid_probes_equal(
                store[f"hybrid/input/{acceleration_key}"], input_hybrid
            )
            del input_hybrid

            mask_raw = read_real_mat_array(
                sources["masks"][acceleration_key]["path"],
                ("mask", "usmask", "usmask_ktGaussian", "sampling_mask"),
            )
            compact_mask = _mask_to_compact(mask_raw)
            del mask_raw
            np.testing.assert_array_equal(
                store[f"mask/{acceleration_key}"][:], compact_mask
            )


def _smoke_representative_profiles(index: dict, manifests: list[Path], config) -> list[dict]:
    manifest_by_pair = {}
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        manifest_by_pair[(payload["target_kspace"], int(payload["acceleration"]))] = manifest_path

    representatives = {}
    for patient in index["patients"]:
        profile = tuple(int(value) for value in patient["accelerations"])
        representatives.setdefault(profile, patient)

    results = []
    for profile, patient in sorted(representatives.items()):
        acceleration = int(profile[0])
        manifest_path = manifest_by_pair[(patient["target_kspace"], acceleration)]
        dataset = Windowed4DFlowDataset(
            [{"kspace": manifest_path}],
            config,
            center_selector=lambda total, count: np.arange(count, dtype=np.int64),
        )
        sample = dataset[0]
        checked = {}
        for key in ("kspace_masked_ifft", "kspace_ifft", "mask", "sensitivity_maps"):
            value = sample.get(key)
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"Non-finite values in {key} for {patient['patient_key']}")
            checked[key] = list(tensor.shape)
        dataset.close()
        results.append(
            {
                "patient_key": patient["patient_key"],
                "profile": list(profile),
                "acceleration": acceleration,
                "shapes": checked,
            }
        )
    return results


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    if config is None:
        raise RuntimeError(f"Could not load config: {cli.config}")
    index = json.loads(cli.index_path.read_text())
    patients = index["patients"]
    if len(patients) != cli.expected_patients:
        raise AssertionError(
            f"Expected {cli.expected_patients} patients, found {len(patients)}"
        )

    cli.work_dir.mkdir(parents=True, exist_ok=True)
    manifests = build_windowed_4dflow_manifests(
        index_path=cli.index_path,
        data_roots=config.data_path_train,
        out_dir=cli.work_dir / "manifests",
        accelerations=config.four_dflow_accelerations,
        encodings=config.four_dflow_encodings,
        joint_encodings=True,
        expected_storage_profile=config.four_dflow_storage.storage_profile,
    )
    if len(manifests) != cli.expected_manifests:
        raise AssertionError(
            f"Expected {cli.expected_manifests} manifests, found {len(manifests)}"
        )

    raw_parity_patients = (
        patients if cli.raw_parity_patients < 0 else patients[: cli.raw_parity_patients]
    )
    for patient in raw_parity_patients:
        _validate_raw_parity(patient)

    profile_smoke = _smoke_representative_profiles(index, manifests, config)
    val_patients = discover_4dflow_patients(
        config.data_path_val, config.four_dflow_accelerations
    )
    val_manifest_count = sum(len(patient["inputs"]) for patient in val_patients)
    if cli.expected_val_patients and len(val_patients) != cli.expected_val_patients:
        raise AssertionError(
            f"Expected {cli.expected_val_patients} validation patients, found {len(val_patients)}"
        )
    if cli.expected_val_manifests and val_manifest_count != cli.expected_val_manifests:
        raise AssertionError(
            f"Expected {cli.expected_val_manifests} validation manifests, found {val_manifest_count}"
        )

    payload = {
        "status": "ok",
        "patients": len(patients),
        "manifests": len(manifests),
        "profiles": profile_smoke,
        "raw_probe_parity_patients": len(raw_parity_patients),
        "val_patients": len(val_patients),
        "val_manifests": val_manifest_count,
        "h5_bytes": sum(int(patient["size_bytes"]) for patient in patients),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
