#!/usr/bin/env python3
"""Convert raw 4D-flow MAT files into restart-safe windowed HDF5 stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils import load_config
from windowed_4dflow import (
    ENCODING_CHUNK_1,
    ENCODING_CHUNK_ALL,
    build_patient_source_manifest,
    convert_patient_to_windowed_hdf5_profiles,
    discover_4dflow_patients,
    rebuild_windowed_index,
    validate_patient_store,
    validate_windowed_profile_pair,
)


CONVERSION_PLAN_SCHEMA = "raw2insights.4dflow.windowed_hdf5_conversion_plan"
CONVERSION_PLAN_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--e1-output-root", type=Path)
    parser.add_argument(
        "--e4-output-root",
        type=Path,
        help="Optional E4 output. Omit it for the recommended E1-only conversion.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-plan", action="store_true")
    mode.add_argument("--run-shard", action="store_true")
    mode.add_argument("--finalize-plan", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--expected-patients", type=int, default=0)
    parser.add_argument("--forbid-path", type=Path, action="append", default=[])
    parser.add_argument("--limit-patients", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--deep-verify", action="store_true")
    return parser.parse_args()


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


def _atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_text_dump(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _plan_hash(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_bytes(value: Any) -> int:
    if isinstance(value, dict):
        if "size_bytes" in value:
            return int(value["size_bytes"])
        return sum(_source_bytes(item) for item in value.values())
    if isinstance(value, list):
        return sum(_source_bytes(item) for item in value)
    return 0


def _check_forbidden_paths(paths: Iterable[str | Path]) -> None:
    present = [str(Path(path)) for path in paths if Path(path).exists()]
    if present:
        raise RuntimeError(f"Forbidden output paths exist: {present}")


def _check_expected_patient_count(patient_count: int, expected_patients: int) -> None:
    if expected_patients > 0 and patient_count != expected_patients:
        raise RuntimeError(
            f"Expected {expected_patients} patients, discovered {patient_count}"
        )


def _plan_patient_record(patient: dict[str, Any], output_root: Path) -> dict[str, Any]:
    inputs = {int(key): Path(value) for key, value in patient["inputs"].items()}
    masks = {int(key): Path(value) for key, value in patient["masks"].items()}
    source_manifest = build_patient_source_manifest(
        target_path=patient["target_path"],
        acceleration_inputs=inputs,
        acceleration_masks=masks,
        coilmap_path=patient["coilmap_path"],
        segmask_path=patient["segmask_path"],
    )
    output_path = output_root / "patients" / f"{patient['patient_key']}.h5"
    conversion_required = True
    if output_path.exists():
        record = validate_patient_store(
            output_path,
            expected_storage_profile=ENCODING_CHUNK_1,
        )
        if record["patient_key"] != patient["patient_key"]:
            raise RuntimeError(f"Patient key mismatch in existing store: {output_path}")
        if record["source_manifest"] != source_manifest:
            raise RuntimeError(f"Source provenance changed for existing store: {output_path}")
        conversion_required = False
    return {
        "patient_key": patient["patient_key"],
        "target_path": str(patient["target_path"]),
        "inputs": {str(key): str(value) for key, value in sorted(inputs.items())},
        "masks": {str(key): str(value) for key, value in sorted(masks.items())},
        "coilmap_path": str(patient["coilmap_path"]) if patient["coilmap_path"] else None,
        "segmask_path": str(patient["segmask_path"]) if patient["segmask_path"] else None,
        "output_path": str(output_path),
        "source_manifest": source_manifest,
        "source_bytes": _source_bytes(source_manifest),
        "conversion_required_at_plan_time": conversion_required,
    }


def build_conversion_plan_payload(
    *,
    config_path: str | Path,
    output_root: str | Path,
    patients: list[dict[str, Any]],
    num_shards: int,
    expected_patients: int = 0,
) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    output_root = Path(output_root)
    if (output_root / "COMPLETED").exists():
        raise RuntimeError(f"Conversion is already marked complete: {output_root}")
    _check_expected_patient_count(len(patients), expected_patients)

    entries = [_plan_patient_record(patient, output_root) for patient in patients]
    entries.sort(key=lambda item: item["patient_key"])
    patient_keys = [entry["patient_key"] for entry in entries]
    if len(patient_keys) != len(set(patient_keys)):
        raise RuntimeError("Duplicate patient keys in conversion plan")

    shards = [
        {
            "shard_index": shard_index,
            "patient_keys": [],
            "estimated_conversion_bytes": 0,
        }
        for shard_index in range(num_shards)
    ]
    pending = [entry for entry in entries if entry["conversion_required_at_plan_time"]]
    for entry in sorted(pending, key=lambda item: (-item["source_bytes"], item["patient_key"])):
        shard = min(
            shards,
            key=lambda item: (
                item["estimated_conversion_bytes"],
                len(item["patient_keys"]),
                item["shard_index"],
            ),
        )
        shard["patient_keys"].append(entry["patient_key"])
        shard["estimated_conversion_bytes"] += entry["source_bytes"]

    reused = [entry for entry in entries if not entry["conversion_required_at_plan_time"]]
    for entry in reused:
        shard = min(
            shards,
            key=lambda item: (
                len(item["patient_keys"]),
                item["estimated_conversion_bytes"],
                item["shard_index"],
            ),
        )
        shard["patient_keys"].append(entry["patient_key"])
    for shard in shards:
        shard["patient_keys"].sort()
        shard["patient_count"] = len(shard["patient_keys"])

    payload = {
        "schema": CONVERSION_PLAN_SCHEMA,
        "schema_version": CONVERSION_PLAN_VERSION,
        "config_path": str(Path(config_path).resolve()),
        "output_root": str(output_root),
        "storage_profile": ENCODING_CHUNK_1,
        "patient_count": len(entries),
        "conversion_required_count": len(pending),
        "reused_count": len(reused),
        "shard_count": num_shards,
        "patients": entries,
        "shards": shards,
    }
    payload["plan_sha256"] = _plan_hash(payload)
    return payload


def write_conversion_plan(payload: dict[str, Any], path: str | Path) -> None:
    if payload.get("plan_sha256") != _plan_hash(payload):
        raise RuntimeError("Refusing to write a conversion plan with an invalid hash")
    _atomic_json_dump(payload, path)


def load_conversion_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != CONVERSION_PLAN_SCHEMA:
        raise RuntimeError(f"Unexpected conversion plan schema in {path}")
    if int(payload.get("schema_version", -1)) != CONVERSION_PLAN_VERSION:
        raise RuntimeError(f"Unsupported conversion plan version in {path}")
    if payload.get("storage_profile") != ENCODING_CHUNK_1:
        raise RuntimeError("Production conversion plan must use encoding_chunk_1")
    expected_hash = _plan_hash(payload)
    if payload.get("plan_sha256") != expected_hash:
        raise RuntimeError(f"Conversion plan hash mismatch in {path}")
    if int(payload["patient_count"]) != len(payload["patients"]):
        raise RuntimeError("Conversion plan patient count mismatch")
    if int(payload["shard_count"]) != len(payload["shards"]):
        raise RuntimeError("Conversion plan shard count mismatch")
    planned_keys = [entry["patient_key"] for entry in payload["patients"]]
    assigned_keys = [key for shard in payload["shards"] for key in shard["patient_keys"]]
    if sorted(planned_keys) != sorted(assigned_keys) or len(assigned_keys) != len(set(assigned_keys)):
        raise RuntimeError("Conversion plan does not assign each patient exactly once")
    return payload


def run_conversion_plan_shard(plan_path: str | Path, shard_index: int) -> dict[str, Any]:
    plan_path = Path(plan_path)
    plan = load_conversion_plan(plan_path)
    if not 0 <= shard_index < int(plan["shard_count"]):
        raise ValueError(f"Invalid shard index {shard_index}")
    config = load_config(Path(plan["config_path"]))
    if config is None:
        raise RuntimeError(f"Could not load config: {plan['config_path']}")
    entries = {entry["patient_key"]: entry for entry in plan["patients"]}
    shard = plan["shards"][shard_index]
    converted_count = 0
    reused_count = 0
    for patient_index, patient_key in enumerate(shard["patient_keys"], start=1):
        entry = entries[patient_key]
        output_path = Path(entry["output_path"])
        existed_before = output_path.exists()
        records = convert_patient_to_windowed_hdf5_profiles(
            patient_key=patient_key,
            target_path=entry["target_path"],
            acceleration_inputs={int(key): value for key, value in entry["inputs"].items()},
            acceleration_masks={int(key): value for key, value in entry["masks"].items()},
            output_paths={ENCODING_CHUNK_1: output_path},
            coilmap_path=entry["coilmap_path"],
            segmask_path=entry["segmask_path"],
            mask_args=config,
        )
        record = records[ENCODING_CHUNK_1]
        if record["source_manifest"] != entry["source_manifest"]:
            raise RuntimeError(f"Post-conversion provenance mismatch for {patient_key}")
        status = "reused" if existed_before else "converted"
        converted_count += int(status == "converted")
        reused_count += int(status == "reused")
        print(
            json.dumps(
                {
                    "status": status,
                    "shard_index": shard_index,
                    "shard_patient_index": patient_index,
                    "shard_patient_count": len(shard["patient_keys"]),
                    "patient_key": patient_key,
                    "profile": record,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    marker = {
        "status": "shard_complete",
        "plan_sha256": plan["plan_sha256"],
        "shard_index": shard_index,
        "patient_keys": shard["patient_keys"],
        "converted_count": converted_count,
        "reused_count": reused_count,
    }
    marker_path = plan_path.parent / "shards" / f"shard-{shard_index:02d}.json"
    _atomic_json_dump(marker, marker_path)
    print(json.dumps(marker, sort_keys=True), flush=True)
    return marker


def finalize_conversion_plan(
    plan_path: str | Path,
    *,
    deep_verify: bool,
    expected_patients: int = 0,
    forbid_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    plan = load_conversion_plan(plan_path)
    _check_expected_patient_count(int(plan["patient_count"]), expected_patients)
    _check_forbidden_paths(forbid_paths)
    output_root = Path(plan["output_root"])

    for shard in plan["shards"]:
        marker_path = plan_path.parent / "shards" / f"shard-{shard['shard_index']:02d}.json"
        if not marker_path.exists():
            raise RuntimeError(f"Missing shard completion marker: {marker_path}")
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "shard_complete":
            raise RuntimeError(f"Invalid shard marker status: {marker_path}")
        if marker.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError(f"Shard marker plan hash mismatch: {marker_path}")
        if marker.get("patient_keys") != shard["patient_keys"]:
            raise RuntimeError(f"Shard marker patient set mismatch: {marker_path}")

    temporary_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and (".tmp" in path.name or ".partial" in path.name)
    )
    if temporary_files:
        raise RuntimeError(f"Temporary or partial files remain: {temporary_files}")

    expected_paths = {Path(entry["output_path"]) for entry in plan["patients"]}
    actual_paths = set((output_root / "patients").glob("**/*.h5"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(f"Patient store set mismatch: missing={missing}, extra={extra}")

    entries = {entry["patient_key"]: entry for entry in plan["patients"]}
    records = []
    for path in sorted(expected_paths):
        record = validate_patient_store(
            path,
            deep=deep_verify,
            expected_storage_profile=ENCODING_CHUNK_1,
        )
        entry = entries.get(record["patient_key"])
        if entry is None or Path(entry["output_path"]) != path:
            raise RuntimeError(f"Unexpected patient identity in {path}")
        if record["source_manifest"] != entry["source_manifest"]:
            raise RuntimeError(f"Final provenance mismatch for {record['patient_key']}")
        records.append(record)

    index = rebuild_windowed_index(output_root, deep=deep_verify)
    indexed_keys = [record["patient_key"] for record in index["patients"]]
    if sorted(indexed_keys) != sorted(entries):
        raise RuntimeError("Final index patient set does not match conversion plan")
    if index["storage_profile"] != ENCODING_CHUNK_1:
        raise RuntimeError("Final index does not use encoding_chunk_1")

    completed_payload = {
        "status": "complete",
        "patients": len(records),
        "e1_profile": index["storage_profile"],
        "plan_sha256": plan["plan_sha256"],
    }
    _atomic_text_dump(
        json.dumps(completed_payload, sort_keys=True) + "\n",
        output_root / "COMPLETED",
    )
    return completed_payload


def _discover_from_config(
    config_path: Path, limit_patients: int
) -> tuple[Any, list[dict[str, Any]]]:
    config = load_config(config_path)
    if config is None:
        raise RuntimeError(f"Could not load config: {config_path}")
    roots = list(dict.fromkeys([*config.data_path_train, *config.data_path_val]))
    accelerations = [int(value) for value in config.four_dflow_accelerations]
    patients = discover_4dflow_patients(roots, accelerations)
    if limit_patients > 0:
        patients = patients[:limit_patients]
    if not patients:
        raise RuntimeError("No fully populated 4D-flow patients were discovered")
    return config, patients


def _run_verify_only(cli: argparse.Namespace) -> None:
    output_root = _require(cli.e1_output_root, "--e1-output-root is required")
    e1_index = rebuild_windowed_index(output_root, deep=cli.deep_verify)
    if e1_index["storage_profile"] != ENCODING_CHUNK_1:
        raise RuntimeError(
            f"Expected E1 profile at {output_root}, found {e1_index['storage_profile']!r}"
        )
    _check_expected_patient_count(len(e1_index["patients"]), cli.expected_patients)
    payload = {
        "status": "verified",
        "patients": len(e1_index["patients"]),
        "e1_index": str(output_root / "index.json"),
        "e1_profile": e1_index["storage_profile"],
    }
    if cli.e4_output_root is not None:
        payload["pair"] = validate_windowed_profile_pair(
            output_root,
            cli.e4_output_root,
            deep=cli.deep_verify,
        )
    print(json.dumps(payload, sort_keys=True))


def _run_sequential(cli: argparse.Namespace) -> None:
    config_path = _require(cli.config, "--config is required")
    output_root = _require(cli.e1_output_root, "--e1-output-root is required")
    config, patients = _discover_from_config(config_path, cli.limit_patients)
    _check_expected_patient_count(len(patients), cli.expected_patients)
    _check_forbidden_paths(cli.forbid_path)

    for patient_index, patient in enumerate(patients, start=1):
        output_paths = {
            ENCODING_CHUNK_1: output_root / "patients" / f"{patient['patient_key']}.h5",
        }
        if cli.e4_output_root is not None:
            output_paths[ENCODING_CHUNK_ALL] = (
                cli.e4_output_root / "patients" / f"{patient['patient_key']}.h5"
            )
        records = convert_patient_to_windowed_hdf5_profiles(
            patient_key=patient["patient_key"],
            target_path=patient["target_path"],
            acceleration_inputs=patient["inputs"],
            acceleration_masks=patient["masks"],
            output_paths=output_paths,
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
    e1_index = rebuild_windowed_index(output_root, deep=cli.deep_verify)
    payload = {
        "status": "complete",
        "patients": len(e1_index["patients"]),
        "e1_profile": e1_index["storage_profile"],
    }
    if cli.e4_output_root is not None:
        e4_index = rebuild_windowed_index(cli.e4_output_root, deep=cli.deep_verify)
        payload["e4_profile"] = e4_index["storage_profile"]
        payload["pair"] = validate_windowed_profile_pair(
            output_root,
            cli.e4_output_root,
            deep=cli.deep_verify,
        )
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    cli = parse_args()
    if cli.prepare_plan:
        config_path = _require(cli.config, "--config is required with --prepare-plan")
        output_root = _require(
            cli.e1_output_root,
            "--e1-output-root is required with --prepare-plan",
        )
        plan_out = _require(cli.plan_out, "--plan-out is required with --prepare-plan")
        _check_forbidden_paths(cli.forbid_path)
        _, patients = _discover_from_config(config_path, cli.limit_patients)
        plan = build_conversion_plan_payload(
            config_path=config_path,
            output_root=output_root,
            patients=patients,
            num_shards=cli.num_shards,
            expected_patients=cli.expected_patients,
        )
        write_conversion_plan(plan, plan_out)
        print(
            json.dumps(
                {
                    "status": "planned",
                    "plan": str(plan_out),
                    "plan_sha256": plan["plan_sha256"],
                    "patients": plan["patient_count"],
                    "conversion_required": plan["conversion_required_count"],
                    "reused": plan["reused_count"],
                    "shards": plan["shard_count"],
                },
                sort_keys=True,
            )
        )
        return
    if cli.run_shard:
        plan_path = _require(cli.plan, "--plan is required with --run-shard")
        shard_index = _require(
            cli.shard_index,
            "--shard-index is required with --run-shard",
        )
        run_conversion_plan_shard(plan_path, shard_index)
        return
    if cli.finalize_plan:
        plan_path = _require(cli.plan, "--plan is required with --finalize-plan")
        payload = finalize_conversion_plan(
            plan_path,
            deep_verify=cli.deep_verify,
            expected_patients=cli.expected_patients,
            forbid_paths=cli.forbid_path,
        )
        print(json.dumps(payload, sort_keys=True))
        print(
            json.dumps(
                {
                    "status": "verified",
                    "patients": payload["patients"],
                    "e1_profile": payload["e1_profile"],
                    "plan_sha256": payload["plan_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if cli.verify_only:
        _run_verify_only(cli)
        return
    _run_sequential(cli)


if __name__ == "__main__":
    main()
