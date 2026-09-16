#!/usr/bin/env python3
"""Build a deterministic patient-level train/val/test view of 4D Flow data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from path_safety import assert_outputs_not_in_data


SCHEMA = "raw2insights.4dflow.unified_split"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Case:
    uid: str
    source_task: str
    source_split: str
    source_root: str
    data_root: str
    organ: str
    center: str
    scanner: str
    patient: str
    raw_origin_path: str
    source_path: str
    available_accelerations: tuple[int, ...]
    acceleration_profile: str
    files_present: tuple[str, ...]

    @property
    def source_dataset(self) -> str:
        return f"{self.source_task}/{self.source_split}"

    @property
    def organ_center(self) -> str:
        return f"{self.organ}/{self.center}"

    def inventory_record(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_task": self.source_task,
            "source_split": self.source_split,
            "source_dataset": self.source_dataset,
            "source_root": self.source_root,
            "data_root": self.data_root,
            "organ": self.organ,
            "center": self.center,
            "organ_center": self.organ_center,
            "scanner": self.scanner,
            "patient": self.patient,
            "raw_origin_path": self.raw_origin_path,
            "source_path": self.source_path,
            "available_accelerations": list(self.available_accelerations),
            "acceleration_profile": self.acceleration_profile,
            "files_present": list(self.files_present),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory multiple CMRx4DFlow task roots, create a deterministic "
            "patient-level split plan, and optionally materialize symlink views."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output_root from the config without modifying the config file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create patient-directory symlinks after writing the split plan.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _atomic_text_dump(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_json_dump(payload: Any, path: Path) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_text_dump(text, path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not component or component in {".", ".."}:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return component


def _case_uid(
    task: str,
    source_split: str,
    organ: str,
    center: str,
    scanner: str,
    patient: str,
) -> str:
    parts = (task, source_split, organ, center, scanner, patient)
    return "__".join(_safe_component(part) for part in parts)


def _acceleration_profile(available: tuple[int, ...], configured: tuple[int, ...]) -> str:
    if available == configured:
        return "all_" + "_".join(str(value) for value in configured)
    return "partial_" + "_".join(str(value) for value in available)


def _invalid_case(message: str, policy: str) -> bool:
    if policy == "skip":
        print(f"WARNING: skipping invalid case: {message}", file=sys.stderr)
        return True
    if policy != "error":
        raise ValueError(f"Unknown invalid_case_policy: {policy!r}")
    raise ValueError(message)


def discover_cases(config: dict[str, Any]) -> list[Case]:
    validation = config.get("validation", {})
    required_files = tuple(validation.get("required_files", []))
    optional_files = tuple(validation.get("optional_files", []))
    invalid_policy = str(validation.get("invalid_case_policy", "error")).lower()

    acceleration_cfg = config.get("accelerations", {})
    acceleration_values = tuple(int(value) for value in acceleration_cfg["values"])
    if not acceleration_values or len(acceleration_values) != len(set(acceleration_values)):
        raise ValueError("accelerations.values must contain unique acceleration factors")
    input_pattern = str(acceleration_cfg["input_pattern"])
    mask_pattern = str(acceleration_cfg["mask_pattern"])
    minimum_available = int(acceleration_cfg.get("minimum_available", 1))

    cases: list[Case] = []
    seen_uids: set[str] = set()
    seen_sources: set[str] = set()

    for source in config["sources"]:
        root = Path(source["root"]).expanduser().resolve()
        data_root = Path(source.get("prepared_root", root)).expanduser().resolve()
        task = str(source["task"])
        source_split = str(source["source_split"])
        if not root.is_dir():
            raise FileNotFoundError(f"Source root does not exist: {root}")
        if not data_root.is_dir():
            raise FileNotFoundError(f"Prepared source root does not exist: {data_root}")

        source_count = 0
        for organ_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for center_dir in sorted(organ_dir.glob("Center*")):
                if not center_dir.is_dir():
                    continue
                for scanner_dir in sorted(path for path in center_dir.iterdir() if path.is_dir()):
                    for patient_dir in sorted(path for path in scanner_dir.iterdir() if path.is_dir()):
                        relative_patient = patient_dir.relative_to(root)
                        data_patient_dir = data_root / relative_patient
                        if not data_patient_dir.is_dir():
                            if _invalid_case(
                                f"Prepared patient directory is missing: {data_patient_dir}",
                                invalid_policy,
                            ):
                                continue
                        missing = [
                            name
                            for name in required_files
                            if not (data_patient_dir / name).is_file()
                        ]
                        if missing:
                            if _invalid_case(
                                f"{data_patient_dir} is missing required files: {missing}",
                                invalid_policy,
                            ):
                                continue

                        available: list[int] = []
                        for acceleration in acceleration_values:
                            input_path = data_patient_dir / input_pattern.format(
                                acceleration=acceleration
                            )
                            mask_path = data_patient_dir / mask_pattern.format(
                                acceleration=acceleration
                            )
                            if input_path.is_file() != mask_path.is_file():
                                if _invalid_case(
                                    f"{data_patient_dir} has an incomplete acceleration "
                                    f"{acceleration} pair",
                                    invalid_policy,
                                ):
                                    available = []
                                    break
                            if input_path.is_file():
                                available.append(acceleration)
                        if len(available) < minimum_available:
                            if _invalid_case(
                                f"{data_patient_dir} has {len(available)} acceleration pairs; "
                                f"minimum_available={minimum_available}",
                                invalid_policy,
                            ):
                                continue

                        available_tuple = tuple(available)
                        uid = _case_uid(
                            task,
                            source_split,
                            organ_dir.name,
                            center_dir.name,
                            scanner_dir.name,
                            patient_dir.name,
                        )
                        source_path = str(data_patient_dir.resolve())
                        raw_origin_path = str(patient_dir.resolve())
                        if uid in seen_uids:
                            raise ValueError(f"Duplicate case UID: {uid}")
                        if source_path in seen_sources:
                            raise ValueError(f"The same source patient was discovered twice: {source_path}")

                        files_present = tuple(
                            name
                            for name in (*required_files, *optional_files)
                            if (data_patient_dir / name).is_file()
                        )
                        cases.append(
                            Case(
                                uid=uid,
                                source_task=task,
                                source_split=source_split,
                                source_root=str(root),
                                data_root=str(data_root),
                                organ=organ_dir.name,
                                center=center_dir.name,
                                scanner=scanner_dir.name,
                                patient=patient_dir.name,
                                raw_origin_path=raw_origin_path,
                                source_path=source_path,
                                available_accelerations=available_tuple,
                                acceleration_profile=_acceleration_profile(
                                    available_tuple, acceleration_values
                                ),
                                files_present=files_present,
                            )
                        )
                        seen_uids.add(uid)
                        seen_sources.add(source_path)
                        source_count += 1

        expected_source_count = source.get("expected_patients")
        if expected_source_count is not None and source_count != int(expected_source_count):
            raise ValueError(
                f"{task}/{source_split}: expected {expected_source_count} patients, "
                f"discovered {source_count}"
            )

    cases.sort(key=lambda case: case.uid)
    expected_total = validation.get("expected_total_patients")
    if expected_total is not None and len(cases) != int(expected_total):
        raise ValueError(f"Expected {expected_total} total patients, discovered {len(cases)}")
    if not cases:
        raise ValueError("No valid patient cases were discovered")
    return cases


def _validate_ratios(raw_ratios: dict[str, Any]) -> dict[str, float]:
    if set(raw_ratios) != set(SPLITS):
        raise ValueError(f"split_ratios must define exactly {SPLITS}")
    ratios = {split: float(raw_ratios[split]) for split in SPLITS}
    if any(value <= 0.0 for value in ratios.values()):
        raise ValueError("Every split ratio must be greater than zero")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split_ratios must sum to 1.0, got {sum(ratios.values())}")
    return ratios


def _largest_remainder(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    result = {split: math.floor(raw[split]) for split in SPLITS}
    missing = total - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - result[split]), SPLITS.index(split)))
    for split in order[:missing]:
        result[split] += 1
    return result


def _dimension_value(case: Case, dimension: str) -> str:
    values = {
        "organ": case.organ,
        "center": case.center,
        "organ_center": case.organ_center,
        "scanner": case.scanner,
        "source_task": case.source_task,
        "source_split": case.source_split,
        "source_dataset": case.source_dataset,
        "acceleration_profile": case.acceleration_profile,
    }
    try:
        return values[dimension]
    except KeyError as error:
        raise ValueError(f"Unsupported stratification dimension: {dimension!r}") from error


def _allocate_primary_quotas(
    cases: list[Case],
    primary_dimension: str,
    ratios: dict[str, float],
    capacities: dict[str, int],
    seed: int,
) -> dict[str, dict[str, int]]:
    group_counts = Counter(_dimension_value(case, primary_dimension) for case in cases)
    raw = {
        group: {split: count * ratios[split] for split in SPLITS}
        for group, count in group_counts.items()
    }
    quotas = {
        group: {split: math.floor(raw[group][split]) for split in SPLITS}
        for group in group_counts
    }
    row_remaining = {
        group: group_counts[group] - sum(quotas[group].values()) for group in group_counts
    }
    column_remaining = {
        split: capacities[split] - sum(quotas[group][split] for group in group_counts)
        for split in SPLITS
    }

    while sum(row_remaining.values()):
        candidates: list[tuple[float, str, str, str]] = []
        for group in sorted(group_counts):
            if row_remaining[group] <= 0:
                continue
            for split in SPLITS:
                if column_remaining[split] <= 0:
                    continue
                current = quotas[group][split]
                target = raw[group][split]
                delta = (current + 1 - target) ** 2 - (current - target) ** 2
                tie = hashlib.sha256(f"{seed}|{group}|{split}".encode()).hexdigest()
                candidates.append((delta, tie, group, split))
        if not candidates:
            raise RuntimeError("Unable to satisfy primary stratification quotas")
        _, _, group, split = min(candidates)
        quotas[group][split] += 1
        row_remaining[group] -= 1
        column_remaining[split] -= 1

    if any(column_remaining.values()):
        raise RuntimeError(f"Primary quotas do not fill split capacities: {column_remaining}")
    return quotas


def _secondary_counts(
    cases: Iterable[Case], dimensions: dict[str, float]
) -> dict[tuple[str, str], int]:
    totals: Counter[tuple[str, str]] = Counter()
    for case in cases:
        for dimension in dimensions:
            totals[(dimension, _dimension_value(case, dimension))] += 1
    return dict(totals)


def _assignment_objective(
    counts: dict[tuple[str, str, str], int],
    totals: dict[tuple[str, str], int],
    dimensions: dict[str, float],
    ratios: dict[str, float],
    coverage_min_cases: int,
    coverage_penalty: float,
) -> float:
    score = 0.0
    for (dimension, value), total in totals.items():
        weight = dimensions[dimension]
        for split in SPLITS:
            count = counts.get((dimension, value, split), 0)
            target = total * ratios[split]
            score += weight * (count - target) ** 2 / max(total, 1)
            if total >= coverage_min_cases and count == 0:
                score += weight * coverage_penalty
    return score


def assign_splits(
    cases: list[Case], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, int], dict[str, dict[str, int]], float]:
    ratios = _validate_ratios(config["split_ratios"])
    seed = int(config.get("seed", 0))
    stratification = config.get("stratification", {})
    primary_dimension = str(stratification.get("primary_dimension", "organ"))
    dimensions = {
        str(name): float(weight)
        for name, weight in stratification.get("secondary_dimensions", {}).items()
        if float(weight) > 0.0
    }
    if primary_dimension in dimensions:
        dimensions.pop(primary_dimension)
    search_trials = int(stratification.get("search_trials", 256))
    coverage_min_cases = int(stratification.get("coverage_min_cases", 5))
    coverage_penalty = float(stratification.get("coverage_penalty", 50.0))
    if search_trials < 1:
        raise ValueError("stratification.search_trials must be at least 1")

    capacities = _largest_remainder(len(cases), ratios)
    primary_quotas = _allocate_primary_quotas(
        cases, primary_dimension, ratios, capacities, seed
    )
    totals = _secondary_counts(cases, dimensions)
    rarity = {
        case.uid: sum(
            weight / totals[(dimension, _dimension_value(case, dimension))]
            for dimension, weight in dimensions.items()
        )
        for case in cases
    }

    best_assignment: dict[str, str] | None = None
    best_score = math.inf
    best_signature = ""

    for trial in range(search_trials):
        rng = random.Random(seed + trial * 1_000_003)
        order = sorted(cases, key=lambda case: (-rarity[case.uid], rng.random(), case.uid))
        assigned_primary: Counter[tuple[str, str]] = Counter()
        counts: Counter[tuple[str, str, str]] = Counter()
        assignment: dict[str, str] = {}

        for case in order:
            primary_value = _dimension_value(case, primary_dimension)
            choices = [
                split
                for split in SPLITS
                if assigned_primary[(primary_value, split)] < primary_quotas[primary_value][split]
            ]
            if not choices:
                raise RuntimeError(f"No remaining quota for primary group {primary_value}")

            scored_choices: list[tuple[float, float, int, str]] = []
            for split in choices:
                delta = 0.0
                for dimension, weight in dimensions.items():
                    value = _dimension_value(case, dimension)
                    total = totals[(dimension, value)]
                    current = counts[(dimension, value, split)]
                    target = total * ratios[split]
                    delta += weight * (
                        (current + 1 - target) ** 2 - (current - target) ** 2
                    ) / max(total, 1)
                    if total >= coverage_min_cases and current == 0:
                        delta -= weight * coverage_penalty
                scored_choices.append((delta, rng.random(), SPLITS.index(split), split))

            split = min(scored_choices)[-1]
            assignment[case.uid] = split
            assigned_primary[(primary_value, split)] += 1
            for dimension in dimensions:
                value = _dimension_value(case, dimension)
                counts[(dimension, value, split)] += 1

        score = _assignment_objective(
            dict(counts),
            totals,
            dimensions,
            ratios,
            coverage_min_cases,
            coverage_penalty,
        )
        signature = "|".join(assignment[case.uid] for case in cases)
        if score < best_score or (math.isclose(score, best_score) and signature < best_signature):
            best_assignment = assignment
            best_score = score
            best_signature = signature

    if best_assignment is None:
        raise RuntimeError("Split assignment search produced no result")
    actual_capacities = Counter(best_assignment.values())
    if any(actual_capacities[split] != capacities[split] for split in SPLITS):
        raise RuntimeError(f"Split capacity mismatch: {dict(actual_capacities)} != {capacities}")
    return best_assignment, capacities, primary_quotas, best_score


def _destination_for(case: Case, split: str, output_root: Path) -> Path:
    return output_root / split / case.organ / case.center / case.scanner / case.uid


def _distribution(
    cases: list[Case], assignment: dict[str, str], dimension: str
) -> dict[str, dict[str, int]]:
    values = sorted({_dimension_value(case, dimension) for case in cases})
    counters = {value: Counter() for value in values}
    for case in cases:
        counters[_dimension_value(case, dimension)][assignment[case.uid]] += 1
    return {
        value: {split: counters[value][split] for split in SPLITS}
        for value in values
    }


def build_outputs(
    cases: list[Case],
    assignment: dict[str, str],
    capacities: dict[str, int],
    primary_quotas: dict[str, dict[str, int]],
    objective: float,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = []
    for case in cases:
        record = case.inventory_record()
        split = assignment[case.uid]
        record["split"] = split
        record["destination"] = str(_destination_for(case, split, output_root))
        record["materialized_files"] = sorted(
            path.name
            for path in Path(case.source_path).iterdir()
            if not path.name.startswith(".") and path.is_file()
        )
        records.append(record)

    dimensions = (
        "organ",
        "center",
        "organ_center",
        "scanner",
        "source_dataset",
        "acceleration_profile",
    )
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "total_patients": len(cases),
        "split_counts": capacities,
        "split_ratios": _validate_ratios(config["split_ratios"]),
        "objective": objective,
        "primary_quotas": primary_quotas,
        "distributions": {
            dimension: _distribution(cases, assignment, dimension)
            for dimension in dimensions
        },
    }
    plan = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": _stable_hash(config),
        "output_root": str(output_root),
        "seed": int(config.get("seed", 0)),
        "summary": summary,
        "cases": records,
    }
    plan["plan_sha256"] = _stable_hash(plan)

    organs_by_split = {
        split: sorted({case.organ for case in cases if assignment[case.uid] == split})
        for split in SPLITS
    }
    training_paths = {
        "data_path_train": [str(output_root / "train" / organ) for organ in organs_by_split["train"]],
        "data_path_val": [str(output_root / "val" / organ) for organ in organs_by_split["val"]],
        "data_path_test": [str(output_root / "test" / organ) for organ in organs_by_split["test"]],
        "note": (
            "Use these organ roots with the raw MAT backend. Cases originating from "
            "ValidationSet may contain only one acceleration pair."
        ),
    }
    return plan, summary, training_paths


def write_outputs(
    cases: list[Case],
    plan: dict[str, Any],
    summary: dict[str, Any],
    training_paths: dict[str, Any],
    output_root: Path,
) -> None:
    inventory_text = "".join(
        json.dumps(case.inventory_record(), sort_keys=True) + "\n" for case in cases
    )
    _atomic_text_dump(inventory_text, output_root / "inventory.jsonl")
    _atomic_json_dump(plan, output_root / "split_manifest.json")
    _atomic_json_dump(summary, output_root / "split_summary.json")
    _atomic_json_dump(training_paths, output_root / "training_paths.json")

    csv_path = output_root / "split_manifest.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fieldnames = [
        "uid",
        "split",
        "organ",
        "center",
        "scanner",
        "patient",
        "source_task",
        "source_split",
        "acceleration_profile",
        "available_accelerations",
        "raw_origin_path",
        "source_path",
        "destination",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in plan["cases"]:
            row = {key: record[key] for key in fieldnames}
            row["available_accelerations"] = ",".join(
                str(value) for value in record["available_accelerations"]
            )
            writer.writerow(row)
    os.replace(temporary, csv_path)


def materialize_file_symlinks(plan: dict[str, Any]) -> tuple[int, int, int]:
    directories_created = 0
    links_created = 0
    links_existing = 0
    for record in plan["cases"]:
        source = Path(record["source_path"])
        destination = Path(record["destination"])
        if not source.is_dir():
            raise FileNotFoundError(f"Source patient disappeared before apply: {source}")
        if destination.is_symlink():
            raise FileExistsError(
                f"Expected a real output patient directory, found a symlink: {destination}"
            )
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(f"Refusing to replace existing path: {destination}")
        if not destination.exists():
            destination.mkdir(parents=True)
            directories_created += 1

        for filename in record["materialized_files"]:
            source_file = source / filename
            destination_file = destination / filename
            if not source_file.is_file():
                raise FileNotFoundError(f"Source file disappeared before apply: {source_file}")
            if destination_file.is_symlink():
                if destination_file.resolve() != source_file.resolve():
                    raise FileExistsError(
                        f"Existing link points to a different source: {destination_file}"
                    )
                links_existing += 1
                continue
            if destination_file.exists():
                raise FileExistsError(
                    f"Refusing to replace existing output file: {destination_file}"
                )
            relative_target = os.path.relpath(source_file, start=destination_file.parent)
            destination_file.symlink_to(relative_target)
            links_created += 1
    return directories_created, links_created, links_existing


def main() -> None:
    cli = parse_args()
    config = _load_json(cli.config)
    source_roots = [
        Path(path)
        for source in config["sources"]
        for path in (source["root"], source.get("prepared_root", source["root"]))
    ]
    configured_output = cli.output_root or Path(config["output_root"])
    output_root = assert_outputs_not_in_data([configured_output], source_roots)[0]

    materialization = config.get("materialization", {})
    if str(materialization.get("mode", "file_symlink")).lower() != "file_symlink":
        raise ValueError("Only materialization.mode='file_symlink' is supported")

    cases = discover_cases(config)
    assignment, capacities, primary_quotas, objective = assign_splits(cases, config)
    plan, summary, training_paths = build_outputs(
        cases,
        assignment,
        capacities,
        primary_quotas,
        objective,
        config,
        cli.config,
        output_root,
    )
    write_outputs(cases, plan, summary, training_paths, output_root)

    print(f"Planned {len(cases)} patients: {capacities}")
    print(f"Split plan: {output_root / 'split_manifest.json'}")
    if cli.apply:
        directories, links, existing = materialize_file_symlinks(plan)
        print(
            f"Patient directories created={directories}, file links created={links}, "
            f"already_correct={existing}"
        )
    else:
        print("Plan-only mode: no patient symlinks were created. Use --apply to materialize.")


if __name__ == "__main__":
    main()
