"""Read the supplied manifest without changing labels, paths or dataset layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .safety import OutputGuard, sha256
from .path_mapping import map_cases, protected_case_paths


FILENAMES = ("kdata_full.mat", "coilmap.mat", "corrmap.npz", "params.csv", "segmask.mat", "segmask_new.mat")
GROUPS = {"train", "ordinary_test", "focused_test"}


def validate_train_index(index, source_cases, *, require_complete=False):
    allowed = {c["directory"]: c for c in source_cases if c["dataset_group"] == "train"}
    for case in index:
        original = allowed.get(case["directory"])
        if original is None or case["dataset_group"] != "train" or case["files"] != original["files"]:
            raise ValueError(f"Training index does not match original CSV train mapping: {case['directory']}")
    if require_complete and {c["directory"] for c in index} != set(allowed):
        raise ValueError("Formal training requires the complete original train cohort; rerun availability indexing after copying finishes")


def read_manifest(path, source_mapping=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"target_file", "dataset_group", "disk", "destination_path", "shape", "size_bytes"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest fields missing: {required - set(reader.fieldnames or ())}")
        rows = list(reader)
    cases, issues, destinations = {}, [], set()
    for line, row in enumerate(rows, 2):
        if not row.get("dataset_group") and row.get("target_file") == "/README.md":
            issues.append({"line": line, "kind": "dataset_readme_metadata", "row": row})
            continue
        group = row["dataset_group"]
        if group not in GROUPS:
            raise ValueError(f"Invalid dataset_group on line {line}: {group!r}")
        destination = Path(row["destination_path"])
        if not destination.is_absolute():
            raise ValueError(f"destination_path must be absolute, line {line}")
        if str(destination) in destinations:
            raise ValueError(f"Duplicate destination_path: {destination}")
        destinations.add(str(destination))
        target = Path(row["target_file"].lstrip("/"))
        if destination.name != target.name or destination.parent.name != target.parent.name:
            raise ValueError(f"target_file/destination_path disagree: {row}")
        key = str(destination.parent)
        case = cases.setdefault(key, {"case_id": destination.parent.name, "directory": key,
                                      "dataset_group": group, "disk": row["disk"], "files": {}, "rows": []})
        if case["dataset_group"] != group:
            raise ValueError(f"Case has inconsistent split labels: {key}")
        case["files"][destination.name] = row["destination_path"]
        case["rows"].append(row)
    if len({c["case_id"] for c in cases.values()}) != len(cases):
        raise ValueError("Patient directory identifiers collide across disks")
    ordered = sorted(cases.values(), key=lambda c: c["case_id"])
    return map_cases(ordered, source_mapping) if source_mapping is not None else ordered, issues


def read_params(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row: {path}")
    row = rows[0]
    for key in ("VENC", "venc_order", "spatial_order", "matrix_size"):
        if not row.get(key):
            raise ValueError(f"Missing {key}: {path}")
    venc = [float(v) for v in row["VENC"].split(";")]
    if len(venc) != 3 or min(venc) <= 0:
        raise ValueError(f"Invalid VENC: {path}: {venc}")
    if len(row["venc_order"].split(";")) != 3:
        raise ValueError(f"Invalid encoding directions: {path}")
    return row


def inspect_train_case(case, headers=False):
    import h5py

    result = {**case, "available": True, "problems": [], "headers": {}}
    for name in FILENAMES:
        path = Path(case["files"].get(name, "/__missing__"))
        if not path.is_file():
            result["problems"].append(f"Missing {name}: {path}")
        elif case.get("source_mapping"):
            row = next(r for r in case["rows"] if Path(r["target_file"]).name == name)
            if path.stat().st_size != int(row["size_bytes"]):
                result["problems"].append(f"Mapped source size changed since manifest: {name}")
    if result["problems"]:
        result["available"] = False
        return result
    try:
        result["params"] = read_params(case["files"]["params.csv"])
        if headers:
            for name in ("kdata_full.mat", "coilmap.mat", "segmask.mat", "segmask_new.mat"):
                with h5py.File(case["files"][name], "r") as f:
                    key = "segmask" if name.startswith("segmask") else name[:-4]
                    d = f[key]
                    result["headers"][name] = {"key": key, "shape": list(d.shape), "dtype": str(d.dtype)}
            k = result["headers"]["kdata_full.mat"]
            shape = k["shape"]
            if len(shape) != 6 or shape[0] != 4:
                raise ValueError(f"Expected [4,T,C,Z,Y,X], got {shape}")
            if "real" not in k["dtype"] and "complex" not in k["dtype"]:
                raise ValueError(f"K-space is not complex: {k}")
            if result["headers"]["coilmap.mat"]["shape"] != shape[2:]:
                raise ValueError("Coilmap shape disagrees with k-space")
            if result["headers"]["segmask.mat"]["shape"] != shape[3:]:
                raise ValueError("Static segmentation shape disagrees with k-space")
            if result["headers"]["segmask_new.mat"]["shape"] != [shape[1], *shape[3:]]:
                raise ValueError("Temporal segmentation shape disagrees with k-space")
            declared = next(json.loads(r["shape"]) for r in case["rows"] if r["target_file"].endswith("/kdata_full.mat"))
            if declared != shape:
                raise ValueError(f"Manifest/actual shape mismatch: {declared} vs {shape}")
    except Exception as exc:
        result["available"] = False
        result["problems"].append(f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, help="Resolve original destinations to read-only source_file paths")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/index"))
    parser.add_argument("--headers", action="store_true", help="Read training file headers only; never inspect test contents")
    args = parser.parse_args()
    cases, issues = read_manifest(args.manifest, args.source_mapping)
    protected = list(protected_case_paths(cases))
    protected.extend((args.manifest.parent,))
    if args.source_mapping:
        protected.append(args.source_mapping.parent)
    guard = OutputGuard(protected)
    output = guard.directory(args.output)
    train = [inspect_train_case(c, args.headers) for c in cases if c["dataset_group"] == "train"]
    usable = [c for c in train if c["available"]]
    summary = {"manifest_sha256": sha256(args.manifest), "tree_sha256": sha256(args.tree),
               "source_mapping_sha256": sha256(args.source_mapping) if args.source_mapping else None,
               "case_counts": dict(Counter(c["dataset_group"] for c in cases)), "train_available": len(usable),
               "train_unavailable": len(train) - len(usable), "header_check": args.headers,
               "validation": None, "issues": issues, "test_contents_read": False}
    guard.json(output / "cases_all_labels.json", cases)
    guard.json(output / "train_index.json", usable)
    guard.json(output / "availability.json", {"summary": summary, "train": train})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not usable:
        raise SystemExit("No complete train case is currently available; see availability.json")


if __name__ == "__main__":
    main()
