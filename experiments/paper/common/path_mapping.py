"""Resolve logical manifest paths through an explicit read-only source-file mapping."""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

from .safety import sha256


def read_source_mapping(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_file", "target_file", "size_bytes", "shape"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Source mapping fields missing: {required - set(reader.fieldnames or ())}")
        rows = list(reader)
    result, sources = {}, {}
    for row in rows:
        target, source = row["target_file"], row["source_file"]
        logical, physical = PurePosixPath(target), Path(source)
        if not logical.is_absolute() or ".." in logical.parts or str(logical) != target:
            raise ValueError(f"Invalid logical target in source mapping: {target!r}")
        if not physical.is_absolute() or ".." in physical.parts or physical.name != logical.name:
            raise ValueError(f"Invalid source path or filename mismatch: {target!r} -> {source!r}")
        if target in result:
            raise ValueError(f"Duplicate mapping target: {target}")
        if source in sources:
            raise ValueError(f"Source reused by distinct targets: {sources[source]} and {target}")
        if int(row["size_bytes"]) < 0:
            raise ValueError(f"Invalid mapped file size: {target}")
        if target != "/README.md":
            shape = json.loads(row["shape"])
            if not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
                raise ValueError(f"Invalid mapped shape: {target}")
        result[target], sources[source] = row, target
    return result


def map_cases(cases, source_mapping):
    """Join by target_file, never by shape, patient order, or guessed source folder."""
    mapping = read_source_mapping(source_mapping)
    expected = {r["target_file"] for case in cases for r in case["rows"]}
    missing, extra = expected - set(mapping), set(mapping) - expected - {"/README.md"}
    if missing or extra:
        raise ValueError(f"Source mapping coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    provenance = {"path": str(Path(source_mapping).resolve()), "sha256": sha256(source_mapping),
                  "mode": "read_only_path_resolution"}
    mapped = []
    for case in cases:
        files = {}
        for row in case["rows"]:
            entry = mapping[row["target_file"]]
            if int(entry["size_bytes"]) != int(row["size_bytes"]) or json.loads(entry["shape"]) != json.loads(row["shape"]):
                raise ValueError(f"Source mapping/manifest size or shape mismatch: {row['target_file']}")
            files[Path(row["destination_path"]).name] = entry["source_file"]
        mapped.append({**case, "files": files, "logical_files": dict(case["files"]), "source_mapping": provenance})
    return mapped


def protected_case_paths(cases):
    paths = {case["directory"] for case in cases}
    for case in cases:
        for path in case["files"].values():
            paths.add(path)
            paths.add(str(Path(path).parent))
    return paths
