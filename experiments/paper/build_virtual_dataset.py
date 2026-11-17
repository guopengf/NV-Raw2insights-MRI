"""Build read-only logical-to-NAS indices without changing the final manifest or data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import uuid

from .common.manifest import inspect_train_case, read_manifest, validate_train_index
from .common.path_mapping import protected_case_paths
from .common.safety import ROOT, OutputGuard, sha256


def portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", type=Path, required=True, help="CSV with source_file,target_file,size_bytes,shape")
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/paper/inputs/final_dataset_manifest.csv")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/paper/virtual_dataset")
    p.add_argument("--check-train-cases", type=int, default=1, help="Read headers of 1-3 smallest Train cases only")
    args = p.parse_args()
    if not 1 <= args.check_train_cases <= 3:
        p.error("Only 1-3 Train cases are inspected during virtual-path setup")
    manifest_sha, mapping_sha = sha256(args.manifest), sha256(args.mapping)
    cases, issues = read_manifest(args.manifest, args.mapping)
    guard = OutputGuard([args.manifest.parent, args.mapping.parent, *protected_case_paths(cases)])
    output = guard.directory(args.output)
    inputs = guard.directory(output / "inputs")
    snapshot = guard.check(inputs / "source_mapping.csv")
    if snapshot.exists() and sha256(snapshot) != mapping_sha:
        raise ValueError("Output contains another source mapping; use a separate --output directory")
    if not snapshot.exists():
        temporary = guard.check(inputs / f"mapping.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(args.mapping, temporary)
            if sha256(temporary) != mapping_sha:
                raise RuntimeError("Source mapping changed while copying")
            temporary.replace(snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    cases, _ = read_manifest(args.manifest, snapshot)
    train = [c for c in cases if c["dataset_group"] == "train"]
    validate_train_index(train, cases, require_complete=True)
    candidates = sorted(train, key=lambda c: next(int(r["size_bytes"]) for r in c["rows"] if r["target_file"].endswith("/kdata_full.mat")))
    checked = [inspect_train_case(c, headers=True) for c in candidates[:args.check_train_cases]]
    guard.json(output / "cases_all_labels.json", cases)
    guard.json(output / "train_index.json", train)
    guard.json(output / "checked_train_cases.json", checked)
    pipeline = json.loads((ROOT / "configs/paper/evaluation.json").read_text())
    pipeline.update(manifest=portable(args.manifest), source_mapping=portable(snapshot),
                    cache_root=portable(output / "cache"), output_root=portable(output / "full_case"))
    guard.json(output / "pipeline.json", pipeline)
    registry = json.loads((ROOT / "experiments/paper/experiment_registry.json").read_text())["experiments"]
    configurations = []
    for entry in registry:
        cfg = json.loads((ROOT / entry["config"]).read_text())
        cfg["paper"].update(source_mapping=portable(snapshot), train_index=portable(output / "train_index.json"),
                            output_dir=portable(output / "runs" / entry["name"]))
        path = guard.json(output / "configs" / f"{entry['name']}.json", cfg)
        configurations.append({"experiment": entry["name"], "config": portable(path), "sha256": sha256(path)})
    protected = True
    for path in (train[0]["directory"], *train[0]["files"].values()):
        try:
            guard.check(path)
            protected = False
        except ValueError:
            pass
    if sha256(args.manifest) != manifest_sha or sha256(args.mapping) != mapping_sha:
        raise RuntimeError("An input CSV changed during virtual-index generation")
    report = {"mode": "read_only_path_resolution", "symlinks_created": False, "dataset_files_copied": False,
              "original_manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
              "source_mapping_input": str(args.mapping.resolve()), "source_mapping_sha256": mapping_sha,
              "source_mapping_snapshot": portable(snapshot), "case_counts": dict(Counter(c["dataset_group"] for c in cases)),
              "matched_data_files": sum(len(c["files"]) for c in cases), "unmapped_data_files": 0,
              "metadata_issues": issues, "test_contents_read": False,
              "train_headers_checked": [{"case_id": c["case_id"], "passed": c["available"], "problems": c["problems"],
                                         "kspace_shape": c["headers"].get("kdata_full.mat", {}).get("shape")} for c in checked],
              "all_indexed_sources_physically_checked": False,
              "physical_and_logical_paths_write_protected": protected,
              "pipeline": portable(output / "pipeline.json"), "training_configs": configurations}
    report["passed"] = protected and bool(checked) and all(c["available"] for c in checked)
    guard.json(output / "mapping_report.json", report)
    print(json.dumps({key: report[key] for key in ("case_counts", "matched_data_files", "train_headers_checked", "passed")}, indent=2), flush=True)
    print(f"Virtual pipeline: {output / 'pipeline.json'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
