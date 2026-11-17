"""Logical identity/splits stay fixed while only physical read locations change."""

import csv
import json
from pathlib import Path

import pytest

from .common.case_data import select_cases
from .common.manifest import FILENAMES, inspect_train_case, read_manifest, validate_train_index
from .common.path_mapping import read_source_mapping
from .common.safety import OutputGuard


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def tables(tmp_path):
    manifests = tmp_path / "manifests"
    mappings = tmp_path / "mapping_inputs"
    manifests.mkdir()
    mappings.mkdir()
    rows, mapping = [], []
    for i, group in enumerate(("train", "ordinary_test"), 1):
        case_id = f"Center000_P{i:06d}_Aorta"
        physical = tmp_path / "nas" / f"original_case_{i}"
        physical.mkdir(parents=True)
        for name in FILENAMES:
            source = physical / name
            source.write_bytes(b"fixture")
            target = f"/{case_id}/{name}"
            row = {"target_file": target, "size_bytes": source.stat().st_size, "shape": "[4,2,1,3,4,5]",
                   "dataset_group": group, "disk": "unplugged", "destination_path": str(tmp_path / "absent_disk" / case_id / name)}
            rows.append(row)
            mapping.append({"source_file": str(source), "target_file": target,
                            "size_bytes": row["size_bytes"], "shape": row["shape"]})
    manifest, mp = manifests / "final.csv", mappings / "source.csv"
    write_csv(manifest, rows)
    write_csv(mp, mapping)
    return manifest, mp, mapping


def test_virtual_reads_preserve_original_identity_rows_and_splits(tables, tmp_path):
    manifest, mp, _ = tables
    before_manifest, before_mapping = manifest.read_bytes(), mp.read_bytes()
    original, _ = read_manifest(manifest)
    virtual, _ = read_manifest(manifest, mp)
    for old, new in zip(original, virtual):
        assert {k: old[k] for k in ("case_id", "directory", "dataset_group", "disk", "rows")} == {
            k: new[k] for k in ("case_id", "directory", "dataset_group", "disk", "rows")}
        assert new["logical_files"] == old["files"]
        assert new["files"] != old["files"]
        assert all(Path(path).is_file() for path in new["files"].values())
        assert not Path(new["directory"]).exists()
    selected, guard = select_cases(manifest, source_mapping=mp)
    assert [case["dataset_group"] for case in selected] == ["train"]
    validate_train_index(selected, virtual, require_complete=True)
    for path in (*selected[0]["files"].values(), selected[0]["directory"],
                 str(Path(selected[0]["files"]["kdata_full.mat"]).parent / "forbidden_cache.npy")):
        with pytest.raises(ValueError, match="read-only"):
            guard.check(path)
    assert guard.directory(tmp_path / "outputs").is_dir()
    assert manifest.read_bytes() == before_manifest and mp.read_bytes() == before_mapping


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_target", "duplicate_source", "size", "shape", "relative", "traversal"])
def test_bad_mappings_fail_explicitly(tables, change):
    manifest, mp, rows = tables
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append({**rows[0], "target_file": "/Unknown/kdata_full.mat", "source_file": "/unknown/kdata_full.mat"})
    elif change == "duplicate_target":
        rows.append(dict(rows[0]))
    elif change == "duplicate_source":
        rows[6]["source_file"] = rows[0]["source_file"]
    elif change == "size":
        rows[0]["size_bytes"] += 1
    elif change == "shape":
        rows[0]["shape"] = "[1,2,3]"
    elif change == "relative":
        rows[0]["source_file"] = "relative/kdata_full.mat"
    elif change == "traversal":
        rows[0]["target_file"] = "/case/../kdata_full.mat"
    write_csv(mp, rows)
    with pytest.raises(ValueError):
        read_manifest(manifest, mp)


def test_source_size_changes_are_not_silently_accepted(tables):
    manifest, mp, _ = tables
    cases, _ = read_manifest(manifest, mp)
    Path(cases[0]["files"]["coilmap.mat"]).write_bytes(b"changed fixture size")
    checked = inspect_train_case(cases[0])
    assert not checked["available"]
    assert any("Mapped source size changed" in problem for problem in checked["problems"])


def test_source_file_symlinks_cannot_escape_write_protection(tables, tmp_path):
    manifest, mp, rows = tables
    source = Path(rows[0]["source_file"])
    outside = tmp_path / "separate_disk" / source.name
    outside.parent.mkdir()
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    _, guard = select_cases(manifest, source_mapping=mp)
    with pytest.raises(ValueError, match="read-only"):
        guard.check(outside)
    link = tmp_path / "virtual_link"
    link.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="read-only"):
        guard.check(link / "forbidden.npy")


def test_mapping_does_not_authorize_test_as_train(tables):
    manifest, mp, _ = tables
    cases, _ = read_manifest(manifest, mp)
    invalid = {**cases[1], "dataset_group": "train"}
    with pytest.raises(ValueError, match="original CSV"):
        validate_train_index([invalid], cases)


def test_readme_row_is_metadata_not_an_extra_case(tables):
    manifest, mp, rows = tables
    rows.append({"source_file": "/unavailable/source/README.md", "target_file": "/README.md", "size_bytes": 942, "shape": ""})
    write_csv(mp, rows)
    assert len(read_source_mapping(mp)) == 13
    cases, _ = read_manifest(manifest, mp)
    assert len(cases) == 2


def test_virtual_builder_keeps_model_configs_and_inputs_unchanged(tmp_path, monkeypatch):
    import h5py

    from .build_virtual_dataset import main
    from .common.safety import ROOT
    from .test_full_case import case_and_truth

    case, _, _ = case_and_truth.__wrapped__(tmp_path)
    inputs = tmp_path / "index_inputs"
    inputs.mkdir()
    rows, mapping = [], []
    for name, physical in case["files"].items():
        source = Path(physical)
        shape = [1, 4]
        if name.endswith(".mat"):
            with h5py.File(source, "r") as f:
                shape = list(next(iter(f.values())).shape)
        target = f"/{case['case_id']}/{name}"
        row = {"target_file": target, "size_bytes": source.stat().st_size, "shape": json.dumps(shape),
               "dataset_group": "train", "disk": "unplugged", "destination_path": str(tmp_path / "absent_disk" / case["case_id"] / name)}
        rows.append(row)
        mapping.append({"source_file": physical, "target_file": target, "size_bytes": row["size_bytes"], "shape": row["shape"]})
    manifest, mp = inputs / "final.csv", inputs / "mapping.csv"
    write_csv(manifest, rows)
    write_csv(mp, mapping)
    before = manifest.read_bytes(), mp.read_bytes()
    output = tmp_path / "virtual"
    monkeypatch.setattr("sys.argv", ["build_virtual_dataset", "--manifest", str(manifest), "--mapping", str(mp), "--output", str(output)])
    main()
    report = json.loads((output / "mapping_report.json").read_text())
    assert report["passed"] and report["train_headers_checked"][0]["passed"]
    assert not report["symlinks_created"] and not report["dataset_files_copied"] and not report["test_contents_read"]
    for entry in json.loads((ROOT / "experiments/paper/experiment_registry.json").read_text())["experiments"]:
        original = json.loads((ROOT / entry["config"]).read_text())
        generated = json.loads((output / "configs" / f"{entry['name']}.json").read_text())
        new_paper, old_paper = generated.pop("paper"), original.pop("paper")
        assert generated == original
        for key in ("source_mapping", "train_index", "output_dir"):
            new_paper.pop(key, None)
            old_paper.pop(key, None)
        assert new_paper == old_paper
    assert before == (manifest.read_bytes(), mp.read_bytes())
    assert (output / "inputs/source_mapping.csv").read_bytes() == mp.read_bytes()


def test_guard_prunes_nested_roots_without_weakening_protection(tmp_path):
    parent = tmp_path / "readonly"
    guard = OutputGuard([parent, parent / "case", parent / "case/file.mat"])
    assert parent.resolve() in guard.protected
    assert parent.resolve() / "case" not in guard.protected
    with pytest.raises(ValueError, match="read-only"):
        guard.check(parent / "unlisted_case/cache.npy")
