"""Critical data safety, split and measurement tests for the isolated package."""

import csv
import json
from pathlib import Path

import pytest
import torch

from .common.data import sense_adjoint, sense_forward
from .common.manifest import read_manifest, validate_train_index
from .common.safety import OutputGuard
from .generate_configs import configuration


def test_protected_symlink_output_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    link = tmp_path / "innocent_link"
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="read-only"):
        OutputGuard([raw]).check(link / "new_cache.npy")
    assert not (raw / "new_cache.npy").exists()


def test_sense_forward_and_adjoint_match_for_odd_native_zy_sizes():
    torch.manual_seed(2026)
    x = torch.randn(1, 4, 5, 3, 7, 9, dtype=torch.complex64)
    s = torch.randn(1, 2, 3, 7, 9, dtype=torch.complex64)
    mask = (torch.rand(1, 5, 7, 9) > 0.4).float()
    y = torch.randn(1, 4, 5, 2, 3, 7, 9, dtype=torch.complex64)
    lhs = torch.vdot(sense_forward(x, s, mask).flatten(), y.flatten())
    rhs = torch.vdot(x.flatten(), sense_adjoint(y, s, mask).flatten())
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-4)


def test_ablation_switches_and_no_test_validation():
    for i in range(5):
        cfg = configuration(f"ablation_A{i}", Path("manifest.csv"), Path("tree.txt"))
        p = cfg["phase3"]
        assert cfg["num_cascades"] == 6 and p["num_slices"] == 3 and cfg["num_frames"] == 5
        assert p["backbone"]["spatial_dims"] == (2 if i == 0 else 3)
        assert p["flowvn_mixer"]["enabled"] == (i >= 2)
        assert p["joint_encoding"]["enabled"] == (i >= 3)
        assert p["loss"]["joint"]["enabled"] == (i >= 4)
        assert not p["enable_vaa"]
        assert cfg["data_path_val"] == [] and not cfg["paper"]["validation_enabled"]
        assert cfg["resume_ckpt"] is None


def test_manifest_preserves_test_label_and_readme_record(tmp_path):
    p = tmp_path / "manifest.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_file", "size_bytes", "shape", "dataset_group", "disk", "destination_path"])
        writer.writeheader()
        writer.writerow({"target_file": "/case/kdata_full.mat", "dataset_group": "ordinary_test", "disk": "Elements",
                         "destination_path": "/media/Elements/4DFlow/case/kdata_full.mat"})
        writer.writerow({"target_file": "/README.md", "dataset_group": "", "destination_path": "/media/Elements/4DFlow/README.md"})
    cases, issues = read_manifest(p)
    assert cases[0]["dataset_group"] == "ordinary_test"
    assert issues[0]["kind"] == "dataset_readme_metadata"


def test_index_cannot_relabel_a_test_case_as_train():
    original = {"directory": "/media/Elements/4DFlow/test_case", "dataset_group": "focused_test",
                "files": {"kdata_full.mat": "/media/Elements/4DFlow/test_case/kdata_full.mat"}}
    forged = {**original, "dataset_group": "train"}
    with pytest.raises(ValueError, match="original CSV"):
        validate_train_index([forged], [original])
