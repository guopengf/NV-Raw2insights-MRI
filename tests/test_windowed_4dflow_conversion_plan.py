from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TOOLS_ROOT = SCRIPTS_ROOT / "tools"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from build_4dflow_windowed_h5 import (
    build_conversion_plan_payload,
    finalize_conversion_plan,
    load_conversion_plan,
    run_conversion_plan_shard,
    write_conversion_plan,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json"
)


def _assert_runtime_error(message: str, callback) -> None:
    try:
        callback()
    except RuntimeError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected RuntimeError containing {message!r}")


def _write_bytes(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * count)
    return path


def _fake_patient(root: Path, index: int, target_bytes: int) -> dict:
    patient_key = f"Center001/ScannerA/P{index:03d}"
    patient_root = root / patient_key
    return {
        "patient_key": patient_key,
        "target_path": _write_bytes(patient_root / "kdata_full.mat", target_bytes),
        "inputs": {10: _write_bytes(patient_root / "kdata_ktGaussian10.mat", index + 5)},
        "masks": {10: _write_bytes(patient_root / "usmask_ktGaussian10.mat", 3)},
        "coilmap_path": None,
        "segmask_path": None,
    }


def _synthetic_patient(root: Path) -> dict:
    rng = np.random.default_rng(17)
    shape = (4, 2, 2, 3, 4, 5)
    target = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    masked = (target * (0.8 + 0.05j)).astype(np.complex64)
    mask = rng.integers(0, 2, size=(1, 2, 1, 3, 4, 1)).astype(np.float32)
    coilmap = (
        rng.normal(size=(2, 3, 4, 5)) + 1j * rng.normal(size=(2, 3, 4, 5))
    ).astype(np.complex64)
    patient_root = root / "Center001" / "ScannerA" / "P001"
    patient_root.mkdir(parents=True)
    target_path = patient_root / "kdata_full.mat"
    input_path = patient_root / "kdata_ktGaussian10.mat"
    mask_path = patient_root / "usmask_ktGaussian10.mat"
    coilmap_path = patient_root / "coilmap.mat"
    scipy.io.savemat(target_path, {"kdata_full": target})
    scipy.io.savemat(input_path, {"kdata": masked})
    scipy.io.savemat(mask_path, {"mask": mask})
    scipy.io.savemat(coilmap_path, {"coilmap": coilmap})
    return {
        "patient_key": "Center001/ScannerA/P001",
        "target_path": target_path,
        "inputs": {10: input_path},
        "masks": {10: mask_path},
        "coilmap_path": coilmap_path,
        "segmask_path": None,
    }


def test_ten_shard_plan_is_deterministic_complete_and_balanced(tmp_path):
    patients = [
        _fake_patient(tmp_path / "raw", index, 100 + index * 13)
        for index in range(1, 38)
    ]
    first = build_conversion_plan_payload(
        config_path=CONFIG_PATH,
        output_root=tmp_path / "converted",
        patients=patients,
        num_shards=10,
        expected_patients=37,
    )
    second = build_conversion_plan_payload(
        config_path=CONFIG_PATH,
        output_root=tmp_path / "converted",
        patients=list(reversed(patients)),
        num_shards=10,
        expected_patients=37,
    )
    assert first == second
    assigned = [key for shard in first["shards"] for key in shard["patient_keys"]]
    assert len(assigned) == 37
    assert len(set(assigned)) == 37
    weights = [shard["estimated_conversion_bytes"] for shard in first["shards"]]
    assert max(weights) / (sum(weights) / len(weights)) < 1.15
    assert max(shard["patient_count"] for shard in first["shards"]) <= 4


def test_plan_hash_rejects_mutation(tmp_path):
    plan = build_conversion_plan_payload(
        config_path=CONFIG_PATH,
        output_root=tmp_path / "converted",
        patients=[_fake_patient(tmp_path / "raw", 1, 101)],
        num_shards=2,
    )
    plan_path = tmp_path / "plan.json"
    write_conversion_plan(plan, plan_path)
    assert load_conversion_plan(plan_path) == plan
    tampered = json.loads(plan_path.read_text())
    tampered["patient_count"] = 2
    plan_path.write_text(json.dumps(tampered))
    _assert_runtime_error("hash mismatch", lambda: load_conversion_plan(plan_path))


def test_workers_do_not_finalize_and_finalizer_is_strict(tmp_path):
    output_root = tmp_path / "converted"
    plan = build_conversion_plan_payload(
        config_path=CONFIG_PATH,
        output_root=output_root,
        patients=[_synthetic_patient(tmp_path / "raw")],
        num_shards=2,
        expected_patients=1,
    )
    plan_path = tmp_path / "control" / "plan.json"
    write_conversion_plan(plan, plan_path)

    run_conversion_plan_shard(plan_path, 0)
    assert not (output_root / "index.json").exists()
    assert not (output_root / "COMPLETED").exists()
    _assert_runtime_error(
        "Missing shard completion marker",
        lambda: finalize_conversion_plan(plan_path, deep_verify=True, expected_patients=1),
    )

    run_conversion_plan_shard(plan_path, 1)
    expected_store = output_root / "patients" / "Center001" / "ScannerA" / "P001.h5"
    extra_store = output_root / "patients" / "Center001" / "ScannerA" / "EXTRA.h5"
    shutil.copy2(expected_store, extra_store)
    _assert_runtime_error(
        "Patient store set mismatch",
        lambda: finalize_conversion_plan(plan_path, deep_verify=True, expected_patients=1),
    )
    extra_store.unlink()

    result = finalize_conversion_plan(plan_path, deep_verify=True, expected_patients=1)
    assert result["patients"] == 1
    assert result["e1_profile"] == "encoding_chunk_1"
    assert (output_root / "index.json").exists()
    assert (output_root / "COMPLETED").exists()
    assert not list(tmp_path.rglob("*.pt"))
