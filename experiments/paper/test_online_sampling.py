"""Online masks obey case/epoch scope and the final suite's native conventions."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from .common.data import TrainSlabs
from .common.online_sampling import METHODS, OnlineUndersampling, online_sampler, sampling_log
from .generate_configs import ONLINE_UNDERSAMPLING


@pytest.fixture
def suite_config():
    if not Path(ONLINE_UNDERSAMPLING["suite_path"]).is_file():
        pytest.skip("final external mask suite is not mounted")
    return {**ONLINE_UNDERSAMPLING, "cache_cases": 1}


@pytest.mark.parametrize("method", METHODS)
def test_final_suite_native_shape_budget_and_distinct_seeds(suite_config, method):
    cfg = {**suite_config, "methods": [method], "rate_range": [27.5, 27.5]}
    sampler = OnlineUndersampling(cfg, 2026)
    shape = (5, 19, 97)
    first, meta = sampler.get("caseA", 0, *shape)
    np.testing.assert_array_equal(sampler.get("caseA", 0, *shape)[0], first)
    second, later = sampler.get("caseA", 1, *shape)
    assert meta["seed"] != later["seed"]
    assert meta["encodings_shared"]
    assert meta["requested_R_grid"] == later["requested_R_grid"] == 27.5
    assert not np.array_equal(first, second)
    assert first.shape == (shape[1:] if method == "uih" else shape)
    np.testing.assert_array_equal(first.sum((-2, -1)), int(np.floor(shape[1] * shape[2] / 27.5)))
    if method != "uih":
        assert np.all(np.any(first[1:] != first[:-1], axis=(1, 2)))
        assert first[:, 8:11, 47:50].all()
    regenerated, same = sampler.get("caseA", 0, *shape)
    np.testing.assert_array_equal(first, regenerated)
    assert same["seed"] == meta["seed"]


@pytest.fixture
def train_index(tmp_path):
    nt, z, y, x = 7, 19, 96, 3
    arrays = {"kdata_full.mat": ("kdata_full", np.ones((4, nt, 1, z, y, x), np.complex64)),
              "coilmap.mat": ("coilmap", np.ones((1, z, y, x), np.complex64)),
              "segmask.mat": ("segmask", np.ones((z, y, x), np.uint8))}
    files = {}
    for name, (key, array) in arrays.items():
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f[key] = array
        files[name] = str(path)
    params = tmp_path / "params.csv"
    params.write_text(f"VENC,venc_order,spatial_order,matrix_size\n100;150;200,FH;AP;RL,FH;AP;RL,4;{nt};1;{z};{y};{x}\n")
    files["params.csv"] = str(params)
    index = tmp_path / "index.json"
    # No undersampled files are available in this fixture.
    index.write_text(json.dumps([{"case_id": "caseA", "dataset_group": "train", "files": files,
                                  "rows": [{"target_file": "/caseA/kdata_full.mat", "size_bytes": 1000}]}]))
    return index


@pytest.mark.parametrize("method", ["pseudo", "uih"])
def test_one_case_mask_survives_window_size_shuffling_and_workers(train_index, suite_config, method):
    cfg = {**suite_config, "methods": [method]}
    first = TrainSlabs(train_index, num_frames=5, num_slices=3, samples_per_case=4, online_undersampling=cfg)
    second = TrainSlabs(train_index, num_frames=6, num_slices=1, samples_per_case=4, online_undersampling=cfg)
    patterns = {}
    expected = None
    for dataset, workers in ((first, 0), (second, 2)):
        worker_options = {"multiprocessing_context": "spawn", "timeout": 60} if workers else {}
        for batch in DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=True, **worker_options):
            current = sampling_log(batch)
            if expected is None:
                expected = current
            assert current == expected
            assert 10 < current["requested_R"] < 50 and not current["requested_R"].is_integer()
            torch.testing.assert_close(batch["kspace"], batch["full_kspace"] * batch["mask"][:, None, :, None, None])
            for ti, plane in zip(batch["frame_indices"][0].tolist(), batch["mask"][0]):
                torch.testing.assert_close(plane, patterns.setdefault(ti, plane))
    if method == "uih":
        assert all(torch.equal(p, next(iter(patterns.values()))) for p in patterns.values())


def test_switch_off_does_not_require_external_files(train_index):
    assert online_sampler({"enabled": False, "suite_path": "/absent/mask_suite.py"}, 2026) is None
    before = TrainSlabs(train_index)[0]
    after = TrainSlabs(train_index, online_undersampling={"enabled": False})[0]
    torch.testing.assert_close(before["mask"], after["mask"])
    torch.testing.assert_close(before["kspace"], after["kspace"])
    assert before["acceleration"] == after["acceleration"] and "sampling_seed" not in after


def test_integer_mode_includes_boundaries(suite_config):
    cfg = {**suite_config, "methods": ["gaussian"], "rate_mode": "integer"}
    sampler = OnlineUndersampling(cfg, 2026)
    rates = {sampler.get(f"integer-case-{i}", 0, 2, 19, 96)[1]["requested_R_grid"] for i in range(256)}
    assert rates == set(range(10, 51))
