"""Case scheduling, bounded cropped-frame reuse, and cluster delivery invariants."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .common.data import CaseFrames, TrainSlabs, ifftc
from .common.safety import ROOT
from .common.train_loader import CaseWindowLoader, case_order
from .test_online_sampling import train_index  # noqa: F401


class CountedKspace:
    def __init__(self):
        rng = np.random.default_rng(4)
        shape = (4, 4, 2, 5, 8, 9)
        self.array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        self.shape = shape
        self.reads = []

    def __getitem__(self, item):
        self.reads.append(item)
        return self.array[item]


@pytest.mark.parametrize("limit", [0, 1920, 100000])
def test_cropped_cache_matches_reference_and_reuses_reads(limit):
    raw = CountedKspace()
    plans = [(np.array(xs), np.array(ts), None, None)
             for xs, ts in [([0, 0, 1], [3, 0, 1]), ([5, 6, 7], [0, 1, 2]), ([7, 8, 8], [2, 3, 0, 0])]]
    frames = CaseFrames(raw, plans, limit)
    for i, (xs, ts, _, _) in enumerate(plans):
        expected = ifftc(raw.array[:, ts], (-1,))[..., xs].transpose(0, 1, 2, 5, 3, 4)
        np.testing.assert_allclose(frames.read(i, ts, xs), expected, atol=1e-6)
        assert frames.bytes <= limit
    assert not frames.cache
    if limit == 100000:
        assert len(raw.reads) == len(set(raw.reads)) == 16
    if limit == 0:
        assert len(raw.reads) == 36


def test_shuffle_preserves_complete_cases_and_seed():
    data = SimpleNamespace(seed=2, epoch=0, cases=list(range(5)), samples_per_case=8)
    first = list(case_order(data, True))
    assert len(first) == 5 and first == list(case_order(data, True))
    assert sorted(i for _, samples in first for i in samples) == list(range(40))
    assert all(all(i // 8 == case for i in samples) for case, samples in first)
    data.epoch = 1
    assert first != list(case_order(data, True))


@pytest.mark.parametrize("method", ["ours", "cut_4dflow"])
def test_prefetch_matches_direct_sample_and_closes_early(train_index, method):
    data = TrainSlabs(train_index)
    expected = data[0]
    loader = CaseWindowLoader(data, "cpu", method, shuffle=False)
    for _ in range(2):
        with loader.epoch_batches() as batches:
            batch = next(batches)
            for key in ("kspace", "target", "coils", "mask"):
                torch.testing.assert_close(batch[key][0], expected[key])
            assert ("full_kspace" in batch) == (method == "cut_4dflow")


def test_nine_cluster_configs_enable_online_sampling_and_prefetch():
    paths = (ROOT / "configs/paper").rglob("*.json")
    jobs = [cfg["paper"] for p in paths if (cfg := json.loads(p.read_text())).get("paper", {}).get("method")
            in {"ours", "flowvn", "flowmodl", "cut_4dflow"}]
    assert len(jobs) == 9
    for p in jobs:
        online = p["online_undersampling"]
        assert online["enabled"] and online["rate_range"] == [10, 50]
        assert online["rate_mode"] == "continuous" and online["methods"] == ["gaussian", "pr4d", "pseudo", "uih"]
        assert p["loader"]["prefetch"]
