"""Checkpoint retention, resume fallback and the per-rank case split."""

import tempfile
import unittest
from pathlib import Path

import torch

from .common import checkpoints
from .common.distributed import rank_share, size_buckets, sync_gradients
from .common.safety import OutputGuard


def state(completed):
    return {"model": {"w": torch.full((2,), float(completed))}, "optimizer": {}, "epoch": completed - 1, "step": completed * 10}


class CheckpointPolicyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir, self.guard = Path(self._tmp.name), OutputGuard()

    def tearDown(self):
        self._tmp.cleanup()

    def run_epochs(self, last, paper):
        for completed in range(1, last + 1):
            checkpoints.save_epoch(state(completed), self.dir, self.guard, completed, paper)
        return sorted(checkpoints.epoch_files(self.dir))

    def test_ours_keeps_two_rolling_and_every_permanent(self):
        paper = {"checkpoint_rolling_every": 5, "checkpoint_rolling_keep": 2, "checkpoint_permanent_every": 100}
        self.assertEqual(checkpoints.retained(range(5, 220, 5), 5, 2, 100), {100, 200, 210, 215})
        self.assertEqual(self.run_epochs(12, paper), [5, 10])
        self.assertEqual(self.run_epochs(17, paper), [10, 15])
        self.assertEqual(checkpoints.load(self.dir / "latest.pt")["epoch"], 16)
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")], [])

    def test_permanent_file_survives_rolling_cleanup(self):
        paper = {"checkpoint_rolling_every": 2, "checkpoint_rolling_keep": 2, "checkpoint_permanent_every": 6}
        self.assertEqual(self.run_epochs(13, paper), [6, 10, 12])

    def test_baseline_keeps_every_tenth_only(self):
        self.assertEqual(self.run_epochs(25, {"checkpoint_permanent_every": 10}), [10, 20])

    def test_default_policy_keeps_only_latest(self):
        self.assertEqual(self.run_epochs(3, {}), [])
        self.assertTrue((self.dir / "latest.pt").is_file())

    def test_resume_falls_back_when_latest_is_unreadable(self):
        self.run_epochs(12, {"checkpoint_rolling_every": 5, "checkpoint_rolling_keep": 2})
        (self.dir / "latest.pt").write_bytes(b"truncated")
        (self.dir / "epoch_0010.pt").write_bytes(b"")
        loaded = []
        for path in checkpoints.resume_candidates(self.dir):
            try:
                loaded.append((path.name, checkpoints.load(path)["epoch"]))
                break
            except Exception:
                loaded.append((path.name, None))
        self.assertEqual(loaded, [("latest.pt", None), ("epoch_0010.pt", None), ("epoch_0005.pt", 4)])

    def test_failed_write_keeps_previous_latest(self):
        self.run_epochs(2, {})
        with self.assertRaises(Exception):
            checkpoints.write({"model": lambda: None, **{k: 0 for k in ("optimizer", "epoch", "step")}}, self.dir / "latest.pt", self.guard)
        self.assertEqual(checkpoints.load(self.dir / "latest.pt")["epoch"], 1)

    def test_invalid_policy_is_rejected(self):
        for paper in ({"checkpoint_rolling_every": -1}, {"checkpoint_rolling_every": 5, "checkpoint_rolling_keep": 0},
                      {"checkpoint_permanent_every": 2.5}):
            with self.assertRaises(ValueError):
                checkpoints.policy(paper)


class RankShareTest(unittest.TestCase):
    def test_equal_counts_cover_everything(self):
        shares = [rank_share(range(10), 4, rank) for rank in range(4)]
        self.assertEqual({len(s) for s in shares}, {3})
        self.assertEqual(set().union(*shares), set(range(10)))
        self.assertEqual(sorted(x for s in shares for x in s), sorted([*range(10), 0, 1]))
        self.assertEqual(rank_share(range(8), 4, 1), [1, 5])
        self.assertEqual(rank_share(range(7), 1, 0), list(range(7)))

    def test_size_buckets_group_similar_costs_per_step(self):
        import numpy as np

        world, steps = 4, 2
        order = [int(v) for v in np.random.default_rng(0).permutation(21)]
        bucketed = size_buckets(order, lambda v: v, world, steps, np.random.default_rng(1))
        padded = order + order[:3]
        self.assertEqual(sorted(bucketed), sorted(padded))
        self.assertEqual(bucketed, size_buckets(order, lambda v: v, world, steps, np.random.default_rng(1)))
        for start in range(0, len(padded), world * steps):
            block = sorted(padded[start:start + world * steps])
            chunks = {tuple(block[i:i + world]) for i in range(0, len(block), world)}
            got = {tuple(sorted(bucketed[i:i + world])) for i in range(start, min(start + world * steps, len(padded)), world)}
            self.assertEqual(got, chunks)
        shares = [rank_share(bucketed, world, rank) for rank in range(world)]
        self.assertEqual({len(s) for s in shares}, {6})

    def test_single_process_gradients_are_untouched(self):
        weight = torch.nn.Parameter(torch.ones(3))
        weight.sum().backward()
        sync_gradients([weight], 1)
        self.assertTrue(torch.equal(weight.grad, torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
