"""Standard-library tests: python3 -m unittest discover -s tests -p test_cleanup_checkpoints.py."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/cleanup_checkpoints.py"
if not SCRIPT.exists():  # Also allow testing a standalone development copy.
    SCRIPT = Path(__file__).with_name("cleanup_checkpoints.py")
SPEC = importlib.util.spec_from_file_location("cleanup_checkpoints", SCRIPT)
cleanup_checkpoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup_checkpoints)


class CheckpointCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def file(self, name):
        path = self.out / name
        path.write_bytes(b"checkpoint")
        return path

    def run_cleanup(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cleanup_checkpoints.time, "sleep"):
            return cleanup_checkpoints.cleanup(self.out, "model", min_bytes=1, **kwargs)

    def test_dry_run_default(self):
        old = self.file("model_epoch5.pt")
        self.file("model_epoch10.pt")
        self.assertEqual(self.run_cleanup(), [])
        self.assertTrue(old.exists())

    def test_retention_and_unrelated_files(self):
        names = ["model_epoch5.pt", "model_epoch95.pt", "model_epoch100.pt",
                 "model_epoch105.pt", "model_epoch200.pt", "model_epoch205.pt",
                 "model.pt", "model_best.pt", "model_epochbroken.pt", "other_epoch5.pt",
                 "model_epoch5.pt.tmp", "model_epoch-1.pt"]
        for name in names:
            self.file(name)
        removed = self.run_cleanup(apply=True)
        self.assertEqual({p.name for p in removed},
                         {"model_epoch5.pt", "model_epoch95.pt", "model_epoch105.pt"})
        self.assertEqual({p.name for p in self.out.iterdir()}, set(names) - {p.name for p in removed})

    def test_numeric_order_and_leading_zero(self):
        self.file("model_epoch9.pt")
        self.file("model_epoch010.pt")
        self.file("model_epoch10.pt")
        self.assertEqual([p.name for p in self.run_cleanup(apply=True)], ["model_epoch9.pt"])

    def test_latest_multiple_of_100(self):
        self.file("model_epoch95.pt")
        self.file("model_epoch100.pt")
        self.assertEqual([p.name for p in self.run_cleanup(apply=True)], ["model_epoch95.pt"])

    def test_no_numbered_files(self):
        rolling = self.file("model.pt")
        self.assertEqual(self.run_cleanup(apply=True), [])
        self.assertTrue(rolling.exists())

    def test_small_latest_refuses_deletion(self):
        old = self.file("model_epoch5.pt")
        self.file("model_epoch10.pt")
        with patch.object(cleanup_checkpoints.time, "sleep"), self.assertRaises(ValueError):
            cleanup_checkpoints.cleanup(self.out, "model", apply=True)
        self.assertTrue(old.exists())

    def test_latest_replacement_same_size_refuses_deletion(self):
        old = self.file("model_epoch5.pt")
        latest = self.file("model_epoch10.pt")
        replacement = self.file("replacement.tmp")
        with patch.object(cleanup_checkpoints.time, "sleep", side_effect=lambda _: replacement.replace(latest)):
            with self.assertRaises(ValueError):
                cleanup_checkpoints.cleanup(self.out, "model", min_bytes=1, apply=True)
        self.assertTrue(old.exists())

    def test_changed_candidate_is_skipped(self):
        old = self.file("model_epoch5.pt")
        self.file("model_epoch10.pt")
        with patch.object(cleanup_checkpoints.time, "sleep", side_effect=lambda _: old.write_bytes(b"changed")):
            self.assertEqual(cleanup_checkpoints.cleanup(self.out, "model", min_bytes=1, apply=True), [])
        self.assertTrue(old.exists())

    def test_symlink_and_directory_ignored(self):
        target = self.file("unrelated.pt")
        (self.out / "model_epoch5.pt").symlink_to(target)
        (self.out / "model_epoch10.pt").mkdir()
        self.assertEqual(self.run_cleanup(apply=True), [])
        self.assertTrue(target.exists())
        self.assertTrue((self.out / "model_epoch5.pt").is_symlink())

    def test_invalid_arguments(self):
        for kwargs in ({"keep_every": 0}, {"stability_seconds": -1}, {"stability_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                self.run_cleanup(**kwargs)
        with self.assertRaises(ValueError):
            cleanup_checkpoints.cleanup(self.out, "../model")

    def test_prefix_is_literal(self):
        old = self.file("model[1]_epoch5.pt")
        self.file("model[1]_epoch10.pt")
        other = self.file("model1_epoch5.pt")
        with patch.object(cleanup_checkpoints.time, "sleep"):
            self.assertEqual(cleanup_checkpoints.cleanup(self.out, "model[1]", min_bytes=1, apply=True), [old])
        self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
