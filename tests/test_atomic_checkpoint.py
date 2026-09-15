from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils import atomic_torch_save


class AtomicTorchSaveTest(unittest.TestCase):
    def test_replaces_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "checkpoint.pt"
            destination.write_bytes(b"old checkpoint")

            atomic_torch_save({"epoch": 12, "tensor": torch.arange(4)}, destination)

            payload = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 12)
            self.assertTrue(torch.equal(payload["tensor"], torch.arange(4)))
            self.assertFalse((root / ".checkpoint.pt.tmp").exists())

    def test_preserves_destination_when_save_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "checkpoint.pt"
            destination.write_bytes(b"known-good checkpoint")

            def fail_after_partial_write(_payload, path):
                Path(path).write_bytes(b"partial checkpoint")
                raise RuntimeError("injected save failure")

            with mock.patch("utils.torch.save", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(RuntimeError, "injected save failure"):
                    atomic_torch_save({"epoch": 13}, destination)

            self.assertEqual(destination.read_bytes(), b"known-good checkpoint")
            self.assertFalse((root / ".checkpoint.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
