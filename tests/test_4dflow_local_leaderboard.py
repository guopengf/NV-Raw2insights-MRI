import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tools import build_4dflow_validation_gt as gt_builder
from scripts.tools import evaluate_4dflow_submission as evaluator


class LocalLeaderboardTest(unittest.TestCase):
    def test_compound_complex_conversion(self):
        stored = np.zeros((2, 3), dtype=[("real", "<f4"), ("imag", "<f4")])
        stored["real"] = 2
        stored["imag"] = -3
        result = gt_builder.compound_to_complex(stored)
        self.assertEqual(result.dtype, np.complex64)
        np.testing.assert_array_equal(result, np.full((2, 3), 2 - 3j, dtype=np.complex64))

    def test_centered_ifft_matches_organizer_formula(self):
        rng = np.random.default_rng(7)
        value = (rng.standard_normal((2, 4, 5, 6)) + 1j * rng.standard_normal((2, 4, 5, 6))).astype(
            np.complex64
        )
        axes = (-3, -2, -1)
        expected = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(value, axes=axes), axes=axes, norm="ortho"), axes=axes
        )
        actual = gt_builder.centered_ifft3(value, workers=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_venc_csv_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "params.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["VENC", "system_model"])
                writer.writeheader()
                writer.writerow({"VENC": "150; 160;170", "system_model": "scanner"})
            np.testing.assert_array_equal(
                evaluator.load_venc(path), np.array([150, 160, 170], dtype=np.float32)
            )

    def test_dense_gt_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "img_gt.npy"
            expected = np.ones((4, 2, 3, 4, 5), dtype=np.complex64) * (1 + 2j)
            np.save(path, expected)
            actual = evaluator.load_dense(None, path)
            np.testing.assert_array_equal(actual, expected)

    def test_bbox_is_tight_and_preserves_prefix_axes(self):
        mask = np.zeros((5, 6, 7), dtype=np.uint8)
        mask[1:4, 2:5, 3:6] = 1
        gt = np.zeros((4, 2, 5, 6, 7), dtype=np.complex64)
        pred = gt.copy()
        corr = np.zeros((3, 2, 5, 6, 7), dtype=np.float32)
        gt_c, pred_c, mask_c, corr_c = evaluator.crop_to_seg_bbox(gt, pred, mask, corr)
        self.assertEqual(gt_c.shape, (4, 2, 3, 3, 3))
        self.assertEqual(pred_c.shape, gt_c.shape)
        self.assertEqual(mask_c.shape, (3, 3, 3))
        self.assertEqual(corr_c.shape, (3, 2, 3, 3, 3))


if __name__ == "__main__":
    unittest.main()
