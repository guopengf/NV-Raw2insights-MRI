import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

import final_v2_h5 as fv2  # noqa: E402
from mra_utils import load_vessel_mask_prior, read_mat_array  # noqa: E402
from transforms import raw_4dflow_to_joint_hybrid  # noqa: E402
from windowed_4dflow import _coilmap_to_compact  # noqa: E402

SHAPE = [4, 3, 2, 4, 6, 8]  # E,T,C,Z,Y,X
COMPLEX = np.dtype([("real", "<f4"), ("imag", "<f4")])


def complex_array(rng, shape):
    out = np.empty(shape, COMPLEX)
    out["real"], out["imag"] = rng.normal(size=shape), rng.normal(size=shape)
    return out


def make_case(root, case_id, seed=0):
    rng = np.random.default_rng(seed)
    d = Path(root) / case_id
    d.mkdir(parents=True)
    e, t, c, z, y, x = SHAPE
    with h5py.File(d / "kdata_full.mat", "w") as f:
        f["kdata_full"] = complex_array(rng, SHAPE)
    with h5py.File(d / "coilmap.mat", "w") as f:
        f["coilmap"] = complex_array(rng, (c, z, y, x))
    with h5py.File(d / "segmask.mat", "w") as f:
        f["segmask"] = (rng.random((z, y, x)) > 0.5).astype(np.float32)
    (d / "params.csv").write_text("VENC,matrix_size\n1;1;1,4;3;2\n")
    return d


def case_info(d):
    return {"sizes": {n: (d / n).stat().st_size for n in fv2.REQUIRED}, "kdata_shape": SHAPE}


class FinalV2H5Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data, self.out = self.root / "data", self.root / "out"
        self.src = make_case(self.data, "Center001_P000001_Aorta")
        self.info = case_info(self.src)
        self.cid = "Center001_P000001_Aorta"

    def tearDown(self):
        self._tmp.cleanup()

    def sources(self, **kw):
        old = time.time() - 3600
        for n in fv2.REQUIRED:
            os.utime(self.src / n, (old, old))
        return fv2.scan(self.data, {self.cid: self.info}, **kw)[self.cid]

    def test_matches_existing_pipeline_exactly(self):
        s = self.sources()
        self.assertTrue(s["ready"])
        status, _, _ = fv2.convert_case(self.cid, self.info, self.data, self.out, s["sources"], samples=3)
        self.assertEqual(status, "converted")
        expected = raw_4dflow_to_joint_hybrid(read_mat_array(self.src / "kdata_full.mat", ("kdata_full",)))
        coil, layout = _coilmap_to_compact(read_mat_array(self.src / "coilmap.mat", ("coilmap",)),
                                           nx=8, nc=2, n_enc=4, axis_order="auto", normalize=True)
        with h5py.File(fv2.store_path(self.out, self.cid), "r") as f:
            self.assertTrue(np.array_equal(f["hybrid/target"][()], expected))
            self.assertTrue(np.array_equal(f["coilmap"][()], coil))
            self.assertEqual(f["coilmap"].attrs["layout"], layout)
            self.assertTrue(np.array_equal(f["segmask"][()], load_vessel_mask_prior(self.src / "segmask.mat", None)))
            self.assertEqual(f["hybrid/target"].chunks, (1, 1, 1, 2, 4, 6, 2))
            self.assertIn("matrix_size", f.attrs["params_csv"])
        self.assertEqual(list((self.out / "cases").glob(".*")), [])

    def test_skip_then_rebuild_when_source_changes(self):
        s = self.sources()["sources"]
        self.assertEqual(fv2.convert_case(self.cid, self.info, self.data, self.out, s)[0], "converted")
        self.assertEqual(fv2.convert_case(self.cid, self.info, self.data, self.out, s)[0], "skipped")
        new = os.stat(self.src / "kdata_full.mat").st_mtime + 5
        os.utime(self.src / "kdata_full.mat", (new, new))
        s2 = fv2.scan(self.data, {self.cid: self.info}, recent=0)[self.cid]["sources"]
        self.assertNotEqual(s, s2)
        self.assertEqual(fv2.convert_case(self.cid, self.info, self.data, self.out, s2)[0], "converted")

    def test_corrupt_store_is_rebuilt(self):
        s = self.sources()["sources"]
        fv2.convert_case(self.cid, self.info, self.data, self.out, s)
        fv2.store_path(self.out, self.cid).write_bytes(b"junk")
        self.assertEqual(fv2.convert_case(self.cid, self.info, self.data, self.out, s)[0], "converted")

    def test_scan_flags_unsafe_sources(self):
        self.assertTrue(self.sources()["ready"])
        (self.src / ".kdata_full.mat.AbCdEf").write_bytes(b"x")
        self.assertIn("temp_file", self.sources()["reasons"])
        (self.src / ".kdata_full.mat.AbCdEf").unlink()
        now = time.time()
        os.utime(self.src / "params.csv", (now, now))
        self.assertIn("recent:params.csv", fv2.scan(self.data, {self.cid: self.info})[self.cid]["reasons"])
        bad = dict(self.info, sizes=dict(self.info["sizes"], **{"segmask.mat": 1}))
        self.assertIn("size:segmask.mat", fv2.scan(self.data, {self.cid: bad}, recent=0)[self.cid]["reasons"])
        missing = fv2.scan(self.data, {"Center009_P000009_Aorta": self.info})["Center009_P000009_Aorta"]
        self.assertFalse(missing["ready"])

    def test_claim_is_exclusive(self):
        first = fv2.claim(self.out, self.cid)
        self.assertIsNotNone(first)
        self.assertIsNone(fv2.claim(self.out, self.cid))
        first.unlink()
        self.assertIsNotNone(fv2.claim(self.out, self.cid))

    def test_shape_mismatch_fails_without_leaving_store(self):
        s = self.sources()["sources"]
        bad = dict(self.info, kdata_shape=[4, 3, 2, 4, 6, 10])
        with self.assertRaises(ValueError):
            fv2.convert_case(self.cid, bad, self.data, self.out, s)
        self.assertFalse(fv2.store_path(self.out, self.cid).exists())
        self.assertEqual(list((self.out / "cases").glob(".*")), [])


if __name__ == "__main__":
    unittest.main()
