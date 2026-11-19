"""The converted-HDF5 reader yields the same training windows as the raw .mat reader."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/tools")]

import final_v2_h5 as fv2  # noqa: E402

from .common.data import TrainSlabs, h5_root  # noqa: E402

SHAPE = [4, 5, 2, 6, 8, 8]  # E,T,C,Z,Y,X; X even (all real cases are; odd X differs by an fft shift)
CID = "Center001_P000001_Aorta"
COMPLEX = np.dtype([("real", "<f4"), ("imag", "<f4")])


class H5BackendTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        src = root / "data" / CID
        src.mkdir(parents=True)
        rng = np.random.default_rng(0)
        e, t, c, z, y, x = SHAPE

        def cplx(shape):
            out = np.empty(shape, COMPLEX)
            out["real"], out["imag"] = rng.normal(size=shape), rng.normal(size=shape)
            return out

        for name, key, array in (("kdata_full.mat", "kdata_full", cplx(SHAPE)), ("coilmap.mat", "coilmap", cplx((c, z, y, x))),
                                 ("segmask.mat", "segmask", (rng.random((z, y, x)) > 0.5).astype(np.float32))):
            with h5py.File(src / name, "w") as f:
                f[key] = array
        matrix = ";".join(map(str, SHAPE))
        (src / "params.csv").write_text(f"VENC,venc_order,spatial_order,matrix_size\n100;150;200,FH;AP;RL,FH;AP;RL,{matrix}\n")
        info = {"sizes": {n: (src / n).stat().st_size for n in fv2.REQUIRED}, "kdata_shape": SHAPE}
        sources = fv2.scan(root / "data", {CID: info}, recent=0)[CID]["sources"]
        self.out = root / "out"
        self.assertEqual(fv2.convert_case(CID, info, root / "data", self.out, sources)[0], "converted")
        self.index = root / "index.json"
        self.index.write_text(json.dumps([{"case_id": CID, "dataset_group": "train",
                                           "files": {n: str(src / n) for n in fv2.REQUIRED},
                                           "rows": [{"target_file": f"/{CID}/kdata_full.mat", "size_bytes": 1}]}]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_windows_match_raw_reader(self):
        for slices, frames in ((3, 5), (1, 2), (3, 7)):
            kw = dict(num_slices=slices, num_frames=frames, samples_per_case=12, seed=7)
            raw, h5 = TrainSlabs(self.index, **kw), TrainSlabs(self.index, h5_root=self.out, **kw)
            for i in range(len(raw)):
                a, b = raw[i], h5[i]
                for key in ("full_kspace", "kspace", "coils", "mask", "target", "segmask", "scale", "venc",
                            "frame_indices", "slice_indices"):
                    torch.testing.assert_close(b[key], a[key], rtol=1e-5, atol=1e-5, msg=lambda m, key=key: f"{key}: {m}")

    def test_backend_switch(self):
        self.assertIsNone(h5_root({}))
        self.assertEqual(h5_root({"train_backend": "windowed_hdf5", "h5_root": "/x"}), "/x")
        with self.assertRaises(ValueError):
            h5_root({"train_backend": "windowed_hdf5"})


if __name__ == "__main__":
    unittest.main()
