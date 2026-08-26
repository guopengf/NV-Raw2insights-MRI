#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from inference import prepare_inference_output_for_save  # noqa: E402
from mri_data.coil_combine import complex_to_ri, to_complex_np  # noqa: E402


def make_normalized_csm(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    csm = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=2, keepdims=True))
    return csm / np.maximum(rss, 1e-8)


def main() -> None:
    rng = np.random.default_rng(11)
    nt, nfe, nc, nspe, npe = 5, 2, 3, 3, 4
    truth = (rng.normal(size=(nt, nfe, nspe, npe)) + 1j * rng.normal(size=(nt, nfe, nspe, npe))).astype(
        np.complex64
    )
    csm = make_normalized_csm(rng, (nt, nfe, nc, nspe, npe))
    coil_images = truth[:, :, None] * csm
    csm_flat = complex_to_ri(csm).reshape(nt * nfe, nc, nspe, npe, 2)
    args = SimpleNamespace(
        dataset="cmrxrecon",
        is_multi_coil=True,
        do_mapping_shuffle=False,
        save_coil_combined_output=True,
    )
    saved = prepare_inference_output_for_save(
        complex_to_ri(coil_images),
        sensitivity_maps=csm_flat,
        args=args,
        final_shape=(1, nt, nfe, nc, nspe, npe),
        temporal_shuffle=np.arange(nt),
    )
    expected = np.transpose(truth, (3, 2, 1, 0))
    actual = to_complex_np(saved)
    max_error = float(np.max(np.abs(actual - expected)))
    print(f"prepare_output_shape={saved.shape} max_error={max_error:.3e}")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    with tempfile.TemporaryDirectory(prefix="raw2ins-merge-") as tmp:
        root = Path(tmp)
        case_rel = Path("Center007") / "GE_30T_Architect" / "P001"
        data_case = root / "data" / case_rel
        recon_case = root / "recon" / case_rel
        data_case.mkdir(parents=True)
        recon_case.mkdir(parents=True)

        with h5py.File(data_case / "kdata_ktGaussian10.mat", "w") as f:
            f.create_dataset("kdata_ktGaussian", shape=(4, nt, nc, nspe, npe, nfe), dtype=np.float32)
        segmask = np.ones((nspe, npe, nfe), dtype=np.uint8)
        segmask[:, 0] = 0
        scipy.io.savemat(data_case / "segmask.mat", {"segmask": segmask})

        for encoding in range(4):
            encoded = saved.copy()
            encoded[..., 0] += encoding
            scipy.io.savemat(
                recon_case / f"kdata_ktGaussian10_enc{encoding}_recon.mat",
                {"img4ranking": encoded},
            )

        out_root = root / "submission"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "tools" / "export_4dflow_submission.py"),
                "--recon-root",
                str(root / "recon"),
                "--data-root",
                str(root / "data"),
                "--out-root",
                str(out_root),
                "--task",
                "TaskR1R2",
                "--split",
                "ValidationSet",
                "--anatomy",
                "Aorta",
                "--recon-layout",
                "yzxt",
            ],
            check=True,
        )
        exported = out_root / "TaskR1R2" / "ValidationSet" / "Aorta" / case_rel / "img_ktGaussian10.npz"
        with np.load(exported) as archive:
            shape = tuple(int(v) for v in archive["shape"])
            coords = archive["coords"]
        print(f"export_shape={shape} sparse_entries={len(coords)}")
        assert shape == (4, nt, nspe, npe, nfe)


if __name__ == "__main__":
    main()
