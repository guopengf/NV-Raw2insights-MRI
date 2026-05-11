"""Verify that inference_4dflow.py's load+reshape+IFFT path produces the same
zero-filled recon as a straightforward 3D-IFFT on the raw kdata_us.

If they match (within numerical precision), the loading is correct and the
remaining quality loss is purely due to the model/mask/aspect-ratio, not a bug.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path("/workspace/code/NV-Raw2insights-MRI-fork/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from monai.apps.reconstruction.complex_utils import complex_abs
from monai.data.fft_utils import ifftn_centered

from inference_4dflow import (  # type: ignore
    find_case_files, load_mat_complex, load_mat_real,
    ifft1d_kx, reshape_one_enc, reshape_mask_one_enc,
)
from einops import rearrange


def pipeline_zf(case_dir: Path):
    """Reproduce exactly what inference_4dflow.py does up through the zero-filled
    image (i.e. no model call). Returns RSS per (enc, t, x, kz, ky)."""
    info = find_case_files(case_dir)
    kdata_path = info["kdata_us"] if info["kdata_us"] else info["kdata_full"]
    mask_path = info["mask_us"]
    print(f"  kdata path: {kdata_path.name}")
    print(f"  mask  path: {mask_path.name if mask_path else '(none)'}")
    kspace = load_mat_complex(kdata_path)
    enc, t, c, kz, ky, kx = kspace.shape
    print(f"  loaded kspace shape {kspace.shape} dtype={kspace.dtype}")

    mask_raw = load_mat_real(mask_path).astype(np.float32) if mask_path else \
        np.ones((1, t, 1, kz, ky, 1), dtype=np.float32)
    print(f"  mask shape {mask_raw.shape} unique={np.unique(mask_raw)}")

    # Mirror inference_4dflow.py exactly.
    kspace_hy = ifft1d_kx(kspace)  # (enc, t, c, kz, ky, x)

    zf_per_enc = []
    for e in range(enc):
        ks_one = kspace_hy[e]  # (t, c, kz, ky, x)
        ms_one = reshape_mask_one_enc(mask_raw, e)  # (t, 1, kz, ky)
        # The run_inference_one_case zf block:
        ks_ri = np.stack((ks_one.real, ks_one.imag), axis=-1).astype(np.float32)
        img_zf = complex_abs(
            ifftn_centered(
                torch.from_numpy(rearrange(ks_ri, "t c kz ky x ri -> t x c kz ky ri")),
                spatial_dims=2, is_complex=True,
            )
        )  # (t, x, c, kz, ky)
        zf_per_enc.append(
            torch.sqrt(torch.sum(img_zf ** 2, dim=2)).numpy()  # (t, x, kz, ky)
        )
    return np.stack(zf_per_enc, axis=0), mask_raw, kspace


def reference_zf(kspace: np.ndarray):
    """Straightforward 3D IFFT on (kz, ky, kx) + RSS. Shape (enc, t, z, y, x)."""
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=(-3, -2, -1)),
                     axes=(-3, -2, -1), norm="ortho"),
        axes=(-3, -2, -1),
    )
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=2))  # (enc, t, z, y, x)


def main():
    case = Path(sys.argv[1])
    print(f"case = {case}")
    pipeline, mask_raw, kspace_loaded = pipeline_zf(case)
    print(f"\npipeline zf shape: {pipeline.shape}")
    ref = reference_zf(kspace_loaded)
    print(f"reference zf shape: {ref.shape}")

    # pipeline: (enc, t, x, kz, ky); reference: (enc, t, z=kz, y=ky, x)
    # Reorder pipeline to (enc, t, kz, ky, x) to match reference (enc, t, z, y, x).
    pipe_ordered = np.transpose(pipeline, (0, 1, 3, 4, 2))
    print(f"pipeline re-ordered to (enc, t, kz, ky, x): {pipe_ordered.shape}")
    print(f"reference               (enc, t, kz, ky, x): {ref.shape}")

    assert pipe_ordered.shape == ref.shape

    diff = np.abs(pipe_ordered - ref)
    rel = diff / (np.abs(ref) + 1e-12)
    print(f"\n|pipeline - reference|:  max={diff.max():.4g}  mean={diff.mean():.4g}")
    print(f"reference:                max={ref.max():.4g}  mean={ref.mean():.4g}")
    print(f"relative diff:             max={rel.max():.4g}  mean={rel.mean():.4g}")

    per_enc_corr = []
    for e in range(pipe_ordered.shape[0]):
        p = pipe_ordered[e].ravel()
        r = ref[e].ravel()
        p_n = (p - p.mean()) / (p.std() + 1e-9)
        r_n = (r - r.mean()) / (r.std() + 1e-9)
        per_enc_corr.append(float((p_n * r_n).mean()))
    print(f"\npearson per enc: {per_enc_corr}")

    if diff.max() < 1e-4 * ref.max():
        print("\nVERDICT: loading+IFFT path is equivalent to a direct 3D IFFT."
              " Loading is CORRECT.")
    else:
        print("\nVERDICT: non-trivial mismatch. LOADING OR IFFT HAS AN ERROR.")


if __name__ == "__main__":
    main()
