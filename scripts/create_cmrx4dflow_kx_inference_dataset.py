"""Physics-faithful CMRx 4D-Flow conversion for stock inference.

Difference vs `create_cmrx4dflow_inference_dataset.py`:
  - Uses the physically correct FE axis: 1D IFFT along **kx** (the fully-sampled
    readout), so each (t, x) slice is a proper 2D k-space of shape (kz, ky).
  - Retrospectively undersamples with a fresh **ktGaussian R=24** mask on the
    (kz, ky) plane, matching one of the accelerations the model was trained on.
    kz is small (~19), so the kt-Gaussian pattern is applied along ky (undersampled)
    with kz tiled (fully sampled) — this matches the trained 1D-PE kt undersampling
    family while keeping the physical FE axis correct.

Output layout mirrors the colleague's script:

  <out_root>/
    <case>_enc<e>_kus_ktGaussian<R>.json
    MultiCoil/Flow2d/UnderSample_TaskR1/<case>_enc<e>_kus_ktGaussian<R>.mat  (kus)
    MultiCoil/Flow2d/Mask_TaskR1/<case>_enc<e>_mask_ktGaussian<R>.mat        (mask)

The stored kspace per-enc has shape (t, x, c, kz, ky) — matches the model's
(t, z, c, y, x) convention where x plays the role of the slice axis.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mri_data.ktSampling import kt_gaussian_sampling  # noqa: E402


DEFAULT_ACS_LINES = 20
DEFAULT_ALPHA = 0.2
DEFAULT_ACCELERATION = 24


def centered_ifft_1d(data: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(data, axes=axis), axis=axis, norm="ortho"),
        axes=axis,
    )


def to_mat_complex(data: np.ndarray) -> np.ndarray:
    mat_dtype = np.dtype([("real", np.float32), ("imag", np.float32)])
    mat_data = np.empty(data.shape, dtype=mat_dtype)
    mat_data["real"] = np.asarray(data.real, dtype=np.float32)
    mat_data["imag"] = np.asarray(data.imag, dtype=np.float32)
    return mat_data


def save_hdf5_mat(path: Path, key: str, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data)


def read_kdata_full(case_dir: Path) -> np.ndarray:
    kspace_file = case_dir / "kdata_full.mat"
    with h5py.File(kspace_file, "r") as f:
        if "kdata_full" not in f:
            raise KeyError(f"Expected 'kdata_full' in {kspace_file}, found: {list(f.keys())}")
        kdata = f["kdata_full"][()]
    if kdata.dtype.fields is None or {"real", "imag"} - set(kdata.dtype.fields):
        raise TypeError(f"Expected compound real/imag in {kspace_file}, got {kdata.dtype}")
    return np.asarray(kdata["real"] + 1j * kdata["imag"], dtype=np.complex64)


def read_case_metadata(case_dir: Path) -> dict:
    metadata = {"case_dir": str(case_dir.resolve())}
    params_file = case_dir / "params.csv"
    if not params_file.exists():
        return metadata
    df = pd.read_csv(params_file)
    if df.empty:
        return metadata
    record = df.iloc[0].to_dict()
    metadata.update({k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()})
    return metadata


def create_retrospective_mask_kzky(nt: int, nkz: int, nky: int, R: int, acs_lines: int,
                                    alpha: float, seed: int) -> np.ndarray:
    """kt-Gaussian undersampling on the (kz, ky) plane.

    kt_gaussian_sampling(nx, ny, nt) returns shape (nx, ny, nt) where ny is the
    kt-Gaussian-undersampled axis and nx is tiled (fully sampled). We pick
    nx=nkz=19 (tiled) and ny=nky=112 (undersampled) since nkz is too small to
    meaningfully sub-sample with ACS=20 > nkz. Then transpose axes so the mask
    indexes as (t, kz, ky).
    """
    mask = kt_gaussian_sampling(nx=nkz, ny=nky, nt=nt,
                                ncalib=acs_lines, R=R, alpha=alpha, seed=seed)
    # (nkz, nky, nt) -> (nt, nkz, nky)
    return np.transpose(mask, (2, 0, 1)).astype(np.float32)


def get_case_prefix(case_dir: Path) -> str:
    return "_".join(case_dir.parts[-3:])


def convert_case(case_dir: Path, output_dir: Path, acceleration: int, acs_lines: int,
                 alpha: float, seed: int) -> None:
    kdata_full = read_kdata_full(case_dir)
    if kdata_full.ndim != 6:
        raise ValueError(f"Expected 6D [venc, t, c, kz, ky, kx], got {kdata_full.shape}")
    num_venc, nt, num_coils, nkz, nky, nkx = kdata_full.shape
    case_prefix = get_case_prefix(case_dir)
    metadata = read_case_metadata(case_dir)

    # 1D IFFT along kx (axis=-1), the physically-fully-sampled FE direction.
    hybrid_all = centered_ifft_1d(kdata_full, axis=-1)  # (enc, t, c, kz, ky, x)

    undersample_dir = output_dir / "MultiCoil" / "Flow2d" / "UnderSample_TaskR1"
    mask_dir = output_dir / "MultiCoil" / "Flow2d" / "Mask_TaskR1"
    undersample_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_case_dir": str(case_dir.resolve()),
        "acquisition": "Flow2d",
        "acceleration": acceleration,
        "mask_type": f"ktGaussian{acceleration}",
        "ifft_axis": "kx (FE, physically fully-sampled)",
        "shape_in": [int(v) for v in kdata_full.shape],
        "shape_out_per_case": [int(nt), int(nkx), int(num_coils), int(nkz), int(nky)],
        "metadata": metadata,
        "cases": [],
    }

    for e in range(num_venc):
        # (t, c, kz, ky, x) -> (t, x, c, kz, ky) so model sees 2D (kz, ky) k-space per (t, x).
        hy = hybrid_all[e]  # (t, c, kz, ky, x)
        hy = np.transpose(hy, (0, 4, 1, 2, 3))  # (t, x, c, kz, ky)

        mask = create_retrospective_mask_kzky(
            nt=nt, nkz=nkz, nky=nky, R=acceleration,
            acs_lines=acs_lines, alpha=alpha, seed=seed + e,
        )  # (t, kz, ky)
        masked_kspace = hy * mask[:, None, None, :, :]

        base_name = f"{case_prefix}_enc{e}"
        kspace_file = undersample_dir / f"{base_name}_kus_ktGaussian{acceleration}.mat"
        mask_file = mask_dir / f"{base_name}_mask_ktGaussian{acceleration}.mat"
        json_file = output_dir / f"{base_name}_kus_ktGaussian{acceleration}.json"

        save_hdf5_mat(kspace_file, "kus", to_mat_complex(masked_kspace))
        save_hdf5_mat(mask_file, "mask", mask)

        json_file.write_text(json.dumps(
            {"kspace": str(kspace_file.resolve()),
             "mask": [str(mask_file.resolve())]},
            indent=4))

        manifest["cases"].append({
            "venc_index": e,
            "json": str(json_file.resolve()),
            "kspace": str(kspace_file.resolve()),
            "mask": str(mask_file.resolve()),
            "shape": [int(nt), int(nkx), int(num_coils), int(nkz), int(nky)],
        })

    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "cmrx4dflow_kx_manifest.json").write_text(json.dumps(manifest, indent=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--acceleration", type=int, default=DEFAULT_ACCELERATION)
    p.add_argument("--acs_lines", type=int, default=DEFAULT_ACS_LINES)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    src = a.source_dir.resolve()
    out = a.output_dir.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    convert_case(src, out, a.acceleration, a.acs_lines, a.alpha, a.seed)
    print(f"Converted {src} -> {out} (kx-IFFT, physics-faithful)")


if __name__ == "__main__":
    main()
