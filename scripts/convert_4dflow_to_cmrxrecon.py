"""Convert CMRx 4D Flow cases to CMRxRecon-compatible JSON + .mat files,
one file set per encoding direction, so stock inference.py works unchanged.

Output layout (absolute paths written into the JSONs so cwd doesn't matter):

  <out_root>/
    <case>_enc<e>_kus_<maskType>.json
    MultiCoil/Flow2d/UnderSample_TaskR1R2/<case>_enc<e>_kus_<maskType>.mat  (key: "kus")
    MultiCoil/Flow2d/Mask_TaskR1R2/<case>_enc<e>_mask_<maskType>.mat        (key: "mask")

For each enc, the stored kspace is hybrid-domain: 1D IFFT along kx has already
been applied so the fully-sampled readout becomes the model's "slice" axis:
    (t, coil, kz, ky, kx) -> 1D IFFT on kx -> (t, coil, kz, ky, x)
    -> rearrange to (t, x, coil, kz, ky) which matches the model's (t, z, c, y, x) layout.

The stored mask is (t, kz, ky). The reader broadcasts it to (t, 1, 1, kz, ky).
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from einops import rearrange


COMPLEX_DTYPE = np.dtype([("real", "<f4"), ("imag", "<f4")])


def _first_key(f: h5py.File) -> str:
    for k in f.keys():
        if not k.startswith("#"):
            return k
    raise KeyError("no data key")


def load_mat(path: Path, key: str | None = None):
    with h5py.File(path, "r", swmr=True) as f:
        k = key if key and key in f else _first_key(f)
        arr = f[k][()]
    if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
        return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    return np.asarray(arr)


def save_complex_mat(path: Path, key: str, arr: np.ndarray) -> None:
    compound = np.empty(arr.shape, dtype=COMPLEX_DTYPE)
    compound["real"] = arr.real.astype(np.float32)
    compound["imag"] = arr.imag.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=compound)


def save_real_mat(path: Path, key: str, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=arr.astype(np.float32))


def ifft1d_kx(kspace: np.ndarray) -> np.ndarray:
    """Centered 1D IFFT along last axis (kx)."""
    s = np.fft.ifftshift(kspace, axes=-1)
    s = np.fft.ifft(s, axis=-1, norm="ortho")
    return np.fft.fftshift(s, axes=-1)


def find_case_files(case_dir: Path) -> dict:
    files = {p.name: p for p in case_dir.iterdir() if p.suffix == ".mat"}
    kdata_us = None
    mask_us = None
    for name, path in files.items():
        if name.startswith("kdata_kt") or name.startswith("kdata_Uniform"):
            kdata_us = path
        if name.startswith("usmask"):
            mask_us = path
    return {
        "kdata_full": files.get("kdata_full.mat"),
        "kdata_us": kdata_us,
        "mask_us": mask_us,
    }


def infer_orig_mask_type(kdata_us_path: Path | None) -> str:
    if kdata_us_path is None:
        return "fully_sampled"
    # e.g. kdata_ktGaussian20.mat -> ktGaussian20
    m = re.match(r"kdata_(.+)\.mat$", kdata_us_path.name)
    return m.group(1) if m else "ktGaussian20"


def convert_case(case_dir: Path, out_root: Path, override_mask_type: str | None = None) -> list[Path]:
    """Write per-enc JSONs for one case. Returns list of JSON paths written."""
    info = find_case_files(case_dir)
    kdata_path = info["kdata_us"] if info["kdata_us"] is not None else info["kdata_full"]
    assert kdata_path is not None, f"no kdata in {case_dir}"
    kspace = load_mat(kdata_path)  # (enc, t, c, kz, ky, kx)
    assert kspace.ndim == 6, f"expected 6D, got {kspace.shape}"
    enc, t, c, kz, ky, kx = kspace.shape

    # 1D IFFT along kx.
    kspace_hy = ifft1d_kx(kspace)  # (enc, t, c, kz, ky, x)

    # Mask.
    if info["mask_us"] is not None:
        mask_raw = load_mat(info["mask_us"]).astype(np.float32)  # (em, t, cm, kz, ky, kxm)
        em = mask_raw.shape[0]
    else:
        mask_raw = np.ones((1, t, 1, kz, ky, 1), dtype=np.float32)
        em = 1

    orig_mask_type = infer_orig_mask_type(info["kdata_us"])
    mask_type = override_mask_type or orig_mask_type
    case_name = "_".join(case_dir.parts[-4:])  # <Task>_<Center>_<Scanner>_<Pid>

    written_jsons: list[Path] = []
    for e in range(enc):
        # Per-enc 5D k-space: (t, x, c, kz, ky) complex — matches model's (t, z, c, y, x).
        ks_e = kspace_hy[e]  # (t, c, kz, ky, x)
        ks_5d = rearrange(ks_e, "t c kz ky x -> t x c kz ky")

        # Per-enc 3D mask: (t, kz, ky) real.
        idx = 0 if em == 1 else e
        m_e = mask_raw[idx, :, 0, :, :, 0]  # (t, kz, ky)

        ks_mat = out_root / "MultiCoil" / "Flow2d" / "UnderSample_TaskR1R2" / (
            f"{case_name}_enc{e}_kus_{mask_type}.mat"
        )
        mask_mat = out_root / "MultiCoil" / "Flow2d" / "Mask_TaskR1R2" / (
            f"{case_name}_enc{e}_mask_{mask_type}.mat"
        )
        save_complex_mat(ks_mat, "kus", ks_5d)
        save_real_mat(mask_mat, "mask", m_e)

        json_path = out_root / f"{case_name}_enc{e}_kus_{mask_type}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "kspace": str(ks_mat.resolve()),
                    "mask": [str(mask_mat.resolve())],
                },
                f,
            )
        written_jsons.append(json_path)
        print(
            f"  enc{e}: wrote kus shape={ks_5d.shape} {ks_5d.dtype} -> {ks_mat}\n"
            f"        mask shape={m_e.shape} -> {mask_mat}\n"
            f"        json -> {json_path}"
        )
    return written_jsons


def iter_cases(root: Path):
    for p in sorted(root.rglob("kdata_full.mat")):
        yield p.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_root", type=Path, required=True,
                   help="Root of 4D flow data (e.g. .../TaskR1R2_demo/ValidationSet)")
    p.add_argument("-o", "--output_root", type=Path, required=True,
                   help="Destination root for CMRxRecon-format files.")
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--mask_type_override", type=str, default=None,
                   help="If set, replace the acc number in the filename (e.g. 'ktGaussian24'). "
                        "Useful to steer the model into a trained acceleration bucket.")
    return p.parse_args()


def main():
    a = parse_args()
    a.output_root.mkdir(parents=True, exist_ok=True)
    cases = list(iter_cases(a.input_root))
    if a.max_cases > 0:
        cases = cases[: a.max_cases]
    print(f"converting {len(cases)} case(s) -> {a.output_root}")
    for c in cases:
        print(f"--- {c}")
        convert_case(c, a.output_root, override_mask_type=a.mask_type_override)
    print("done.")


if __name__ == "__main__":
    main()
