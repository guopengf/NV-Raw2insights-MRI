#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from path_safety import assert_outputs_not_in_data


def read_mat_dict(path: Path) -> dict:
    try:
        with h5py.File(path, "r", swmr=True) as f:
            return {key: f[key][()] for key in f}
    except Exception:
        return {key: val for key, val in scipy.io.loadmat(path).items() if not key.startswith("__")}


def first_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    dat = read_mat_dict(path)
    for key in preferred_keys:
        if key in dat:
            return dat[key]
    for val in dat.values():
        return val
    raise ValueError(f"No array found in {path}")


def to_complex(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return arr["real"] + 1j * arr["imag"]
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(np.complex64)


def ifft1c(x: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(x, axes=axis), axis=axis, norm="ortho"),
        axes=axis,
    )


def ifft2c(x: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(x, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def prepare_coilmap(
    coilmap: np.ndarray,
    *,
    nt: int,
    nx: int,
    nc: int,
    enc_idx: int,
    n_enc: int,
    axis_order: str,
    normalize: bool,
) -> np.ndarray:
    csm = np.squeeze(to_complex(coilmap))

    if csm.ndim == 6:
        csm = csm[enc_idx if csm.shape[0] == n_enc else 0]
        if csm.shape[0] != nt:
            csm = np.repeat(csm[:1], nt, axis=0)
        csm = np.transpose(csm, (0, 4, 1, 2, 3))
    elif csm.ndim == 5:
        if csm.shape[0] == n_enc:
            csm = np.transpose(csm[enc_idx], (3, 0, 1, 2))
            csm = np.repeat(csm[None, ...], nt, axis=0)
        elif csm.shape[0] == nt:
            csm = np.transpose(csm, (0, 4, 1, 2, 3))
        else:
            raise ValueError(f"Unsupported 5D coilmap shape: {csm.shape}")
    elif csm.ndim == 4:
        if axis_order == "auto":
            axis_order = "coil_kz_ky_kx" if csm.shape[0] == nc else "kz_ky_kx_coil"
        if axis_order == "coil_kz_ky_kx":
            csm = np.transpose(csm, (3, 0, 1, 2))
        elif axis_order == "kz_ky_kx_coil":
            csm = np.transpose(csm, (2, 3, 0, 1))
        elif axis_order == "kx_ky_kz_coil":
            csm = np.transpose(csm, (0, 3, 2, 1))
        elif axis_order == "kx_kz_ky_coil":
            csm = np.transpose(csm, (0, 3, 1, 2))
        elif axis_order == "coil_x_kz_ky":
            csm = np.transpose(csm, (1, 0, 2, 3))
        else:
            raise ValueError(f"Unsupported axis order: {axis_order}")
        csm = np.repeat(csm[None, ...], nt, axis=0)
    else:
        raise ValueError(f"Unsupported coilmap shape: {csm.shape}")

    if csm.shape[1] != nx or csm.shape[2] != nc:
        raise ValueError(f"Converted coilmap shape {csm.shape} does not match nx={nx}, nc={nc}")

    if normalize:
        rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=2, keepdims=True))
        csm = csm / np.maximum(rss, 1e-8)
    return csm


def robust_mag(x: np.ndarray) -> np.ndarray:
    x = np.abs(x).astype(np.float32)
    hi = np.percentile(x, 99.5)
    return np.clip(x / max(hi, 1e-8), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient-dir",
        type=Path,
        default=Path("/SSDHome/share/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P076/"),
    )
    parser.add_argument("--enc", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--x-slice", type=int, default=None)
    parser.add_argument("--axis-order", default="auto")
    parser.add_argument("--no-normalize-coilmap", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/coilmap_checks"),
        help="Output PNG path or directory. Defaults to a workspace-local directory, never the patient data folder.",
    )
    args = parser.parse_args()

    kspace = to_complex(first_array(args.patient_dir / "kdata_full.mat", ("kdata_full", "kdata", "kspace_full")))
    coilmap = first_array(args.patient_dir / "coilmap.mat", ("coilmap", "csm", "sensitivity_maps", "sens_maps"))

    if kspace.ndim != 6:
        raise ValueError(f"Expected k-space shape (enc,t,coil,kz,ky,kx), got {kspace.shape}")

    n_enc, nt, nc, nkz, nky, nkx = kspace.shape
    enc_idx = args.enc
    frame_idx = args.frame
    x_idx = args.x_slice if args.x_slice is not None else nkx // 2

    hybrid = ifft1c(kspace[enc_idx : enc_idx + 1], axis=-1)
    hybrid = np.transpose(hybrid, (0, 1, 5, 2, 3, 4))[0]
    coil_img = ifft2c(hybrid, axes=(-2, -1))

    csm = prepare_coilmap(
        coilmap,
        nt=nt,
        nx=nkx,
        nc=nc,
        enc_idx=enc_idx,
        n_enc=n_enc,
        axis_order=args.axis_order,
        normalize=not args.no_normalize_coilmap,
    )

    img = coil_img[frame_idx, x_idx]
    maps = csm[frame_idx, x_idx]
    rss = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
    sense = np.sum(img * np.conj(maps), axis=0)
    denom = np.sum(np.abs(maps) ** 2, axis=0)
    sense = sense / np.maximum(denom, 1e-8)

    out = assert_outputs_not_in_data([args.out], [args.patient_dir])[0]
    if out.suffix.lower() != ".png":
        out.mkdir(parents=True, exist_ok=True)
        out = out / f"{args.patient_dir.name}_coilmap_check_enc{enc_idx}_t{frame_idx}_x{x_idx}.png"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(robust_mag(rss), cmap="gray")
    axes[0, 0].set_title("RSS")
    axes[0, 1].imshow(robust_mag(sense), cmap="gray")
    axes[0, 1].set_title("SENSE magnitude")
    axes[0, 2].imshow(np.angle(sense), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("SENSE phase")
    axes[1, 0].imshow(robust_mag(maps[0]), cmap="gray")
    axes[1, 0].set_title("Coilmap coil 0 magnitude")
    axes[1, 1].imshow(np.angle(maps[0]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title("Coilmap coil 0 phase")
    axes[1, 2].imshow(robust_mag(np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))), cmap="gray")
    axes[1, 2].set_title("Coilmap RSS")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{args.patient_dir.name}: enc={enc_idx}, frame={frame_idx}, x={x_idx}")
    fig.savefig(out, dpi=180)
    print(f"kspace shape: {kspace.shape}")
    print(f"coilmap raw shape: {np.asarray(coilmap).shape}")
    print(f"coilmap converted shape: {csm.shape}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
