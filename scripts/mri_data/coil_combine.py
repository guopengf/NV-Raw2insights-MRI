from __future__ import annotations

import numpy as np


def to_complex_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    return arr


def complex_to_ri(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if not np.iscomplexobj(arr):
        raise ValueError(f"Expected complex array, got dtype={arr.dtype}")
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32, copy=False)


def sensitivity_weighted_coil_combine(
    image: np.ndarray,
    sensitivity_maps: np.ndarray,
    *,
    coil_axis: int,
    eps: float = 1e-8,
) -> np.ndarray:
    image = to_complex_np(image)
    sensitivity_maps = to_complex_np(sensitivity_maps)
    if not np.iscomplexobj(image) or not np.iscomplexobj(sensitivity_maps):
        raise ValueError("Image and sensitivity maps must contain complex data")

    image_axis = coil_axis % image.ndim
    sensitivity_axis = coil_axis % sensitivity_maps.ndim
    if image.shape[image_axis] != sensitivity_maps.shape[sensitivity_axis]:
        raise ValueError(
            "Coil count mismatch: "
            f"image shape={image.shape}, sensitivity shape={sensitivity_maps.shape}, coil_axis={coil_axis}"
        )

    try:
        np.broadcast_shapes(image.shape, sensitivity_maps.shape)
    except ValueError as exc:
        raise ValueError(
            f"Image shape {image.shape} and sensitivity shape {sensitivity_maps.shape} are not broadcast-compatible"
        ) from exc

    numerator = np.sum(image * np.conj(sensitivity_maps), axis=image_axis)
    denominator = np.sum(np.abs(sensitivity_maps) ** 2, axis=sensitivity_axis) + eps
    return (numerator / denominator).astype(np.complex64, copy=False)


def combine_tschw_ri_to_tshw_ri(
    image_tschw_ri: np.ndarray,
    sensitivity_tschw_ri: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    image_tschw_ri = np.asarray(image_tschw_ri)
    sensitivity_tschw_ri = np.asarray(sensitivity_tschw_ri)
    if image_tschw_ri.ndim != 6 or image_tschw_ri.shape[-1] != 2:
        raise ValueError(f"Expected image shape (t,s,c,h,w,2), got {image_tschw_ri.shape}")
    if sensitivity_tschw_ri.shape != image_tschw_ri.shape:
        raise ValueError(
            f"Sensitivity shape must match image shape: {sensitivity_tschw_ri.shape} vs {image_tschw_ri.shape}"
        )

    combined = sensitivity_weighted_coil_combine(
        image_tschw_ri,
        sensitivity_tschw_ri,
        coil_axis=2,
        eps=eps,
    )
    return complex_to_ri(combined)


def combine_yzxtc_to_yzxt(
    image_yzxtc: np.ndarray,
    coilmap_czyx: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    image_yzxtc = to_complex_np(image_yzxtc)
    coilmap_czyx = to_complex_np(coilmap_czyx)
    if image_yzxtc.ndim != 5:
        raise ValueError(f"Expected image shape (y,z,x,t,c), got {image_yzxtc.shape}")

    pe, spe, fe, _nt, nc = image_yzxtc.shape
    if coilmap_czyx.shape != (nc, spe, pe, fe):
        raise ValueError(
            f"Coilmap shape mismatch: got {coilmap_czyx.shape}, expected {(nc, spe, pe, fe)} "
            f"for image shape {image_yzxtc.shape}"
        )

    image_tzyxc = np.transpose(image_yzxtc, (3, 1, 0, 2, 4))
    coilmap_zyxc = np.transpose(coilmap_czyx, (1, 2, 3, 0))
    combined_tzyx = sensitivity_weighted_coil_combine(
        image_tzyxc,
        coilmap_zyxc[None, ...],
        coil_axis=-1,
        eps=eps,
    )
    return np.transpose(combined_tzyx, (2, 1, 3, 0)).astype(np.complex64, copy=False)
