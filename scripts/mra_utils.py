from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch
from scipy.ndimage import gaussian_filter


def cfg_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def phase3_enabled(args: Any) -> bool:
    return bool(cfg_get(args, "phase3.enable_vaa", False))


def to_complex_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported complex array dtype/shape: dtype={arr.dtype}, shape={arr.shape}")


def read_mat_array(path: str | Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    path = Path(path)
    try:
        with h5py.File(path, "r") as f:
            for key in preferred_keys:
                if key in f:
                    return to_complex_np(f[key][()])
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    return to_complex_np(f[key][()])
    except OSError:
        dat = scipy.io.loadmat(path)
        for key in preferred_keys:
            if key in dat:
                return to_complex_np(dat[key])
        for key, value in dat.items():
            if not key.startswith("__"):
                return to_complex_np(value)
    raise ValueError(f"No array found in {path}")


def temporal_mean_kspace(path: str | Path, key: str, *, device: torch.device, nonzero: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Missing key '{key}' in {path}")
        dset = f[key]
        if len(dset.shape) != 6:
            raise ValueError(f"Expected k-space shape (enc,t,coil,kz,ky,kx), got {dset.shape} in {path}")
        n_enc, nt, nc, nz, ny, nx = dset.shape
        out = torch.zeros((n_enc, nc, nz, ny, nx), dtype=torch.complex64, device=device)
        cnt = torch.zeros((n_enc, nc, nz, ny, nx), dtype=torch.int32, device=device) if nonzero else None
        for t in range(nt):
            frame = torch.as_tensor(to_complex_np(dset[:, t]), dtype=torch.complex64, device=device)
            out += frame
            if nonzero:
                cnt += frame != 0
        if nonzero:
            mask = cnt > 0
            out = torch.where(mask, out / cnt.clamp_min(1), torch.zeros_like(out))
        else:
            mask = torch.ones_like(out, dtype=torch.bool)
            out /= nt
    return out, mask


def k2i_torch(x: torch.Tensor, dim: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def i2k_torch(x: torch.Tensor, dim: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def direct_recon(kmean: torch.Tensor, coilmap: torch.Tensor) -> torch.Tensor:
    imgc = k2i_torch(kmean, dim=(-1, -2, -3))
    num = torch.sum(imgc * torch.conj(coilmap).unsqueeze(0), dim=1)
    den = torch.sum(torch.abs(coilmap) ** 2, dim=0) + 1e-8
    return (num / den.unsqueeze(0)).to(torch.complex64)


def sense_recon(kdata: torch.Tensor, mask: torch.Tensor, coilmap: torch.Tensor, *, lam: float, niter: int) -> torch.Tensor:
    def A(x: torch.Tensor) -> torch.Tensor:
        return mask * i2k_torch(coilmap * x.unsqueeze(0), dim=(-1, -2, -3))

    def AH(y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.conj(coilmap) * k2i_torch(mask * y, dim=(-1, -2, -3)), dim=0)

    b = AH(kdata)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.real(torch.vdot(r.reshape(-1), r.reshape(-1)))
    for _ in range(niter):
        Ap = AH(A(p)) + lam * p
        alpha = rs / torch.real(torch.vdot(p.reshape(-1), Ap.reshape(-1))).clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.real(torch.vdot(r.reshape(-1), r.reshape(-1)))
        if rs_new < 1e-12:
            break
        p = r + (rs_new / rs.clamp_min(1e-12)) * p
        rs = rs_new
    return x.to(torch.complex64)


def pcmra_from_img(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] < 4:
        raise ValueError(f"Expected enc dimension >=4 for reference + RL/AP/HF, got {img.shape}")
    mag = torch.mean(torch.abs(img), dim=0)
    phase_diff = torch.angle(img[1:4] * torch.conj(img[0:1]))
    flow = torch.sqrt(torch.sum(phase_diff**2, dim=0))
    return (mag * flow).float()


def make_vessel_map_2d(pc_mra: np.ndarray, args: Any) -> np.ndarray:
    # Current 4D Flow path treats raw x as slice. MIP over x leaves a zy prior.
    projection_axis = int(cfg_get(args, "phase3.mra.projection_axis", 2))
    mra_mip = np.max(pc_mra.astype(np.float32), axis=projection_axis)
    lower = float(cfg_get(args, "phase3.mra.vessel_map.lower_percentile", 1.0))
    upper = float(cfg_get(args, "phase3.mra.vessel_map.upper_percentile", 99.5))
    smooth_sigma = float(cfg_get(args, "phase3.mra.vessel_map.smooth_sigma", 0.75))
    threshold = cfg_get(args, "phase3.mra.vessel_map.threshold", 0.35)
    binary = bool(cfg_get(args, "phase3.mra.vessel_map.binary", True))

    lo = float(np.percentile(mra_mip, lower))
    hi = float(np.percentile(mra_mip, upper))
    vessel = np.clip((mra_mip - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    if smooth_sigma > 0:
        vessel = gaussian_filter(vessel, sigma=smooth_sigma)
        vessel = np.clip(vessel, 0.0, 1.0)
    if binary and threshold is not None:
        vessel = (vessel >= float(threshold)).astype(np.float32)
    return vessel[None].astype(np.float32)


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def mra_cache_path(args: Any, json_data: dict[str, Any], source: str) -> Path:
    cache_dir = Path(cfg_get(args, "phase3.mra.cache_dir", "/SSDHome/share/haosen/4dflow/mra_cache"))
    full_kspace = json_data.get("target_kspace", json_data.get("full_kspace", json_data.get("gt_kspace", "")))
    kspace = json_data.get("kspace", "")
    acc = json_data.get("mask_type", "")
    patient = Path(full_kspace or kspace).parent
    source_kspace = "" if source == "gt" else str(kspace)
    source_mask_type = "" if source == "gt" else str(acc)
    key_payload = {
        "patient": str(patient),
        "source": source,
        "kspace": source_kspace,
        "full_kspace": str(full_kspace),
        "mask_type": source_mask_type,
        "projection_axis": cfg_get(args, "phase3.mra.projection_axis", 2),
        "vessel_map": cfg_get(args, "phase3.mra.vessel_map", {}),
        "sense": cfg_get(args, "phase3.mra.sense", {}),
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    acc_part = _sanitize(str(acc)) if source != "gt" else "case"
    return cache_dir / f"{_sanitize('__'.join(patient.parts[-3:]))}__{source}__{acc_part}__{digest}.npz"


def generate_or_load_mra_prior(args: Any, json_data: dict[str, Any]) -> np.ndarray | None:
    if not phase3_enabled(args):
        return None

    source = str(cfg_get(args, "phase3.mra.source", "gt")).lower()
    if source == "phase2":
        raise NotImplementedError("phase3.mra.source=phase2 is reserved as an interface placeholder in this implementation.")
    if source not in {"gt", "zf", "sense"}:
        raise ValueError(f"Unsupported phase3.mra.source: {source}")

    cache_path = mra_cache_path(args, json_data, "zf" if source == "sense" else source)
    use_cache = bool(cfg_get(args, "phase3.mra.use_cache", True))
    if use_cache and cache_path.exists():
        return np.load(cache_path)["vessel_map"].astype(np.float32)

    device = torch.device(str(cfg_get(args, "phase3.mra.device", "cpu")))
    coilmap_path = json_data.get("coilmap")
    if not coilmap_path:
        raise ValueError("VAA MRA generation requires coilmap in the 4D Flow JSON.")
    coilmap = torch.as_tensor(read_mat_array(coilmap_path, ("coilmap", "csm", "sensitivity_maps", "sens_maps")), device=device)

    if source == "gt":
        full_kspace = json_data.get("target_kspace", json_data.get("full_kspace", json_data.get("gt_kspace")))
        kmean, _ = temporal_mean_kspace(full_kspace, "kdata_full", device=device, nonzero=False)
        img = direct_recon(kmean, coilmap)
        prior_from = "pc_mra_full_ifft"
    else:
        kspace = json_data["kspace"]
        kmean, mask = temporal_mean_kspace(kspace, "kdata_ktGaussian", device=device, nonzero=True)
        sense_niter = int(cfg_get(args, "phase3.mra.sense.niter", 5))
        sense_lam = float(cfg_get(args, "phase3.mra.sense.lam", 1e-4))
        img = torch.stack(
            [sense_recon(kmean[enc], mask[enc], coilmap, lam=sense_lam, niter=sense_niter) for enc in range(kmean.shape[0])],
            dim=0,
        )
        prior_from = "pc_mra_us_sense"

    pc_mra = pcmra_from_img(img).detach().cpu().numpy()
    vessel_map = make_vessel_map_2d(pc_mra, args)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vessel_map=vessel_map, pc_mra_shape=np.array(pc_mra.shape), prior_from=prior_from)
    return vessel_map
