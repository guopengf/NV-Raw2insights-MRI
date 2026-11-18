"""Read real training slabs; IFFT raw kx, preserve encoding indices and zy sampling."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, closing
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import fft
import torch
from torch.utils.data import Dataset

from .manifest import read_params
from .online_sampling import online_sampler
from .safety import ROOT


def as_complex(array):
    if array.dtype.fields and {"real", "imag"}.issubset(array.dtype.fields):
        return (array["real"] + 1j * array["imag"]).astype(np.complex64)
    if np.iscomplexobj(array):
        return array.astype(np.complex64, copy=False)
    raise ValueError(f"Expected complex array, got {array.dtype}")


def ifftc(array, axes):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array, axes=axes), axes=axes, norm="ortho"), axes=axes).astype(np.complex64)


def fft2(array, inverse=False):
    fn = torch.fft.ifft2 if inverse else torch.fft.fft2
    return torch.fft.fftshift(fn(torch.fft.ifftshift(array, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def sense_forward(image, coils, mask):
    # image [B,E,T,S,Z,Y], coils [B,C,S,Z,Y], mask [B,T,Z,Y].
    return fft2(image[:, :, :, None] * coils[:, None, None]) * mask[:, None, :, None, None]


def sense_adjoint(kspace, coils, mask):
    image = fft2(kspace * mask[:, None, :, None, None], inverse=True)
    return (image * coils[:, None, None].conj()).sum(dim=3)


def kt_mask(z, y, t, acceleration, seed):
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_4dflow_ktgaussian_train import generate_ktgaussian_mask_2d

    mask = generate_ktgaussian_mask_2d(z, y, t, acceleration, np.random.default_rng(seed))
    return torch.from_numpy(mask.transpose(2, 0, 1).copy())


class TrainSlabs(Dataset):
    """Each item contains all 4 encodings; individual-encoding models split them later."""

    def __init__(self, index, *, seed=2026, num_slices=3, num_frames=5, samples_per_case=8,
                 accelerations=(10, 20, 30, 40, 50), smoke=False, smoke_case_id=None,
                 online_undersampling=None):
        self.cases = json.loads(Path(index).read_text())
        if not self.cases or any(c["dataset_group"] != "train" for c in self.cases):
            raise ValueError("TrainSlabs accepts only an explicit, nonempty train index")
        self.cases.sort(key=lambda c: next(int(r["size_bytes"]) for r in c["rows"] if r["target_file"].endswith("/kdata_full.mat")))
        if smoke:
            self.cases = [case for case in self.cases if case["case_id"] == smoke_case_id] if smoke_case_id else self.cases[:1]
            if not self.cases:
                raise ValueError(f"Smoke case {smoke_case_id!r} is not in the Train index")
        self.smoke = smoke
        self.seed, self.num_slices, self.num_frames = seed, num_slices, num_frames
        self.samples_per_case, self.accelerations, self.epoch = samples_per_case, list(accelerations), 0
        self.online_sampler = online_sampler(online_undersampling, self.seed)
        if num_slices < 1:
            raise ValueError("num_slices must be positive (our ablations use odd S=3; FlowMoDL uses S=64)")

    def __len__(self):
        return len(self.cases) * self.samples_per_case

    def _window(self, index, shape, anatomical_mask):
        _, nt, _, _, _, nx = shape
        sample_seed = self.seed + self.epoch * len(self) + index
        rng = np.random.default_rng(sample_seed)
        center_x, center_t = int(rng.integers(nx)), int(rng.integers(nt))
        if self.smoke:
            center_x = int(np.argmax(anatomical_mask.sum(axis=(0, 1))))
        elif self.num_slices > 3 and nx >= self.num_slices:
            # Preserve FlowMoDL's contiguous readout crop.
            center_x = int(rng.integers(self.num_slices // 2, nx - (self.num_slices - self.num_slices // 2) + 1))
        xs = np.clip(np.arange(self.num_slices) + center_x - self.num_slices // 2, 0, nx - 1)
        ts = (np.arange(self.num_frames) + center_t - self.num_frames // 2) % nt
        return xs, ts, sample_seed, rng

    def __getitem__(self, index):
        case_index = index // self.samples_per_case
        with self.case_windows(case_index, (index,), cache_bytes=0) as samples:
            return next(samples)

    def case_windows(self, case_index, indices, *, cache_bytes):
        # Generator lifetime owns the read-only file handles, even on early smoke exit.
        return closing(self._case_windows(case_index, indices, cache_bytes))

    def _case_windows(self, case_index, indices, cache_bytes):
        case = self.cases[case_index]
        files = case["files"]
        with ExitStack() as stack:
            # Compressed chunks often span several cardiac frames; avoid inflating
            # the same chunk for each adjacent frame with HDF5's 1 MiB default.
            chunk_cache = {"rdcc_nbytes": min(cache_bytes, 64 * 1024**2), "rdcc_nslots": 10007} if cache_bytes else {}
            d = stack.enter_context(h5py.File(files["kdata_full.mat"], "r", **chunk_cache))["kdata_full"]
            shape = d.shape
            enc, _, _, _, _, _ = shape
            if enc != 4:
                raise ValueError(f"Expected 4 encodings, got {shape}")
            with h5py.File(files["segmask.mat"], "r") as f:
                anatomical_mask = f["segmask"][...]
            with h5py.File(files["coilmap.mat"], "r") as f:
                coilmaps = as_complex(f["coilmap"][...])
            params = read_params(files["params.csv"])
            plans = [self._window(index, shape, anatomical_mask) for index in indices]
            frames = CaseFrames(d, plans, cache_bytes)
            for window_index, (xs, ts, seed, rng) in enumerate(plans):
                yield self._sample(case, shape, xs, ts, seed, rng, anatomical_mask, coilmaps, params,
                                   frames.read(window_index, ts, xs))

    def _sample(self, case, shape, xs, ts, sample_seed, rng, anatomical_mask, coilmaps, params, full):
        enc, nt, nc, nz, ny, nx = shape
        sampling = {}
        if self.online_sampler is None:
            acc = int(self.accelerations[int(rng.integers(len(self.accelerations)))])
            mask = kt_mask(nz, ny, len(ts), acc, sample_seed)
        else:
            pattern, info = self.online_sampler.get(case["case_id"], self.epoch, nt, nz, ny)
            acc = info["requested_R_grid"]
            selected = np.broadcast_to(pattern, (len(ts), nz, ny)) if pattern.ndim == 2 else pattern[ts]
            mask = torch.from_numpy(selected.astype(np.float32))
            sampling = {"sampling_method": info["method"], "sampling_seed": info["seed"],
                        "sampling_native_axes": info["axes"]}
        maps = np.take(coilmaps, xs, axis=-1).transpose(0, 3, 1, 2)
        maps /= np.sqrt((abs(maps) ** 2).sum(axis=0, keepdims=True)).clip(1e-8)
        seg = (np.take(anatomical_mask, xs, axis=-1).transpose(2, 0, 1) > 0.5).astype(np.float32)
        declared = [int(v) for v in params["matrix_size"].split(";")]
        shape_warning = "" if declared == list(shape) else f"params={declared}; tensor shape is authoritative"
        kfull, coils = torch.from_numpy(full), torch.from_numpy(maps)
        measured = kfull * mask[None, :, None, None]
        count = mask.sum() * enc * nc * len(xs)
        scale = (measured.abs().square().sum() / count.clamp_min(1)).sqrt().clamp_min(1e-8)
        measured.div_(scale)
        kfull.div_(scale)
        target = (fft2(kfull, inverse=True) * coils[None, None].conj()).sum(dim=2)
        return {"kspace": measured, "full_kspace": kfull, "coils": coils, "mask": mask,
                "target": target, "segmask": torch.from_numpy(seg),
                "venc": torch.tensor([float(v) for v in params["VENC"].split(";")]),
                "acceleration": acc, "scale": scale, "case_id": case["case_id"],
                "source": case["files"]["kdata_full.mat"], "frame_indices": torch.tensor(ts.copy()),
                "slice_indices": torch.tensor(xs.copy()), "venc_order": params["venc_order"],
                "metadata_shape_warning": shape_warning,
                "spatial_order": params["spatial_order"], "native_shape": torch.tensor(shape),
                **sampling}


class CaseFrames:
    """Cache the union of requested x slices per reusable (encoding, absolute time)."""

    def __init__(self, raw, plans, limit):
        self.raw, self.limit = raw, limit
        self.cache, self.bytes = OrderedDict(), 0
        by_time = {}
        for wi, (xs, ts, _, _) in enumerate(plans):
            for t in np.unique(ts):
                by_time.setdefault(int(t), []).append((wi, xs))
        self.plan = {t: (np.unique(np.concatenate([xs for _, xs in windows])), windows[-1][0])
                     for t, windows in by_time.items() if len(windows) > 1}

    def read(self, window, ts, xs):
        enc, _, nc, nz, ny, _ = self.raw.shape
        full = np.empty((enc, len(ts), nc, len(xs), nz, ny), dtype=np.complex64)
        for e in range(enc):
            for t in np.unique(ts):
                t = int(t)
                key = (e, t)
                plan_xs, last = self.plan.get(t, (xs, window))
                cacheable = window < last and nc * nz * ny * len(plan_xs) * 8 <= self.limit
                entry = self.cache.pop(key, None)
                if entry is not None:
                    self.bytes -= entry.nbytes
                    cropped = entry
                    positions = np.searchsorted(plan_xs, xs)
                else:
                    # scipy.fft keeps complex64; numpy.fft promotes it to complex128.
                    raw = as_complex(self.raw[e, t])
                    hybrid = fft.fftshift(fft.ifft(fft.ifftshift(raw, axes=-1), axis=-1, norm="ortho"), axes=-1)
                    cropped = np.take(hybrid, plan_xs if cacheable else xs, axis=-1).transpose(0, 3, 1, 2)
                    positions = np.searchsorted(plan_xs, xs) if cacheable else slice(None)
                plane = cropped[:, positions]
                for ti in np.flatnonzero(ts == t):
                    full[e, ti] = plane
                if cacheable:
                    while self.bytes + cropped.nbytes > self.limit:
                        self.bytes -= self.cache.popitem(last=False)[1].nbytes
                    self.cache[key] = cropped
                    self.bytes += cropped.nbytes
        return full
