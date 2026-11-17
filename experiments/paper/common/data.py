"""Read real training slabs; IFFT raw kx, preserve encoding indices and zy sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import read_params
from .safety import ROOT


def as_complex(array):
    if array.dtype.fields and {"real", "imag"}.issubset(array.dtype.fields):
        return (array["real"] + 1j * array["imag"]).astype(np.complex64)
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
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
                 accelerations=(10, 20, 30, 40, 50), smoke=False):
        self.cases = json.loads(Path(index).read_text())
        if not self.cases or any(c["dataset_group"] != "train" for c in self.cases):
            raise ValueError("TrainSlabs accepts only an explicit, nonempty train index")
        self.cases.sort(key=lambda c: next(int(r["size_bytes"]) for r in c["rows"] if r["target_file"].endswith("/kdata_full.mat")))
        if smoke:
            self.cases = self.cases[:1]
        self.smoke = smoke
        self.seed, self.num_slices, self.num_frames = seed, num_slices, num_frames
        self.samples_per_case, self.accelerations, self.epoch = samples_per_case, list(accelerations), 0
        if num_slices < 1:
            raise ValueError("num_slices must be positive (our ablations use odd S=3; FlowMoDL uses S=64)")

    def __len__(self):
        return len(self.cases) * self.samples_per_case

    def __getitem__(self, index):
        case = self.cases[index // self.samples_per_case]
        sample_seed = self.seed + self.epoch * len(self) + index
        rng = np.random.default_rng(sample_seed)
        files = case["files"]
        with h5py.File(files["segmask.mat"], "r") as f:
            anatomical_mask = f["segmask"][...]
        with h5py.File(files["kdata_full.mat"], "r") as f:
            d = f["kdata_full"]
            enc, nt, nc, nz, ny, nx = d.shape
            if enc != 4:
                raise ValueError(f"Expected 4 encodings, got {d.shape}")
            center_x, center_t = int(rng.integers(nx)), int(rng.integers(nt))
            if self.smoke:
                # Exercise ROI/flow losses on train anatomy instead of an empty background slab.
                center_x = int(np.argmax(anatomical_mask.sum(axis=(0, 1))))
            elif self.num_slices > 3 and nx >= self.num_slices:
                # Preserve FlowMoDL's contiguous readout crop without replicated edge voxels.
                center_x = int(rng.integers(self.num_slices // 2, nx - (self.num_slices - self.num_slices // 2) + 1))
            xs = np.clip(np.arange(self.num_slices) + center_x - self.num_slices // 2, 0, nx - 1)
            ts = (np.arange(self.num_frames) + center_t - self.num_frames // 2) % nt
            full = np.empty((enc, len(ts), nc, len(xs), nz, ny), dtype=np.complex64)
            for ei in range(enc):
                for ti, raw_t in enumerate(ts):
                    # Read one full-kx frame at a time; never materialize the full time series.
                    hybrid = ifftc(as_complex(d[ei, int(raw_t)]), (-1,))
                    full[ei, ti] = np.take(hybrid, xs, axis=-1).transpose(0, 3, 1, 2)
        with h5py.File(files["coilmap.mat"], "r") as f:
            maps = np.take(as_complex(f["coilmap"][...]), xs, axis=-1).transpose(0, 3, 1, 2)
        maps /= np.sqrt((abs(maps) ** 2).sum(axis=0, keepdims=True)).clip(1e-8)
        seg = (np.take(anatomical_mask, xs, axis=-1).transpose(2, 0, 1) > 0.5).astype(np.float32)
        params = read_params(files["params.csv"])
        declared = [int(v) for v in params["matrix_size"].split(";")]
        shape_warning = "" if declared == [enc, nt, nc, nz, ny, nx] else f"params={declared}; tensor shape is authoritative"
        acc = int(self.accelerations[int(rng.integers(len(self.accelerations)))])
        mask = kt_mask(nz, ny, len(ts), acc, sample_seed)
        kfull, coils = torch.from_numpy(full.copy()), torch.from_numpy(maps.copy())
        measured = kfull * mask[None, :, None, None]
        count = mask.sum() * enc * nc * len(xs)
        scale = (measured.abs().square().sum() / count.clamp_min(1)).sqrt().clamp_min(1e-8)
        measured, kfull = measured / scale, kfull / scale
        target = (fft2(kfull, inverse=True) * coils[None, None].conj()).sum(dim=2)
        return {"kspace": measured, "full_kspace": kfull, "coils": coils, "mask": mask,
                "target": target, "segmask": torch.from_numpy(seg),
                "venc": torch.tensor([float(v) for v in params["VENC"].split(";")]),
                "acceleration": acc, "scale": scale, "case_id": case["case_id"],
                "source": files["kdata_full.mat"], "frame_indices": torch.tensor(ts.copy()),
                "slice_indices": torch.tensor(xs.copy()), "venc_order": params["venc_order"],
                "metadata_shape_warning": shape_warning,
                "spatial_order": params["spatial_order"], "native_shape": torch.tensor([enc, nt, nc, nz, ny, nx])}
