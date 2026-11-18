"""Read-only full cases, on-disk hybrid/GT cache, and shared fixed measurements."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import as_complex, ifftc, kt_mask
from .manifest import GROUPS, read_manifest, read_params
from .path_mapping import protected_case_paths
from .safety import ROOT, OutputGuard, sha256


CACHE_VERSION = "hybrid_x_normalized_coils_v1"


def read_pipeline(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["mask"]["method"] != "kt_gaussian":
        raise ValueError("mask.method currently supports only kt_gaussian")
    if cfg["metrics"]["spatial_mode"] not in {"bbox", "full"}:
        raise ValueError("metrics.spatial_mode must be bbox or full")
    for key in ("neural_merge", "ours_merge"):
        if cfg["inference"][key] not in {"center", "tile", "overlap"}:
            raise ValueError(f"inference.{key} must be center, tile or overlap")
    accelerations = cfg["mask"]["accelerations"]
    if not accelerations or any(type(r) is not int or r not in {10, 20, 30, 40, 50} for r in accelerations):
        raise ValueError("mask.accelerations must be integer factors from [10,20,30,40,50]")
    if type(cfg["mask"]["seed"]) is not int:
        raise ValueError("mask.seed must be an integer")
    for key in ("mask_images", "use_venc", "include_complex_diff"):
        if type(cfg["metrics"][key]) is not bool:
            raise ValueError(f"metrics.{key} must be boolean")
    return cfg


def select_cases(manifest, groups=("train",), case_ids=None, limit=None, *, source_mapping=None):
    cases, _ = read_manifest(manifest, source_mapping)
    if not set(groups).issubset(GROUPS):
        raise ValueError(f"Use original dataset labels: {sorted(GROUPS)}")
    requested = set(case_ids or ())
    selected = [c for c in cases if c["dataset_group"] in groups and (not requested or c["case_id"] in requested)]
    if requested - {c["case_id"] for c in selected}:
        raise ValueError(f"Cases absent from requested groups: {requested - {c['case_id'] for c in selected}}")
    selected.sort(key=lambda c: next(int(r["size_bytes"]) for r in c["rows"] if r["target_file"].endswith("/kdata_full.mat")))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit-cases must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No cases selected")
    protected = [Path(manifest).parent, *protected_case_paths(cases)]
    if source_mapping is not None:
        protected.append(Path(source_mapping).parent)
    guard = OutputGuard(protected)
    return selected, guard


def source_signature(case):
    return {name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                   "mtime_ns": Path(path).stat().st_mtime_ns} for name, path in case["files"].items()}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FullCase:
    """Cache unscaled [E,T,C,X,Z,Y], and unscaled coil-combined [E,T,Z,Y,X]."""

    def __init__(self, case, cache_root, guard):
        self.case, self.guard = case, guard
        self.signature = source_signature(case)
        if case.get("source_mapping"):
            for row in case["rows"]:
                name = Path(row["target_file"]).name
                if self.signature[name]["bytes"] != int(row["size_bytes"]):
                    raise ValueError(f"Mapped source size differs from original manifest: {case['case_id']}/{name}")
        identity = digest_json({"version": CACHE_VERSION, "sources": self.signature})[:24]
        self.directory = guard.directory(Path(cache_root) / "cases" / case["case_id"] / identity)
        metadata = self.directory / "case.json"
        if not metadata.is_file():
            self._build(metadata)
        self.info = json.loads(metadata.read_text())
        if self.info["sources"] != self.signature or self.info["version"] != CACHE_VERSION:
            raise ValueError("Stale/incompatible case cache")
        self.shape = tuple(self.info["native_shape"])
        self.params = self.info["params"]
        self.hybrid = np.load(self.directory / "hybrid.npy", mmap_mode="r", allow_pickle=False)
        self.coils = np.load(self.directory / "coils.npy", mmap_mode="r", allow_pickle=False)
        self.gt = np.load(self.directory / "gt.npy", mmap_mode="r", allow_pickle=False)
        if self.hybrid.shape != (self.shape[0], self.shape[1], self.shape[2], self.shape[5], self.shape[3], self.shape[4]):
            raise ValueError("Invalid hybrid cache shape")
        if self.gt.shape != (4, self.shape[1], *self.shape[3:]):
            raise ValueError("Invalid GT cache shape")
        if self.coils.shape != (self.shape[2], self.shape[5], self.shape[3], self.shape[4]):
            raise ValueError("Invalid coil cache shape")

    def _build(self, metadata):
        lock = self.guard.check(self.directory / "build.lock")
        started = time.monotonic()
        while True:
            try:
                with lock.open("x") as stream:
                    stream.write(str(os.getpid()))
                break
            except FileExistsError:
                if metadata.is_file():
                    return
                if time.monotonic() - started > 600:
                    raise RuntimeError(f"Case cache is locked; inspect unfinished builder: {lock}")
                time.sleep(1)
        temporary = []
        try:
            if metadata.is_file():
                return
            print(f"CACHE_BUILD {self.case['case_id']} (dataset is read-only)", flush=True)
            with h5py.File(self.case["files"]["kdata_full.mat"], "r") as f:
                d = f["kdata_full"]
                if len(d.shape) != 6 or d.shape[0] != 4:
                    raise ValueError(f"Expected [4,T,C,Z,Y,X], got {d.shape}")
                if self.case.get("source_mapping"):
                    declared = next(json.loads(r["shape"]) for r in self.case["rows"] if r["target_file"].endswith("/kdata_full.mat"))
                    if list(d.shape) != declared:
                        raise ValueError(f"Mapped k-space shape differs from original manifest: {d.shape} vs {declared}")
                e, nt, nc, z, y, x = d.shape
                params = read_params(self.case["files"]["params.csv"])
                with h5py.File(self.case["files"]["coilmap.mat"], "r") as c:
                    maps = as_complex(c["coilmap"][...]).transpose(0, 3, 1, 2)
                if maps.shape != (nc, x, z, y):
                    raise ValueError("Coilmap is not aligned with raw k-space")
                maps /= np.sqrt((abs(maps) ** 2).sum(axis=0, keepdims=True)).clip(1e-8)
                outputs = {}
                for name, shape in (("hybrid", (e, nt, nc, x, z, y)), ("gt", (e, nt, z, y, x)), ("coils", maps.shape)):
                    path = self.guard.check(self.directory / f"{name}.{uuid.uuid4().hex}.tmp.npy")
                    temporary.append(path)
                    outputs[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex64, shape=shape)
                outputs["coils"][:] = maps
                for ei in range(e):
                    for ti in range(nt):
                        hybrid = ifftc(as_complex(d[ei, ti]), (-1,)).transpose(0, 3, 1, 2)
                        outputs["hybrid"][ei, ti] = hybrid
                        image = (ifftc(hybrid, (-2, -1)) * maps.conj()).sum(axis=0)
                        if not np.isfinite(image).all():
                            raise FloatingPointError("Non-finite GT reconstructed from full k-space")
                        outputs["gt"][ei, ti] = image.transpose(1, 2, 0)
                for value in outputs.values():
                    value.flush()
                del outputs
                if source_signature(self.case) != self.signature:
                    raise RuntimeError("Dataset changed while reading; do not publish stale cache")
                for path, name in zip(temporary, ("hybrid", "gt", "coils")):
                    path.replace(self.guard.check(self.directory / f"{name}.npy"))
                self.guard.json(metadata, {"version": CACHE_VERSION, "sources": self.signature,
                                           "native_shape": list(d.shape), "params": params,
                                           "gt_axes": "etzyx", "coils": "unit RSS; GT uses the same maps as inference"})
        finally:
            lock.unlink(missing_ok=True)

    def fixed_mask(self, acceleration, seed):
        _, nt, _, z, y, _ = self.shape
        if int(acceleration) not in {10, 20, 30, 40, 50}:
            raise ValueError("Supported acceleration factors: 10,20,30,40,50")
        identity = {"method": "project_kt_gaussian_v1", "seed": int(seed), "case_id": self.case["case_id"],
                    "R": int(acceleration), "shape": [nt, z, y],
                    "generator_sha256": sha256(ROOT / "scripts/generate_4dflow_ktgaussian_train.py")}
        token = digest_json(identity)
        directory = self.guard.directory(self.directory / "masks" / token[:24])
        path = self.guard.check(directory / "mask.npy")
        if not path.is_file():
            actual_seed = int(token[:16], 16)
            array = kt_mask(z, y, nt, int(acceleration), actual_seed).numpy().astype(np.uint8)
            temp = self.guard.check(directory / f"mask.{uuid.uuid4().hex}.tmp.npy")
            with temp.open("wb") as f:
                np.save(f, array, allow_pickle=False)
            temp.replace(path)
        mask = np.load(path, allow_pickle=False)
        if mask.shape != (nt, z, y) or not np.isin(mask, (0, 1)).all() or not np.all(mask.sum(axis=(1, 2)) > 0):
            raise ValueError("Invalid cached sampling mask")
        details = {**identity, "sha256": sha256(path), "path": str(path),
                   "actual_R": float(mask.size / mask.sum()), "reference_encoding": 0,
                   "shared_encodings": [0, 1, 2, 3], "fully_sampled_axis": "raw kx"}
        self.guard.json(directory / "mask.json", details)
        return mask, details

    def batch(self, frames, slices, acceleration, mask):
        frames, slices = np.asarray(frames, np.int64), np.asarray(slices, np.int64)
        e, nt, nc, z, y, x = self.shape
        if np.any(frames < 0) or np.any(frames >= nt) or np.any(slices < 0) or np.any(slices >= x):
            raise ValueError("Window indices out of bounds")
        full = np.empty((e, len(frames), nc, len(slices), z, y), np.complex64)
        for j, ti in enumerate(frames):
            full[:, j] = np.take(self.hybrid[:, ti], slices, axis=2)
        local_mask = mask[frames].astype(np.float32)
        measured = full * local_mask[None, :, None, None]
        count = local_mask.sum() * e * nc * len(slices)
        scale = np.float32(np.sqrt(np.sum(abs(measured) ** 2, dtype=np.float64) / max(count, 1)))
        scale = max(float(scale), 1e-8)
        # No GT/segmentation is passed to the inference adapters.
        return {"kspace": torch.from_numpy((measured / scale)[None]),
                "coils": torch.from_numpy(np.take(self.coils, slices, axis=1)[None].copy()),
                "mask": torch.from_numpy(local_mask[None]), "acceleration": torch.tensor([acceleration]),
                "frame_indices": torch.from_numpy(frames[None].copy()),
                "slice_indices": torch.from_numpy(slices[None].copy()), "scale": torch.tensor([scale]),
                "native_shape": torch.tensor([self.shape])}

    def segmentation(self):
        with h5py.File(self.case["files"]["segmask.mat"], "r") as f:
            value = f["segmask"][...]
        if value.shape != self.shape[3:]:
            raise ValueError("Segmentation/GT spatial shape mismatch")
        return (value > 0.5).astype(np.float32)

    def assert_unchanged(self):
        if source_signature(self.case) != self.signature:
            raise RuntimeError("Read-only dataset file metadata changed during the operation")
