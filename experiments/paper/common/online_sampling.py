"""Per-case retrospective sampling using the external final mask suite."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from .safety import ROOT, sha256


METHODS = ("gaussian", "pr4d", "pseudo", "uih")
MODULES = ("mask_suite", "gaussian", "layered_mask", "pr4d_light", "pseudospiral")


@lru_cache(maxsize=1)
def load_suite(directory):
    paths = {name: directory / f"{name}.py" for name in MODULES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Final mask suite source not found: {path}")
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "") or "").resolve() != path.resolve():
            raise ImportError(f"Final mask suite module {name!r} conflicts with an existing import")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module("mask_suite")
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous


class OnlineUndersampling:
    """A worker-local cache with deterministic regeneration after eviction."""

    def __init__(self, config, seed):
        self.suite_path = (ROOT / Path(config["suite_path"]).expanduser()).resolve()
        if self.suite_path.name != "mask_suite.py" or not self.suite_path.is_file():
            raise ValueError(f"online_undersampling.suite_path must point to the final code/mask_suite.py: {self.suite_path}")
        self.methods = tuple(config.get("methods", METHODS))
        if not self.methods or len(set(self.methods)) != len(self.methods) or not set(self.methods).issubset(METHODS):
            raise ValueError(f"online_undersampling.methods must select distinct methods from {METHODS}")
        self.rate_mode = config.get("rate_mode", "continuous")
        if self.rate_mode not in {"continuous", "integer"}:
            raise ValueError("online_undersampling.rate_mode must be continuous or integer")
        self.low, self.high = map(float, config.get("rate_range", (10, 50)))
        if not 10 <= self.low <= self.high <= 50:
            raise ValueError("online_undersampling.rate_range must be within [10,50]")
        if self.rate_mode == "integer" and not (self.low.is_integer() and self.high.is_integer()):
            raise ValueError("Integer sampling requires integer bounds")
        self.acs = tuple(config.get("acs_zy", (3, 3)))
        if len(self.acs) != 2 or any(type(n) is not int or n < 1 for n in self.acs):
            raise ValueError("online_undersampling.acs_zy must contain two positive integers")
        self.pseudo_profile = config.get("pseudo_profile", "aorta_c001_R10")
        self.cache_size = config.get("cache_cases", 1024)
        if type(self.cache_size) is not int or self.cache_size < 1:
            raise ValueError("online_undersampling.cache_cases must be a positive integer")
        self.seed, self.epoch, self.cache = seed, None, OrderedDict()

    def provenance(self):
        code = self.suite_path.parent
        paths = [*(code / f"{name}.py" for name in MODULES),
                 *(code.parent / name for name in ("pseudo_profiles.json", "uih_profile.json"))]
        return {"suite_path": str(self.suite_path), "sha256": {p.name: sha256(p) for p in paths},
                "scope": "one mask per case per epoch, shared across workers and methods using the same policy and seed"}

    def get(self, case_id, epoch, nt, nz, ny):
        if epoch != self.epoch:
            self.cache.clear()
            self.epoch = epoch
        key = (case_id, nt, nz, ny)
        if key in self.cache:
            self.cache.move_to_end(key)
            packed, meta = self.cache[key]
            pattern = np.unpackbits(packed, axis=-1, count=ny)
            pattern.setflags(write=False)
            return pattern, meta
        token = hashlib.sha256(json.dumps([self.seed, epoch, case_id]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:16], "little"))
        method = self.methods[int(rng.integers(len(self.methods)))]
        if self.rate_mode == "integer":
            rate = float(rng.integers(int(self.low), int(self.high) + 1))
        else:
            rate = float(rng.uniform(self.low, self.high))
        seed = int(rng.integers(0, 2**63, dtype=np.int64))
        suite = load_suite(self.suite_path.parent)
        try:
            pattern, meta = suite.generate(method, ny, nz, nt, rate, seed,
                                           acs_size=self.acs, pseudo_profile=self.pseudo_profile)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Online mask failed: case={case_id}, epoch={epoch}, method={method}, "
                             f"R={rate:.8g}, seed={seed}, TZY={(nt, nz, ny)}: {exc}") from exc
        native_shape = (nz, ny) if method == "uih" else (nt, nz, ny)
        if pattern.shape != native_shape or not np.isin(pattern, (0, 1)).all():
            raise ValueError(f"Invalid native binary {method} mask: {pattern.shape}, expected {native_shape}")
        budget = int(np.floor(ny * nz / rate))
        if not np.all(pattern.sum(axis=(-2, -1)) == budget):
            raise ValueError(f"{method} mask violates the exact point budget")
        if method != "uih" and np.any(np.all(pattern[1:] == pattern[:-1], axis=(1, 2))):
            raise ValueError(f"{method} mask repeated in adjacent absolute frames")
        pattern.setflags(write=False)
        # Keep one epoch of shuffled cases in RAM without storing a byte per point.
        self.cache[key] = (np.packbits(pattern, axis=-1), meta)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return pattern, meta


def online_sampler(config, seed):
    if config is None:
        return None
    if not isinstance(config, dict) or type(config.get("enabled")) is not bool:
        raise ValueError("paper.online_undersampling.enabled must be a boolean")
    return OnlineUndersampling(config, seed) if config["enabled"] else None


def sampling_log(batch):
    if "sampling_seed" not in batch:
        return {}
    points = int(batch["mask"][0, 0].sum())
    return {"method": batch["sampling_method"][0], "seed": int(batch["sampling_seed"][0]),
            "native_axes": batch["sampling_native_axes"][0], "requested_R": float(batch["acceleration"][0]),
            "actual_R": batch["mask"][0, 0].numel() / points, "points_per_frame": points}
