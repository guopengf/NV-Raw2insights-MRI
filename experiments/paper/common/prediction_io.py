"""Load complete native reconstructions or explicitly indexed external MAT/NPY/NPZ."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def load_array(path, key=None):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if {"coords", "data", "shape"}.issubset(data.files):
                value = np.zeros(tuple(data["shape"].astype(int)), data["data"].dtype)
                if data["coords"].size:
                    value[tuple(data["coords"].T)] = data["data"]
                return value
            chosen = key or ("reconstruction" if "reconstruction" in data else data.files[0])
            return data[chosen]
    if path.suffix == ".mat":
        keys = (key,) if key else ("reconstruction", "img4ranking", "recon", "image")
        try:
            with h5py.File(path, "r") as f:
                chosen = next((k for k in keys if k in f), None)
                if chosen is None:
                    raise ValueError(f"Missing reconstruction key: {path}; keys={list(f)}")
                return f[chosen][...]
        except OSError:
            data = scipy.io.loadmat(path)
            chosen = next((k for k in keys if k in data), None)
            if chosen is None:
                raise ValueError(f"Missing reconstruction key: {path}")
            return data[chosen]
    raise ValueError(f"Unsupported reconstruction format: {path}")


def oriented_complex(path, layout, key=None):
    a = load_array(path, key)
    if a.dtype.fields and {"real", "imag"}.issubset(a.dtype.fields):
        a = a["real"] + 1j * a["imag"]
    elif not np.iscomplexobj(a) and a.ndim and a.shape[-1] == 2 and a.ndim == len(layout) + 1:
        a = a[..., 0] + 1j * a[..., 1]
    if not np.iscomplexobj(a):
        raise ValueError(f"Magnitude-only reconstruction cannot produce flow metrics: {path}")
    target = "etzyx" if "e" in layout else "tzyx"
    if sorted(layout) != sorted(target) or a.ndim != len(target):
        raise ValueError(f"Invalid layout/shape for {path}: {layout!r}, {a.shape}; no guessing of coil/duplicate axes")
    return np.transpose(a, tuple(layout.index(axis) for axis in target))


def discover(root):
    result = []
    for metadata in sorted(Path(root).rglob("reconstruction.json")):
        value = json.loads(metadata.read_text())
        if value.get("scope") != "full_case" or not value.get("coverage", {}).get("complete"):
            raise ValueError(f"Not a complete-case prediction: {metadata}")
        if (metadata.parent / "inference.lock").exists():
            raise ValueError(f"Inference is still running: {metadata.parent}")
        result.append({"case_id": value["case_id"], "R": int(value["R"]), "experiment": value["experiment"],
                       "metadata": value, "parts": [{"path": metadata.parent / "reconstruction.npy", "layout": "etzyx"}]})
    return result


def read_index(path):
    """Columns: case_id,R,recon_path,layout[,encoding,key,experiment,mask_sha256]."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"case_id", "R", "recon_path", "layout"}.issubset(reader.fieldnames or ()):
            raise ValueError("Prediction CSV requires case_id,R,recon_path,layout")
        groups = defaultdict(list)
        for row in reader:
            source = Path(row["recon_path"])
            source = source if source.is_absolute() else path.parent / source
            experiment = row.get("experiment") or "external"
            groups[(experiment, row["case_id"], int(row["R"]))].append(
                {"path": source, "layout": row["layout"], "key": row.get("key") or None,
                 "encoding": int(row["encoding"]) if row.get("encoding") else None,
                 "mask_sha256": row.get("mask_sha256") or None})
    return [{"case_id": cid, "R": r, "experiment": exp, "metadata": {}, "parts": parts}
            for (exp, cid, r), parts in groups.items()]


def load_prediction(entry):
    parts = entry["parts"]
    signatures = {p.get("mask_sha256") for p in parts}
    if len(signatures) > 1:
        raise ValueError("Per-encoding predictions must all specify the same mask SHA, or all leave it unknown")
    if len(parts) == 1 and parts[0].get("encoding") is None:
        p = parts[0]
        value = oriented_complex(p["path"], p["layout"], p.get("key"))
    else:
        encodings = [p.get("encoding") for p in parts]
        if len(parts) != 4 or set(encodings) != {0, 1, 2, 3}:
            raise ValueError("Per-encoding CSV requires exactly enc0/enc1/enc2/enc3 once")
        value = np.stack([oriented_complex(p["path"], p["layout"], p.get("key"))
                          for p in sorted(parts, key=lambda p: p["encoding"])])
    if value.ndim != 5 or value.shape[0] != 4 or not np.isfinite(value).all():
        raise ValueError(f"Expected finite complex [4,T,Z,Y,X], got {value.shape}")
    return value
