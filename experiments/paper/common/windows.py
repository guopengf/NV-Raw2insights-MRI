"""Explicit full-case window scheduling and complex, physical-scale assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    frames: np.ndarray
    slices: np.ndarray
    emit_t: np.ndarray
    emit_x: np.ndarray


def windows(nt, nx, frames, slices, *, center_time, merge):
    if merge not in {"center", "tile", "overlap"}:
        raise ValueError("merge must be center, tile or overlap")
    if min(nt, nx, frames, slices) < 1:
        raise ValueError("Window/case dimensions must be positive")
    if merge == "center":
        for ti in range(nt):
            ts = (ti + np.arange(frames) - frames // 2) % nt
            for xi in range(nx):
                xs = np.clip(xi + np.arange(slices) - slices // 2, 0, nx - 1)
                yield Window(ts, xs, np.array([frames // 2]), np.array([slices // 2]))
    elif merge == "tile":
        # Our last cascade repeats the center temporal estimate: never emit those copies.
        tstarts = range(nt) if center_time else range(0, nt, frames)
        for ti in tstarts:
            ts = ((ti - frames // 2 if center_time else ti) + np.arange(frames)) % nt
            et = np.array([frames // 2]) if center_time else np.arange(min(frames, nt - ti))
            for xi in range(0, nx, slices):
                start = min(xi, max(0, nx - slices))
                xs = np.clip(start + np.arange(slices), 0, nx - 1)
                yield Window(ts, xs, et, np.arange(xi - start, xi - start + min(slices, nx - xi)))
    else:
        for ti in (range(nt) if center_time else range(0, nt, max(1, frames // 2))):
            ts = (ti + np.arange(frames) - frames // 2) % nt
            et = np.array([frames // 2]) if center_time else np.arange(frames)
            for xi in range(0, nx, max(1, slices // 2)):
                xs = np.clip(xi + np.arange(slices) - slices // 2, 0, nx - 1)
                yield Window(ts, xs, et, np.arange(slices))


class Assembly:
    """Only declared output positions contribute; no interpolation or magnitude conversion."""

    def __init__(self, result, coverage):
        self.result, self.coverage = result, coverage
        self.result[:] = 0
        self.coverage[:] = 0

    def add(self, prediction, window, scale):
        # [1,E,T,S,Z,Y] -> [E,Nt,Z,Y,Nx].
        p = np.asarray(prediction)
        if p.ndim != 6 or p.shape[:2] != (1, 4) or not np.iscomplexobj(p):
            raise ValueError(f"Expected complex [1,4,T,S,Z,Y], got {p.shape}")
        if p.shape[2:4] != (len(window.frames), len(window.slices)):
            raise ValueError("Prediction does not preserve its input window dimensions")
        if p.shape[4:] != self.result.shape[2:4]:
            raise ValueError("Prediction z/y spatial axes do not match the output case")
        if not np.isfinite(p).all() or not np.isfinite(scale) or scale <= 0:
            raise FloatingPointError("Invalid prediction/normalization scale")
        for lt in window.emit_t:
            for lx in window.emit_x:
                ti, xi = int(window.frames[lt]), int(window.slices[lx])
                self.result[:, ti, :, :, xi] += p[0, :, lt, lx] * scale
                self.coverage[ti, xi] += 1

    def finish(self):
        if np.any(self.coverage == 0):
            raise ValueError(f"Full-case assembly has {np.sum(self.coverage == 0)} uncovered time/slice positions")
        # Divide per plane to avoid allocating a second full case in RAM.
        for ti in range(self.result.shape[1]):
            for xi in range(self.result.shape[-1]):
                self.result[:, ti, :, :, xi] /= self.coverage[ti, xi]
        if hasattr(self.result, "flush"):
            self.result.flush()
        return {"coverage_min": int(self.coverage.min()), "coverage_max": int(self.coverage.max()),
                "covered_time_slice_pairs": int(self.coverage.size), "complete": True}
