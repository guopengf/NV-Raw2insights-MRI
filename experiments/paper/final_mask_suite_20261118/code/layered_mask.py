#!/usr/bin/env python3
"""Research approximation of three-layer UIH-like sampling, not vendor code.

Full rectangular ACS and a regular rectangular annulus surround strict outer
Poisson-disc points. Time and all 4 encodings share one pattern by default;
optional varying frames retain the same fixed ACS/grid. Exact counts never
override the prescribed distance or zones.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

PROTECTED_ROOTS = (Path("/mnt/nas"), Path("/media"),
    Path("/home/zhanghs/NV-Raw2insights-MRI/experiments/paper"),
    Path("/home/zhanghs/NV-Raw2insights-MRI/outputs/paper"))


def _ints(values, count, name, positive=False):
    values = tuple(values)
    if len(values) != count or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in values):
        raise ValueError(f"{name} requires {count} integers")
    result = tuple(int(x) for x in values)
    if positive and min(result) < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _rectangle(shape, bounds, name):
    z0, z1, y0, y1 = _ints(bounds, 4, name)
    if not (0 <= z0 < z1 <= shape[0] and 0 <= y0 < y1 <= shape[1]):
        raise ValueError(f"{name} must be half-open [z0,z1,y0,y1] within [Z,Y]")
    result = np.zeros(shape, dtype=bool)
    result[z0:z1, y0:y1] = True
    return result


def acs_checker_band(shape, acs_bounds, width):
    """Return the two lateral band footprints and their staggered sampled sites."""
    z0, z1, y0, y1 = acs_bounds
    z, y = np.ogrid[:shape[0], :shape[1]]
    band = (z > z0) & (z < z1 - 1) & (
        ((y >= y0 - width) & (y < y0)) | ((y >= y1) & (y < y1 + width))
    )
    return band, band & ((z - z0 + y - y0) % 2 == 1)


def make_regions(ny, nz, acs_bounds, ring_bounds, *, grid_mode="lattice",
                 grid_stride=(2, 2), grid_phase=(0, 0), grid_tile=None, support=None,
                 acs_checker_width=0):
    """Return ACS, selected annular grid, and allowed outer boolean maps [Z,Y].

    Grid phase is absolute (z,y). Lattice uses each axis modulo its stride;
    checkerboard uses (z-phase_z+y-phase_y) modulo 2 and requires strides (2,2).
    Periodic mode indexes a binary unit cell with phase modulo its dimensions.
    """
    ny, nz = _ints((ny, nz), 2, "shape", positive=True)
    stride = _ints(grid_stride, 2, "grid_stride", positive=True)
    phase = _ints(grid_phase, 2, "grid_phase")
    if grid_mode == "periodic":
        grid_tile = np.asarray(grid_tile)
        if (grid_tile.ndim != 2 or min(grid_tile.shape) < 1
                or not np.all((grid_tile == 0) | (grid_tile == 1)) or not np.any(grid_tile)):
            raise ValueError("periodic grid_tile must be a nonempty binary 2D matrix")
        grid_tile = grid_tile.astype(bool)
    elif grid_tile is not None:
        raise ValueError("grid_tile is used only with grid_mode='periodic'")
    phase_extent = grid_tile.shape if grid_mode == "periodic" else stride
    if any(p < 0 or p >= s for p, s in zip(phase, phase_extent)):
        raise ValueError("grid_phase must be within the tile dimensions or grid stride")
    acs = _rectangle((nz, ny), acs_bounds, "acs_bounds")
    ring = _rectangle((nz, ny), ring_bounds, "ring_bounds")
    if np.any(acs & ~ring):
        raise ValueError("ring_bounds must contain the full ACS rectangle")
    z, y = np.indices((nz, ny))
    if grid_mode == "lattice":
        regular = ((z - phase[0]) % stride[0] == 0) & ((y - phase[1]) % stride[1] == 0)
    elif grid_mode == "checkerboard" and stride == (2, 2):
        regular = (z - phase[0] + y - phase[1]) % 2 == 0
    elif grid_mode == "periodic":
        regular = grid_tile[(z - phase[0]) % grid_tile.shape[0], (y - phase[1]) % grid_tile.shape[1]]
    else:
        raise ValueError("grid_mode must be lattice, checkerboard (strides 2,2), or periodic")
    grid = ring & ~acs & regular
    width, = _ints((acs_checker_width,), 1, "acs_checker_width")
    if width < 0:
        raise ValueError("acs_checker_width must be nonnegative")
    if width:
        z0, z1, y0, y1 = acs_bounds
        if z1 - z0 < 3 or y0 - width < ring_bounds[2] or y1 + width > ring_bounds[3]:
            raise ValueError("Middle zone must contain both complete ACS checkerboard bands")
        band, checker = acs_checker_band((nz, ny), acs_bounds, width)
        grid = (grid & ~band) | checker
    if support is None:
        support = np.ones((nz, ny), dtype=bool)
    else:
        support = np.asarray(support)
        if support.shape != (nz, ny) or not np.all((support == 0) | (support == 1)):
            raise ValueError("support must be binary [Z,Y]")
        support = support.astype(bool)
    if np.any((acs | grid) & ~support):
        raise ValueError("Mandatory ACS/grid points lie outside support")
    return acs, grid, support & ~ring


def _draw_outer(points, log_weights, target, distance, rng, shape):
    selected = np.zeros(shape, dtype=bool)
    if target == 0:
        return selected
    uniform = np.maximum(rng.random(len(points)), np.finfo(float).tiny)
    order = np.argsort(-(log_weights - np.log(-np.log(uniform))))
    forbidden = np.zeros(shape, dtype=bool)
    radius = min(int(np.ceil(distance)), max(shape))
    dz, dy = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    keep = dz * dz + dy * dy < distance * distance
    offsets = np.column_stack((dz[keep], dy[keep]))
    count = 0
    for index in order:
        z, y = points[index]
        if forbidden[z, y]:
            continue
        selected[z, y] = True
        count += 1
        if count == target:
            return selected
        blocked = offsets + (z, y)
        valid = ((blocked[:, 0] >= 0) & (blocked[:, 0] < shape[0])
                 & (blocked[:, 1] >= 0) & (blocked[:, 1] < shape[1]))
        blocked = blocked[valid]
        forbidden[blocked[:, 0], blocked[:, 1]] = True
    return None


def generate_mask(ny, nz, nt, R, seed, *, acs_bounds, ring_bounds,
                  grid_mode="lattice", grid_stride=(2, 2), grid_phase=(0, 0),
                  grid_tile=None, time_shared=True, acs_checker_width=0,
                  min_distance=2.0, density_sigma=None, center=None, support=None,
                  counts=None, max_attempts=64):
    """Return (uint8 [4,T,Z,Y] shared view, JSON-compatible metadata).

    R uses the full rectangular Ny*Nz grid. Optional counts is an exact length-T
    integer vector. Distance is Euclidean grid spacing between outer points only.
    density_sigma is a fraction of each full axis, with optional center (z,y).
    Finite greedy attempts may fail even when another packing could exist.
    time_shared=True generates one pattern for all T and requires equal counts.
    """
    ny, nz, nt, max_attempts = _ints((ny, nz, nt, max_attempts), 4, "shape/attempts", positive=True)
    if not np.isfinite(R) or R <= 0 or not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError("R and min_distance must be positive and finite")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if density_sigma is not None and (not np.isfinite(density_sigma) or density_sigma <= 0):
        raise ValueError("density_sigma must be positive and finite")
    center = np.asarray(((nz - 1) / 2, (ny - 1) / 2) if center is None else center, dtype=float)
    if (center.shape != (2,) or not np.all(np.isfinite(center)) or np.any(center < 0)
            or center[0] > nz - 1 or center[1] > ny - 1):
        raise ValueError("center must be finite (z,y) inside the grid")
    acs, grid, outer = make_regions(ny, nz, acs_bounds, ring_bounds, grid_mode=grid_mode,
                                   grid_stride=grid_stride, grid_phase=grid_phase, grid_tile=grid_tile,
                                   support=support, acs_checker_width=acs_checker_width)
    mandatory = acs | grid
    mandatory_count = int(mandatory.sum())
    budgets = np.full(nt, int(ny * nz // R)) if counts is None else np.asarray(counts)
    if (budgets.shape != (nt,) or not np.issubdtype(budgets.dtype, np.integer)
            or np.any(budgets < 1) or np.any(budgets > ny * nz)):
        raise ValueError("counts must be a positive integer vector of length T within the grid")
    if np.any(budgets < mandatory_count):
        raise ValueError(f"Mandatory ACS/grid needs {mandatory_count} points; max R={ny * nz / mandatory_count:g}")
    if time_shared and np.any(budgets != budgets[0]):
        raise ValueError("Shared time requires identical counts across T")
    outer_counts = budgets - mandatory_count
    if not time_shared and nt > 1 and np.any(outer_counts == 0):
        raise ValueError("Nt>1 requires outer points in every frame for temporal variation")
    points = np.argwhere(outer)
    if np.any(outer_counts > len(points)):
        raise ValueError("Outer budget exceeds allowed outer support")
    log_weights = np.zeros(len(points))
    if density_sigma is not None:
        normalized = (points - center) / (nz, ny)
        log_weights = -0.5 * np.sum((normalized / density_sigma) ** 2, axis=1)
        log_weights = np.maximum(log_weights, np.log(np.finfo(float).tiny))
    generated_frames = 1 if time_shared else nt
    patterns = np.zeros((generated_frames, nz, ny), dtype=np.uint8)
    attempts_used = []
    rng = np.random.default_rng(int(seed))
    for frame, target in enumerate(outer_counts[:generated_frames]):
        for attempt in range(max_attempts):
            selected = _draw_outer(points, log_weights, int(target), min_distance, rng, (nz, ny))
            if selected is None:
                continue
            candidate = mandatory | selected
            if frame and np.array_equal(candidate, patterns[frame - 1]):
                continue
            patterns[frame] = candidate
            attempts_used.append(attempt + 1)
            break
        else:
            raise ValueError(f"Exact outer Poisson budget {target} with distance {min_distance:g} or temporal "
                             f"variation infeasible in {max_attempts} seeded attempts; explicitly tune distance/bounds")
    mask = np.broadcast_to(patterns[None], (4, nt, nz, ny))
    metadata = dict(method="research approximation; not vendor code", shape=list(mask.shape),
        requested_R_grid=float(R), actual_R=(ny * nz / budgets).tolist(), seed=int(seed),
        acs_bounds=[int(x) for x in acs_bounds], ring_bounds=[int(x) for x in ring_bounds], grid_mode=grid_mode,
        acs_checker_width=int(acs_checker_width),
        grid_stride=[int(x) for x in grid_stride], grid_phase=[int(x) for x in grid_phase], acs_count=int(acs.sum()),
        grid_tile=None if grid_tile is None else np.asarray(grid_tile, dtype=np.uint8).tolist(),
        grid_count=int(grid.sum()), mandatory_count=mandatory_count, outer_support_count=len(points),
        counts=budgets.tolist(), outer_counts=outer_counts.tolist(), min_distance=float(min_distance),
        distance_scope="outer points only", density_sigma=None if density_sigma is None else float(density_sigma), center=center.tolist(),
        support_provided=support is not None, max_attempts=max_attempts, attempts_used=attempts_used,
        encodings_shared=True, time_shared=bool(time_shared), temporal_mode="shared" if time_shared else "varying",
        adjacent_frames_vary=nt > 1 and not time_shared)
    return mask, metadata


def compact_mask(mask):
    """Return [1,1|T,1,Z,Y,1], collapsing identical encodings and time frames."""
    mask = np.asarray(mask)
    if (mask.ndim != 4 or mask.shape[0] != 4 or min(mask.shape) < 1
            or not np.all((mask == 0) | (mask == 1)) or not np.all(mask == mask[:1])):
        raise ValueError("Expected binary [4,T,Z,Y] with shared encodings")
    shared = mask[:1]
    if np.all(shared == shared[:, :1]):
        shared = shared[:, :1]
    return shared[:, :, None, :, :, None]


def broadcast_mask(compact, kspace_shape):
    compact = np.asarray(compact)
    shape = _ints(kspace_shape, 6, "kspace_shape", positive=True)
    if (compact.ndim != 6 or compact.shape[0] != 1 or compact.shape[2] != 1 or compact.shape[5] != 1
            or shape[0] != 4 or compact.shape[1] not in (1, shape[1]) or compact.shape[3:5] != shape[3:5]):
        raise ValueError("Compact mask must match k-space [4,T,C,Z,Y,X]")
    return np.broadcast_to(compact, shape)


def write_mask(path, mask, metadata):
    """Exclusive atomic NPZ/HDF5 output; NAS/media/paper trees are protected."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Output must not be a symlink")
    path = path.resolve()
    if any(path.is_relative_to(root.resolve()) for root in PROTECTED_ROOTS):
        raise ValueError("Refusing output inside protected NAS/media/paper root")
    if path.suffix.lower() not in (".npz", ".mat", ".h5", ".hdf5"):
        raise ValueError("Output must be .npz or HDF5 .mat/.h5/.hdf5")
    if path.exists():
        raise FileExistsError(path)
    mask = np.asarray(mask)
    if mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Output mask must be binary uint8")
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        if path.suffix.lower() == ".npz":
            with open(temporary, "wb") as handle:
                np.savez_compressed(handle, mask=mask, metadata=np.asarray(encoded))
        else:
            import h5py
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset("mask", data=mask, compression="gzip", shuffle=True)
                handle.create_dataset("metadata", data=encoded, dtype=h5py.string_dtype())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_mask(path):
    """Read without squeezing or changing axis order."""
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            return handle["mask"], json.loads(str(handle["metadata"]))
    import h5py
    with h5py.File(path, "r") as handle:
        return handle["mask"][()], json.loads(handle["metadata"][()])
