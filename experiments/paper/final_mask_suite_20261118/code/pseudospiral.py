#!/usr/bin/env python3
"""Paper-inspired Cartesian ky-kz occupancy; never a vendor acquisition replica.

Generate [T,Z,Y] once and broadcast it across four velocity encodings. Each one
selects all frequency-readout and coil samples. Exact unique budgets, independent
frame phases and optional support are retrospective adaptations, not scanner code.
MAT output is ordinary HDF5 for h5py readers, not MATLAB v7.3 serialization.
"""
from __future__ import annotations

import hashlib

import numpy as np

GOLDEN_DEG = 180.0 * (3.0 - np.sqrt(5.0))
DEFAULTS = {'proud': (75, 3.0, GOLDEN_DEG),
            'proud_carotid': (100, 3.0, 23.63)}


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    try:
        valid = np.isfinite(value) and int(value) == value and value >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def generate_pattern(ny: int, nz: int, nt: int, R: float, seed: int = 0, *,
                     model: str = 'proud', counts: np.ndarray | None = None,
                     support: np.ndarray | None = None, denominator: str = 'rect',
                     acs: np.ndarray | None = None,
                     radius_power: float | None = None, turns: float | None = None,
                     points_per_arm: int | None = None, rotation_deg: float | None = None,
                     center_offset: tuple[float, float] = (0.0, 0.0),
                     radial_jitter: float = 0.0, arm_angle_jitter_deg: float = 0.0,
                     max_arms: int | None = None,
                     return_metadata: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """Return binary uint8 [T,Z,Y]; counts override floor(denominator/R).

    counts is a scalar, [T], or identical four rows [4,T]. center_offset is (Y,Z)
    in grid cells.
    All temporal frames use one common family with deterministic random phases.
    arm_angle_jitter_deg perturbs each whole arm uniformly in +/- degrees;
    radial_jitter perturbs each normalized radial coordinate in +/- radius.
    These are retrospective design additions, not recovered scanner parameters.
    Unreachable budgets raise after max_arms; there is no random-fill fallback.
    """
    ny, nz, nt = (_integer(ny, 'ny'), _integer(nz, 'nz'), _integer(nt, 'nt'))
    seed = _integer(seed, 'seed', 0)
    if model not in DEFAULTS or denominator not in ('rect', 'support'):
        raise ValueError('unknown model or denominator')
    if not np.isscalar(R) or not np.isfinite(R) or R < 1:
        raise ValueError('R must be finite and >= 1')
    n_default, turn_default, rot_default = DEFAULTS[model]
    radius_power = 2.0 if radius_power is None else radius_power
    n = _integer(n_default if points_per_arm is None else points_per_arm,
                 'points_per_arm', 2)
    omega = 2*np.pi*turn_default
    if turns is not None:
        omega = 2*np.pi*turns
    rotation = rot_default if rotation_deg is None else rotation_deg
    offset = np.asarray(center_offset, dtype=float)
    if (offset.shape != (2,) or not np.isfinite(offset).all() or
            not np.isfinite([radius_power, omega, rotation, radial_jitter, arm_angle_jitter_deg]).all() or
            radius_power <= 0 or omega < 0 or radial_jitter < 0 or
            not 0 <= arm_angle_jitter_deg <= 180):
        raise ValueError('invalid finite geometry parameters')
    if support is None:
        eligible = np.ones((nz, ny), dtype=bool)
    else:
        eligible = np.asarray(support)
        if eligible.shape != (nz, ny) or not np.isin(eligible, [0, 1]).all():
            raise ValueError('support must be a binary [Z,Y] array')
        eligible = eligible.astype(bool)
    available = int(eligible.sum())
    if not available:
        raise ValueError('support is empty')
    nominal_total = ny*nz if denominator == 'rect' else available
    if counts is None:
        budgets = np.full(nt, np.floor(nominal_total/R))
    else:
        budgets = np.asarray(counts)
        if budgets.shape == (4, nt):
            if not np.all(budgets == budgets[0]):
                raise ValueError('four encodings must have identical budgets')
            budgets = budgets[0]
        if budgets.shape not in ((), (nt,)):
            raise ValueError('counts must be scalar, [T], or shared [4,T]')
        budgets = np.broadcast_to(budgets, (nt,))
    if (not np.issubdtype(budgets.dtype, np.number) or
            not np.isfinite(budgets).all() or np.iscomplexobj(budgets) or
            np.any(budgets != np.floor(budgets)) or
            np.any((budgets < 1) | (budgets > available))):
        raise ValueError('each unique count must be an integer in [1, support size]')
    budgets = budgets.astype(np.int64)
    fixed = np.zeros((nz, ny), bool) if acs is None else np.asarray(acs, dtype=bool)
    if fixed.shape != eligible.shape or np.any(fixed & ~eligible) or fixed.sum() >= budgets.min():
        raise ValueError('ACS must lie in support and leave a free sample in every frame')
    fixed_count = int(fixed.sum())
    limit = max(1024, 32*available) if max_arms is None else _integer(max_arms, 'max_arms')
    pattern = np.zeros((nt, nz, ny), dtype=np.uint8)
    arms_used = []
    for t, budget in enumerate(budgets):
        rng = np.random.default_rng(np.random.SeedSequence([seed, t]))
        phase = rng.uniform(0, 2*np.pi)
        pattern[t] = fixed
        filled = fixed_count
        for arm in range(limit):
            angle_offset = phase + arm*np.deg2rad(rotation)
            if arm_angle_jitter_deg:
                angle_offset += np.deg2rad(rng.uniform(-arm_angle_jitter_deg,arm_angle_jitter_deg))
            u = np.linspace(0, 1, n)
            rho = u**radius_power
            if radial_jitter:
                rho = np.clip(rho + rng.uniform(-radial_jitter, radial_jitter, n), 0, 1)
            theta = omega*u + angle_offset
            y = ny/2*rho*np.cos(theta) + offset[0]
            z = nz/2*rho*np.sin(theta) + offset[1]
            # MATLAB's nearest-integer ties are away from zero.
            ys = (np.sign(y)*np.floor(np.abs(y)+0.5) + ny//2).astype(int)
            zs = (np.sign(z)*np.floor(np.abs(z)+0.5) + nz//2).astype(int)
            for y, z in zip(ys, zs):
                if 0 <= y < ny and 0 <= z < nz and eligible[z, y] and not pattern[t, z, y]:
                    pattern[t, z, y] = 1
                    filled += 1
                    if filled == budget:
                        break
            if filled == budget:
                arms_used.append(arm+1)
                break
        else:
            raise RuntimeError(f'frame {t}: only {filled}/{budget} unique points after '
                               f'{limit} arms; change geometry/support/budget explicitly')
    degenerate = nt > 1 and np.all(pattern == pattern[0])
    if degenerate and not np.all(budgets == available):
        raise RuntimeError('geometry/budget produced identical time frames; vary geometry '
                           'or budget explicitly (no hidden fill or temporal shuffling)')
    metadata = {
        'model': model, 'provenance': 'new paper-inspired retrospective binary generator; not vendor source',
        'source_url': {'proud': 'https://doi.org/10.1002/nbm.4255',
                       'proud_carotid': 'https://doi.org/10.1186/s12968-019-0582-z'}[model],
        'axes': 'TZY', 'seed': seed, 'shape': list(pattern.shape), 'encodings': 4,
        'encoding_policy': 'generate TZY once, broadcast identically over four encodings',
        'time_policy': 'varying frame phases with deterministic SeedSequence([seed,t])',
        'temporal_degeneracy': 'full eligible support' if degenerate else None,
        'R_requested': float(R), 'denominator': denominator,
        'counts': budgets.tolist(), 'R_rect': (ny*nz/budgets).tolist(),
        'R_support': (available/budgets).tolist(), 'support_count': available,
        'support_sha256': hashlib.sha256(eligible.tobytes()).hexdigest(),
        'points_per_arm': n, 'radius_power': float(radius_power),
        'turns': float(omega/(2*np.pi)), 'omega': float(omega),
        'rotation_deg': float(rotation), 'center_offset_YZ': offset.tolist(),
        'radial_jitter': float(radial_jitter),
        'arm_angle_jitter_deg': float(arm_angle_jitter_deg),
        'jitter_policy': 'uniform radial offset per point; uniform angular offset per whole arm; generated once per time frame, shared across encodings',
        'max_arms': limit, 'arms_used': arms_used,
        'quantization': 'nearest integer centered coordinate; half ties away from zero',
        'native_geometry': 'normalized power-law radius and independent arm rotation',
        'adaptation': 'exact unique count via deduplication and continued arms; frame phase seed; optional support; '
                      'shared four encodings; no scanner timing, physiological binning or repeated-readout weights',
    }
    return (pattern, metadata) if return_metadata else pattern
