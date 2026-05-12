#!/usr/bin/env python3
"""Generate kt-Gaussian masks and undersampled k-space for 4D Flow Aorta data.

The expected input layout is:

    Aorta/CenterXXX/Scanner_Model/PYYY/kdata_full.mat

Each full k-space file is expected to contain a 6D array with layout:

    (Nv, Nt, Nc, SPE, PE, FE)

For every requested acceleration R, this script writes:

    usmask_ktGaussian{R}.mat  with shape (1, Nt, 1, SPE, PE, 1)
    kdata_ktGaussian{R}.mat   with shape (Nv, Nt, Nc, SPE, PE, FE)

The 2D kt-Gaussian mask generator follows the CMRx4DFlow2026
CMRx4DFlowMaskGeneration MATLAB/Python implementation. The FE/kx direction is
kept as singleton length 1 in the mask because sampling is accelerated in the
SPE/PE plane and broadcast over FE.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised on systems without h5py.
    h5py = None

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover - exercised on systems without scipy.
    sio = None


DEFAULT_ROOT = Path("/SSDHome/share/haosen/4dflow/TrainSet/Aorta")
DEFAULT_ACCELERATIONS = (10, 20, 30, 40, 50)
KDATA_KEYS = ("kdata_full", "kdata", "kspace_full", "kspace")


class MatArray:
    def __init__(self, source, key: str, close_file=None):
        self.source = source
        self.key = key
        self.close_file = close_file
        self.shape = tuple(source.shape)
        self.dtype = source.dtype

    def __getitem__(self, index):
        return self.source[index]

    def read_all(self):
        return self.source[()]

    def close(self) -> None:
        if self.close_file is not None:
            self.close_file.close()
            self.close_file = None


def parse_accelerations(value: str) -> tuple[int, ...]:
    accels = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not accels:
        raise argparse.ArgumentTypeError("at least one acceleration is required")
    if any(acc <= 0 for acc in accels):
        raise argparse.ArgumentTypeError("accelerations must be positive integers")
    return accels


def stable_seed(base_seed: int | None, patient_dir: Path, acc: int) -> int | None:
    if base_seed is None:
        return None
    text = f"{base_seed}:{patient_dir.as_posix()}:{acc}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def iter_patient_dirs(root: Path) -> Iterable[Path]:
    for center_dir in sorted(root.glob("Center*")):
        if not center_dir.is_dir():
            continue
        for scanner_dir in sorted(center_dir.iterdir()):
            if not scanner_dir.is_dir():
                continue
            for patient_dir in sorted(scanner_dir.iterdir()):
                if patient_dir.is_dir() and (patient_dir / "kdata_full.mat").exists():
                    yield patient_dir


def open_first_mat_array(path: Path, keys: tuple[str, ...]) -> MatArray:
    if h5py is not None:
        try:
            f = h5py.File(path, "r")
        except OSError:
            f = None
        if f is not None:
            for key in keys:
                if key in f:
                    return MatArray(f[key], key, close_file=f)
            array_keys = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
            if array_keys:
                key = array_keys[0]
                return MatArray(f[key], key, close_file=f)
            f.close()

    if sio is None:
        raise RuntimeError(
            f"Could not read {path}. Install h5py for MATLAB v7.3 files or scipy for v7 files."
        )
    mat = sio.loadmat(path)
    for key in keys:
        if key in mat:
            return MatArray(np.asarray(mat[key]), key)
    for key, value in mat.items():
        if not key.startswith("__"):
            return MatArray(np.asarray(value), key)
    raise ValueError(f"No array found in {path}")


def create_gaussian_weight_matrix(mask_size: tuple[int, int], sigma_x: float, sigma_y: float) -> np.ndarray:
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2.0, width - (width + 1) / 2.0, width)
    y = np.linspace(1 - (height + 1) / 2.0, height - (height + 1) / 2.0, height)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.exp(-(x_grid**2 / (2.0 * sigma_x**2) + y_grid**2 / (2.0 * sigma_y**2)))


def random_sampling_optimized(
    mask_size: tuple[int, int],
    total_points: int,
    weight: np.ndarray,
    min_dist_lookup: np.ndarray,
    rng: np.random.Generator,
    existing_mask: np.ndarray,
) -> np.ndarray:
    if total_points <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    width, height = mask_size
    sampled_points: list[list[int]] = []
    current_weight = weight * (1 - existing_mask.astype(np.float32))
    if np.sum(current_weight) <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    forbidden_mask = np.zeros((height, width), dtype=bool)
    batch_size = max(total_points * 2, 1000)
    prob = current_weight.ravel() / np.sum(current_weight)
    indices = rng.choice(width * height, size=batch_size, p=prob)

    count = 0
    idx_ptr = 0
    y_grid, x_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = int(indices[idx_ptr])
        idx_ptr += 1
        y, x = divmod(idx, width)

        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        sampled_points.append([x + 1, y + 1])
        count += 1

        dist = float(min_dist_lookup[y, x])
        if dist > 0:
            y_min = max(0, int(y - dist))
            y_max = min(height, int(y + dist + 1))
            x_min = max(0, int(x - dist))
            x_max = min(width, int(x + dist + 1))
            region_y = y_grid[y_min:y_max, 0]
            region_x = x_grid[0, x_min:x_max]
            dist_sq = (region_y[:, None] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= dist_sq < dist**2

        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask.astype(np.float32)) * (1 - forbidden_mask)
            if np.sum(current_weight) <= 0:
                break
            prob = current_weight.ravel() / np.sum(current_weight)
            indices = rng.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    if not sampled_points:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sampled_points, dtype=np.int64)


def generate_ktgaussian_mask_2d(
    spe: int,
    pe: int,
    nt: int,
    acc: int,
    rng: np.random.Generator,
    *,
    min_dist_factor: float = 3.0,
    rep_decay_factor: float = 0.5,
    center_radius_x: float = 0.5,
    center_radius_y: float = 0.5,
) -> np.ndarray:
    """Return binary masks with shape (SPE, PE, Nt)."""
    mask_size = (pe, spe)
    total_points = max(1, int((spe * pe) // int(acc)))
    sigma_x = pe / 5.0
    sigma_y = spe / 5.0

    width, height = mask_size
    masks = np.zeros((height, width, nt), dtype=np.float32)
    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    if center_radius_x <= 0.5 or center_radius_y <= 0.5:
        center_ellipse = np.zeros((height, width), dtype=bool)
        center_ellipse[height // 2, width // 2] = True
    else:
        x_grid, y_grid = np.meshgrid(
            np.arange(1, width + 1, dtype=np.float64),
            np.arange(1, height + 1, dtype=np.float64),
        )
        center_ellipse = (
            ((x_grid - width / 2.0) / center_radius_x) ** 2
            + ((y_grid - height / 2.0) / center_radius_y) ** 2
            <= 1.0
        )
    num_center = int(np.sum(center_ellipse))
    if num_center > total_points:
        print(
            f"warning: center calibration has {num_center} points, larger than target "
            f"{total_points}; effective acceleration will be lower than {acc}."
        )

    for frame in range(nt):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[center_ellipse] = 1.0

        needed = max(0, total_points - num_center)
        points = random_sampling_optimized(mask_size, needed, weight, min_dist_lookup, rng, mask > 0)
        for x_one_based, y_one_based in points:
            x = int(x_one_based) - 1
            y = int(y_one_based) - 1
            mask[y, x] = 1.0
            weight[y, x] *= rep_decay_factor

        current_points = int(np.sum(mask))
        if current_points < total_points:
            extra = random_sampling_optimized(
                mask_size,
                total_points - current_points,
                weight,
                min_dist_lookup,
                rng,
                mask > 0,
            )
            for x_one_based, y_one_based in extra:
                x = int(x_one_based) - 1
                y = int(y_one_based) - 1
                mask[y, x] = 1.0
                weight[y, x] *= rep_decay_factor

        current_points = int(np.sum(mask))
        if current_points < total_points:
            fallback_weight = weight * (1 - (mask > 0).astype(np.float32))
            flat_weight = fallback_weight.ravel()
            if np.sum(flat_weight) <= 0:
                raise RuntimeError("mask generation ran out of available points")
            sampled = rng.choice(
                width * height,
                size=total_points - current_points,
                replace=False,
                p=flat_weight / np.sum(flat_weight),
            )
            ys, xs = np.divmod(sampled, width)
            mask[ys, xs] = 1.0
            weight[ys, xs] *= rep_decay_factor

        masks[:, :, frame] = mask

    return masks


def to_broadcast_mask(mask_2d_t: np.ndarray) -> np.ndarray:
    return np.transpose(mask_2d_t, (2, 0, 1))[None, :, None, :, :, None].astype(np.float32)


def write_h5_array(path: Path, key: str, array: np.ndarray, aliases: tuple[str, ...] = ()) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to write HDF5 MAT files")
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=array, compression="gzip", shuffle=True)
        for alias in aliases:
            f[alias] = h5py.SoftLink(f"/{key}")


def multiply_block_by_mask(block: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if block.dtype.fields is not None and "real" in block.dtype.fields and "imag" in block.dtype.fields:
        out = np.empty_like(block)
        out["real"] = block["real"] * mask
        out["imag"] = block["imag"] * mask
        return out
    return block * mask


def write_undersampled_kspace_h5(path: Path, src: MatArray, mask: np.ndarray) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to write HDF5 MAT files")
    with h5py.File(path, "w") as f:
        chunks = (1, 1) + tuple(src.shape[2:])
        chunks = tuple(min(dim, chunk) for dim, chunk in zip(src.shape, chunks))
        dst = f.create_dataset(
            "kdata_ktGaussian",
            shape=src.shape,
            dtype=src.dtype,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for frame in range(src.shape[1]):
            index = (slice(None), slice(frame, frame + 1), slice(None), slice(None), slice(None), slice(None))
            block = src[index]
            dst[index] = multiply_block_by_mask(block, mask[:, frame : frame + 1, :, :, :, :])


def write_undersampled_kspace_scipy(path: Path, src: MatArray, mask: np.ndarray) -> None:
    if sio is None:
        raise RuntimeError("scipy is required to write MATLAB v5 MAT files")
    kspace = multiply_block_by_mask(src.read_all(), mask)
    sio.savemat(path, {"kdata_ktGaussian": kspace}, do_compression=True)


def save_mask(path: Path, mask: np.ndarray, writer: str) -> None:
    if writer == "scipy":
        if sio is None:
            raise RuntimeError("scipy is required to write MATLAB v5 MAT files")
        sio.savemat(path, {"usmask_ktGaussian": mask, "mask": mask}, do_compression=True)
    else:
        write_h5_array(path, "usmask_ktGaussian", mask, aliases=("mask",))


def save_undersampled_kspace(path: Path, src: MatArray, mask: np.ndarray, writer: str) -> None:
    if writer == "scipy":
        write_undersampled_kspace_scipy(path, src, mask)
    else:
        write_undersampled_kspace_h5(path, src, mask)


def resolve_writer(writer: str) -> str:
    if writer != "auto":
        return writer
    if h5py is not None:
        return "h5py"
    if sio is not None:
        return "scipy"
    raise RuntimeError("Install h5py or scipy before running this script")


def process_patient(
    patient_dir: Path,
    accels: tuple[int, ...],
    writer: str,
    overwrite: bool,
    dry_run: bool,
    base_seed: int | None,
) -> tuple[int, int]:
    src = open_first_mat_array(patient_dir / "kdata_full.mat", KDATA_KEYS)
    try:
        if len(src.shape) != 6:
            raise ValueError(f"{patient_dir / 'kdata_full.mat'} must be 6D, got {src.shape}")

        _, nt, _, spe, pe, _ = src.shape
        written_masks = 0
        written_kspaces = 0

        for acc in accels:
            mask_path = patient_dir / f"usmask_ktGaussian{acc}.mat"
            kspace_path = patient_dir / f"kdata_ktGaussian{acc}.mat"
            need_mask = overwrite or not mask_path.exists()
            need_kspace = overwrite or not kspace_path.exists()

            if not need_mask and not need_kspace:
                print(f"skip existing: {patient_dir} R={acc}")
                continue

            seed = stable_seed(base_seed, patient_dir, acc)
            rng = np.random.default_rng(seed)
            mask_2d_t = generate_ktgaussian_mask_2d(spe, pe, nt, acc, rng)
            mask = to_broadcast_mask(mask_2d_t)
            sampled = int(np.sum(mask_2d_t[:, :, 0]))
            effective_r = (spe * pe) / max(sampled, 1)
            print(
                f"{patient_dir} R={acc}: kdata={src.shape}, mask={mask.shape}, "
                f"points/frame={sampled}, effective_R={effective_r:.2f}"
            )

            if dry_run:
                continue

            if need_mask:
                save_mask(mask_path, mask, writer)
                written_masks += 1
            if need_kspace:
                save_undersampled_kspace(kspace_path, src, mask, writer)
                written_kspaces += 1

        return written_masks, written_kspaces
    finally:
        src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Aorta TrainSet root")
    parser.add_argument(
        "--accelerations",
        type=parse_accelerations,
        default=DEFAULT_ACCELERATIONS,
        help="comma-separated acceleration factors, e.g. 10,20,30,40,50",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing generated files")
    parser.add_argument("--dry-run", action="store_true", help="show what would be generated without writing")
    parser.add_argument(
        "--writer",
        choices=("auto", "h5py", "scipy"),
        default="auto",
        help="output MAT writer; h5py streams large k-space files, scipy writes MATLAB v5 files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="base seed for deterministic per-patient/per-acceleration masks; use -1 for random",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    writer = resolve_writer(args.writer)
    base_seed = None if args.seed < 0 else args.seed
    patients = list(iter_patient_dirs(root))
    if not patients:
        raise FileNotFoundError(f"no patient folders with kdata_full.mat found under {root}")

    total_masks = 0
    total_kspaces = 0
    print(f"found {len(patients)} patients under {root}")
    print(f"using writer={writer}, accelerations={args.accelerations}")
    for patient_dir in patients:
        n_masks, n_kspaces = process_patient(
            patient_dir=patient_dir,
            accels=args.accelerations,
            writer=writer,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            base_seed=base_seed,
        )
        total_masks += n_masks
        total_kspaces += n_kspaces

    if args.dry_run:
        print("dry run complete; no files written")
    else:
        print(f"done: wrote {total_masks} mask files and {total_kspaces} k-space files")


if __name__ == "__main__":
    main()
