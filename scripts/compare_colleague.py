"""Compare the colleague-conversion recon against GT from kdata_full."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def read_mat_any(path, key=None):
    try:
        with h5py.File(path, "r", swmr=True) as f:
            k = key if key and key in f else [kk for kk in f if not kk.startswith("#")][0]
            arr = f[k][()]
        if arr.dtype.names and "real" in arr.dtype.names:
            return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
        return np.asarray(arr)
    except Exception:
        d = scipy.io.loadmat(path)
        k = key if key and key in d else [kk for kk in d if not kk.startswith("__")][0]
        return d[k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recon_dir", type=Path, required=True)
    p.add_argument("--case_dir", type=Path, required=True)
    args = p.parse_args()

    recons = []
    for e in range(4):
        matches = sorted(args.recon_dir.glob(f"*enc{e}_kus_*.mat"))
        arr = read_mat_any(matches[0], "img4ranking")
        print(f"enc{e}: {matches[0].name} shape={arr.shape} dtype={arr.dtype}  "
              f"min={arr.min():.3g} max={arr.max():.3g} mean={arr.mean():.3g}")
        recons.append(arr)
    recon = np.stack(recons, axis=0)
    print(f"stacked recon {recon.shape}")

    # Load GT
    with h5py.File(args.case_dir / "kdata_full.mat", "r", swmr=True) as f:
        arr = f["kdata_full"][()]
    kf = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)  # (enc, t, c, kz, ky, kx)
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kf, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
        axes=(-3, -2, -1),
    )
    gt_full = np.sqrt(np.sum(np.abs(img) ** 2, axis=2))  # (enc, t, z, y, x)
    print(f"GT shape {gt_full.shape}")

    # run4Ranking for "cine" with z >= 3: picks central 2 slices in the THIRD (sz) dim, :3 time,
    # then crops (sx/3, sy/2, 2, 3).
    # Colleague's output: what's (H, W, slices, time) of their saved img4ranking?
    # Their stored kspace after reshape was (nt, nz, coil, ny, nx), model processes per (t,z) slice
    # giving 2D (ny=112, nx=112). run4Ranking receives recon.transpose() so axes flipped.
    # For our stored recon shape, let's match: recon = (sx=112/3=37, sy=112/2=56, 2, 3).

    # Actually recon after stock's run4Ranking is cropped. Compare carefully.
    # For P006: nz=19, nt=25 -> run4Ranking picks 2 middle z-slices, first 3 frames, crops.
    # Let's align GT the same way before correlating.

    # Auto-detect GT permutation by matching dimensions.
    # stock inference saves after `recon.transpose()`, so axis order is reversed from (t, z, y, x).
    # Colleague kz-IFFT pipeline: outputs_rss = (t, z=kz, y=ky, x=kx) -> stored (kx, ky, kz, t)
    # Our  kx-IFFT pipeline:      outputs_rss = (t, z=x,  y=kz, x=ky) -> stored (ky, kz, x, t)
    gt_perm = None
    recon_shape_per_enc = recon.shape[1:]
    # GT layout (t, z, y, x) per enc = (25, 19, 112, 112). Find permutation matching stored recon.
    from itertools import permutations
    gt_per_enc = gt_full[0]  # (25, 19, 112, 112)
    for perm in permutations(range(4)):
        if tuple(gt_per_enc.shape[i] for i in perm) == recon_shape_per_enc:
            gt_perm = (0,) + tuple(i + 1 for i in perm)
            break
    assert gt_perm is not None, f"cannot match gt shape {gt_per_enc.shape} to recon {recon_shape_per_enc}"
    gt_cropped = np.transpose(gt_full, gt_perm)
    print(f"GT permuted with {gt_perm} -> {gt_cropped.shape}")
    assert recon.shape == gt_cropped.shape

    print()
    print("Per-enc comparison (full-size recon vs GT):")
    print(f"{'enc':>3} {'corr':>7} {'alpha':>7} {'rmse/std_gt':>12} {'recon_max':>10} {'gt_max':>10}")
    for e in range(recon.shape[0]):
        r = recon[e].ravel()
        g = gt_cropped[e].ravel()
        r_n = (r - r.mean()) / (r.std() + 1e-8)
        g_n = (g - g.mean()) / (g.std() + 1e-8)
        corr = float((r_n * g_n).mean())
        alpha = float((r * g).sum() / ((r * r).sum() + 1e-12))
        rmse = float(((g - alpha * r) ** 2).mean() ** 0.5)
        print(f"{e:>3} {corr:>7.3f} {alpha:>7.3f} {rmse/(g.std()+1e-8):>12.3f} "
              f"{r.max():>10.3f} {g.max():>10.3f}")


if __name__ == "__main__":
    main()
