"""Load the 4 per-enc img4ranking .mat files from a stock inference.py run and
visualize them against the fully-sampled GT and the zero-filled baseline."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio


def _first_key(f: h5py.File) -> str:
    for k in f.keys():
        if not k.startswith("#"):
            return k
    raise KeyError("no data key")


def load_mat(path: Path, key: str | None = None):
    # Try h5py first; fall back to scipy.io for older MAT versions.
    try:
        with h5py.File(path, "r", swmr=True) as f:
            k = key if key and key in f else _first_key(f)
            arr = f[k][()]
        if arr.dtype.names and "real" in arr.dtype.names:
            return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
        return np.asarray(arr)
    except Exception:
        d = scipy.io.loadmat(path)
        k = key if key and key in d else [kk for kk in d if not kk.startswith("__")][0]
        return d[k]


def load_img4ranking(mat_path: Path) -> np.ndarray:
    """Stock inference saves via scipy.io.savemat under key 'img4ranking'.
    The run4Ranking postprocess transposes the axes so stored shape is (x, y, z, t)
    i.e. (x, kz, z=model_slice_axis, t). For our setup where model's (t, z, c, y, x)
    maps to (t, x, c, kz, ky), the reshape is:
        model recon per (t, z) slice = (y=kz, x=ky) after RSS
        rearrange back -> (t, x=model_slice, kz, ky)
        run4Ranking transposes to (ky, kz, x_model, t) i.e. (ky, kz, x, t).
    We return the array as stored and let the viewer sort out axes.
    """
    d = scipy.io.loadmat(mat_path)
    return d["img4ranking"]


def load_gt(case_dir: Path) -> np.ndarray:
    path = case_dir / "kdata_full.mat"
    with h5py.File(path, "r", swmr=True) as f:
        arr = f["kdata_full"][()]
    kf = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    # 3D IFFT + coil RSS.
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kf, axes=(-3, -2, -1)),
                     axes=(-3, -2, -1), norm="ortho"),
        axes=(-3, -2, -1),
    )
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=2))  # (enc, t, kz, ky, x)


def load_zf(case_dir: Path, us_mat_path: Path, us_mask_path: Path) -> np.ndarray:
    """Zero-filled recon from undersampled k-space (ifft+RSS), shape (enc, t, kz, ky, x)."""
    with h5py.File(us_mat_path, "r", swmr=True) as f:
        arr = f["kdata_ktGaussian"][()] if "kdata_ktGaussian" in f else f[_first_key(f)][()]
    if arr.dtype.names:
        kus = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
    else:
        kus = np.asarray(arr)
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kus, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
        axes=(-3, -2, -1),
    )
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=2))  # (enc, t, kz, ky, x)


def _to_uint8(a, vmin, vmax):
    rng = max(vmax - vmin, 1e-8)
    a = np.clip((a.astype(np.float32) - vmin) / rng, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recon_dir", type=Path, required=True,
                   help="Stock inference's val_img4ranking/ directory.")
    p.add_argument("--case_dir", type=Path, required=True,
                   help="Original 4D-Flow case directory (must contain kdata_full.mat).")
    p.add_argument("--us_mat", type=Path, required=True,
                   help="Original undersampled k-space mat (kdata_ktGaussianNN.mat) for zf baseline.")
    p.add_argument("--us_mask", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stack per-enc.
    recons = []
    for e in range(4):
        # filename patterns from convert script.
        matches = sorted(args.recon_dir.glob(f"*enc{e}_kus_*.mat"))
        assert matches, f"no mat file for enc{e} in {args.recon_dir}"
        arr = load_img4ranking(matches[0])
        recons.append(arr)
        print(f"enc{e}: {matches[0].name}  shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.3g} max={arr.max():.3g} mean={arr.mean():.3g}")

    # shapes should match across enc.
    recon = np.stack(recons, axis=0)
    print(f"stacked recon: {recon.shape} dtype={recon.dtype}")

    gt = load_gt(args.case_dir)
    zf = load_zf(args.case_dir, args.us_mat, args.us_mask)
    print(f"gt:    {gt.shape} dtype={gt.dtype}  min={gt.min():.3g} max={gt.max():.3g}")
    print(f"zf:    {zf.shape} dtype={zf.dtype}  min={zf.min():.3g} max={zf.max():.3g}")

    # run4Ranking produces (ky, kz, x_slice, t) per enc. Permute to (enc, t, kz, ky, x_slice).
    # recon shape per enc: (ky, kz, x, t) == (112, 19, 112, 25)
    # Permute last axis to front then rearrange.
    if recon.ndim == 5 and recon.shape[-1] in (gt.shape[1],):  # (enc, ky, kz, x, t)
        recon = np.transpose(recon, (0, 4, 2, 1, 3))  # (enc, t, kz, ky, x)
    print(f"recon permuted to: {recon.shape}")

    # Clip to common grid if shapes differ (run4Ranking center-crops).
    def _crop_match(src, ref):
        out = src
        for d in range(len(ref.shape)):
            s = src.shape[d]; r = ref.shape[d]
            if s != r:
                start = (s - r) // 2
                out = out.take(indices=range(start, start + r), axis=d)
                src = out
        return out

    if recon.shape != gt.shape:
        print(f"shape mismatch recon {recon.shape} vs gt {gt.shape}; cropping to common")
        min_shape = tuple(min(a, b) for a, b in zip(recon.shape, gt.shape))
        def _center_crop(a, shape):
            slices = []
            for s_dim, t_dim in zip(a.shape, shape):
                start = (s_dim - t_dim) // 2
                slices.append(slice(start, start + t_dim))
            return a[tuple(slices)]
        recon = _center_crop(recon, min_shape)
        gt = _center_crop(gt, min_shape)
        zf = _center_crop(zf, min_shape)
        print(f"cropped: recon {recon.shape}, gt {gt.shape}, zf {zf.shape}")

    # Save npz
    np.savez_compressed(args.out_dir / "recon.npz", recon=recon, zf=zf, gt=gt)

    enc, T, kz, ky, nx = recon.shape
    mid_t = T // 2
    mid_kz = kz // 2
    mid_x = nx // 2

    # Panel 1: (ky, x) at middle kz for each enc: Model / ZF / GT
    fig, axes = plt.subplots(enc, 3, figsize=(9, 3 * enc))
    if enc == 1:
        axes = axes[None, :]
    for e in range(enc):
        imgs = [recon[e, mid_t, mid_kz, :, :],
                zf[e, mid_t, mid_kz, :, :],
                gt[e, mid_t, mid_kz, :, :]]
        vmax = max(float(np.max(i)) for i in imgs)
        for c, t in enumerate(["Stock-inference recon", "Zero-filled", "GT"]):
            axes[e, c].imshow(imgs[c].T, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
            axes[e, c].axis("off")
            if e == 0:
                axes[e, c].set_title(t)
    fig.suptitle(f"Stock inference on converted 4D Flow data | kz={mid_kz} t={mid_t}/{T}")
    fig.tight_layout()
    p1 = args.out_dir / "panel_kyx.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1}")

    # Panel 2: (kz, ky) at middle x for each enc
    fig, axes = plt.subplots(enc, 3, figsize=(9, 3 * enc))
    if enc == 1:
        axes = axes[None, :]
    for e in range(enc):
        imgs = [recon[e, mid_t, :, :, mid_x],
                zf[e, mid_t, :, :, mid_x],
                gt[e, mid_t, :, :, mid_x]]
        vmax = max(float(np.max(i)) for i in imgs)
        for c, t in enumerate(["Stock-inference recon", "Zero-filled", "GT"]):
            axes[e, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=vmax, aspect="auto")
            axes[e, c].axis("off")
            if e == 0:
                axes[e, c].set_title(t)
    fig.suptitle(f"(kz, ky) plane | x={mid_x} t={mid_t}/{T}")
    fig.tight_layout()
    p2 = args.out_dir / "panel_kzky.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")

    # Time-gif at enc0
    vmax = float(recon[0, :, mid_kz, :, :].max())
    frames = []
    for ti in range(T):
        row = np.concatenate([
            _to_uint8(recon[0, ti, mid_kz, :, :].T, 0, vmax),
            _to_uint8(zf[0, ti, mid_kz, :, :].T, 0, vmax),
            _to_uint8(gt[0, ti, mid_kz, :, :].T, 0, vmax),
        ], axis=1)
        frames.append(row)
    gif = args.out_dir / "enc0_time.gif"
    imageio.mimsave(gif, frames, duration=0.12)
    print(f"saved {gif}")

    # Stats
    print()
    print("Per-enc stats (recon | zf | gt):")
    print(f"{'enc':>3} {'arr':>7} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'q99':>10} {'nan':>5}")
    for e in range(enc):
        for name, arr in [("recon", recon), ("zf", zf), ("gt", gt)]:
            a = arr[e]
            nan = int(np.isnan(a).sum())
            fin = a[np.isfinite(a)]
            print(f"{e:>3} {name:>7} {fin.min():>10.4g} {fin.max():>10.4g} "
                  f"{fin.mean():>10.4g} {fin.std():>10.4g} "
                  f"{np.quantile(fin, 0.99):>10.4g} {nan:>5}")
    print()
    print("Pearson corr recon <-> gt and best-scale rmse per enc:")
    for e in range(enc):
        r = recon[e].ravel(); g = gt[e].ravel()
        r_n = (r - r.mean()) / (r.std() + 1e-8)
        g_n = (g - g.mean()) / (g.std() + 1e-8)
        corr = float((r_n * g_n).mean())
        alpha = float((r * g).sum() / ((r * r).sum() + 1e-12))
        rmse = float(((g - alpha * r) ** 2).mean() ** 0.5)
        print(f"  enc{e}: corr={corr:.3f}  alpha={alpha:.3f}  rmse/std_gt={rmse/(g.std()+1e-8):.3f}")


if __name__ == "__main__":
    main()
