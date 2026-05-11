"""Sanity-check the shipped ValidationSet case load path.

Confirm:
  1. kdata_ktGaussian20.mat is the masked (sparse) k-space vs the full-acquired one.
  2. kdata_ktGaussian20 nonzero positions match usmask_ktGaussian20 positions.
  3. Axis layout matches (enc, t, c, kz, ky, kx) with kx broadcast/fully sampled.
  4. Centered FFT convention: zero-filled IFFT of kdata_ktGaussian20 produces a
     sensibly-aligned image (anatomy in the middle, not shifted by N/2).
"""
import sys
from pathlib import Path

import h5py
import numpy as np

case = Path(sys.argv[1])


def read_complex(path, key=None):
    with h5py.File(path, "r") as f:
        k = key if key and key in f else [kk for kk in f if not kk.startswith("#")][0]
        arr = f[k][()]
    if arr.dtype.names and "real" in arr.dtype.names:
        return arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32), k
    return np.asarray(arr), k


def read_real(path, key=None):
    with h5py.File(path, "r") as f:
        k = key if key and key in f else [kk for kk in f if not kk.startswith("#")][0]
        return np.asarray(f[k][()]), k


kfull, kf_key = read_complex(case / "kdata_full.mat")
kus, ku_key = read_complex(case / "kdata_ktGaussian20.mat")
mask, m_key = read_real(case / "usmask_ktGaussian20.mat")

print(f"kfull key={kf_key!r}  shape={kfull.shape}  dtype={kfull.dtype}")
print(f"kus   key={ku_key!r}  shape={kus.shape}   dtype={kus.dtype}")
print(f"mask  key={m_key!r}  shape={mask.shape}   dtype={mask.dtype}  unique={np.unique(mask)}")

# Check if kus is sparsely masked.
nz_full = np.abs(kfull[0, 0, 0]) > 0  # (kz, ky, kx) bool
nz_us = np.abs(kus[0, 0, 0]) > 0
print(f"\nnonzero fraction (enc0, t0, c0):")
print(f"  kfull: {nz_full.mean():.3f}")
print(f"  kus:   {nz_us.mean():.3f}")

# For the 2D (kz, ky) plane at any (enc, t, c, kx):
kx_mid = kus.shape[-1] // 2
nz_us_kzky = np.abs(kus[0, 0, 0, :, :, kx_mid]) > 0
print(f"  kus (kz,ky) at kx={kx_mid}: {nz_us_kzky.mean():.3f}")

# Check: is kus zero at mask-zero positions?
# mask shape (1, t=25, 1, kz, ky, 1). Broadcast to (enc, t, c, kz, ky, kx) and check.
mask_b = np.broadcast_to(mask, kus.shape)
mask_zero = mask_b == 0
mask_one = mask_b == 1
kus_at_mask_zero = np.abs(kus[mask_zero])
kus_at_mask_one = np.abs(kus[mask_one])
print(f"\nkus values at mask==0 positions: mean={kus_at_mask_zero.mean():.4g} max={kus_at_mask_zero.max():.4g}")
print(f"kus values at mask==1 positions: mean={kus_at_mask_one.mean():.4g} max={kus_at_mask_one.max():.4g}")

if kus_at_mask_zero.max() < 1e-6 * kus_at_mask_one.max():
    print(" -> kus is sparsely masked (zeros match mask exactly)")
else:
    print(" -> kus has non-zero values at mask==0 positions (NOT pre-masked)")

# Zero-filled image from kus — check alignment.
# 3D IFFT on (kz, ky, kx), RSS over coils.
img_zf = np.fft.fftshift(
    np.fft.ifftn(np.fft.ifftshift(kus, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
    axes=(-3, -2, -1),
)
zf_rss = np.sqrt(np.sum(np.abs(img_zf) ** 2, axis=2))  # (enc, t, z, y, x)

# Check: is brightness concentrated near array center or at corners?
e0 = zf_rss[0, 0]  # (z, y, x)
zc, yc, xc = [s // 2 for s in e0.shape]
center_crop = e0[zc - 2 : zc + 3, yc - 10 : yc + 11, xc - 10 : xc + 11]
corner = e0[:5, :10, :10]
print(f"\nzero-filled RSS brightness check (enc0, t0):")
print(f"  center (5x21x21): mean={center_crop.mean():.4g}  max={center_crop.max():.4g}")
print(f"  corner (5x10x10): mean={corner.mean():.4g}  max={corner.max():.4g}")
if center_crop.mean() > 2 * corner.mean():
    print("  -> brightness is at array center — FFT shifts look correct")
else:
    print("  -> brightness is NOT at center — POSSIBLE shift error!")

# Compare to the GT image for alignment.
img_full = np.fft.fftshift(
    np.fft.ifftn(np.fft.ifftshift(kfull, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="ortho"),
    axes=(-3, -2, -1),
)
gt_rss = np.sqrt(np.sum(np.abs(img_full) ** 2, axis=2))
peak_zf = np.unravel_index(zf_rss[0, 0].argmax(), zf_rss[0, 0].shape)
peak_gt = np.unravel_index(gt_rss[0, 0].argmax(), gt_rss[0, 0].shape)
print(f"\npeak location (z,y,x):")
print(f"  zf: {peak_zf} out of {zf_rss[0,0].shape}")
print(f"  gt: {peak_gt} out of {gt_rss[0,0].shape}")

# One more check: is the FE (kx) axis already image-space or still k-space?
# If kx is fully sampled with DC at center, 1D IFFT should produce a smooth profile.
# If kx is stored as image, 1D FFT would produce k-space (Fourier-dense).
kx_prof_kspace = np.mean(np.abs(kus[0, 0, 0]), axis=(0, 1))  # (kx,)
kx_prof_image = np.mean(np.abs(np.fft.ifft(np.fft.ifftshift(kus[0, 0, 0], axes=-1),
                                             axis=-1, norm="ortho")), axis=(0, 1))
print(f"\nalong-kx profile (should be smooth/peaked at center in k-space if kx is really kspace):")
print(f"  as-stored: max@{kx_prof_kspace.argmax()}  center-max={kx_prof_kspace[len(kx_prof_kspace)//2]:.3g}  edge-max={kx_prof_kspace[0]:.3g}")
print(f"  after 1D IFFT on kx: max@{kx_prof_image.argmax()}  center-max={kx_prof_image[len(kx_prof_image)//2]:.3g}  edge-max={kx_prof_image[0]:.3g}")
