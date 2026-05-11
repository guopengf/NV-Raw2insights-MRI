"""Diagnose per-slice continuity along x and produce error-map panels."""
import numpy as np
import sys, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

d = np.load(p, allow_pickle=True)
recon = d["recon"]  # (enc, t, kz, ky, x)
zf = d["zf"]
gt = d["gt"]
E, T, KZ, KY, X = recon.shape

# 1. Spatial x-continuity: mean |diff| along x, averaged.
def x_roughness(a):
    return np.mean(np.abs(np.diff(a, axis=-1)))

print("Spatial x-gradient mean|d/dx| (proxy for slice-wise discontinuity):")
for e in range(E):
    print(
        f"  enc{e}: recon={x_roughness(recon[e]):.4g}  "
        f"zf={x_roughness(zf[e]):.4g}  gt={x_roughness(gt[e]):.4g}  "
        f"ratio recon/gt={x_roughness(recon[e])/max(x_roughness(gt[e]),1e-12):.2f}"
    )

# 2. Global scale match — fit single scalar alpha to minimize |gt - alpha*recon|^2.
print("\nBest scalar rescaling alpha = (recon·gt)/(recon·recon):")
for e in range(E):
    r = recon[e].ravel(); g = gt[e].ravel()
    alpha = float((r * g).sum() / (r * r).sum())
    err = float(np.mean((g - alpha * r) ** 2) ** 0.5)
    gt_std = float(g.std())
    print(f"  enc{e}: alpha={alpha:.3f}  rmse_after_rescale={err:.4g}  (gt std={gt_std:.4g}, nrmse={err/gt_std:.2f})")

# 3. Per-slice panel: recon, gt, |recon - alpha*gt| at mid t, mid kz, all enc.
mid_t, mid_kz = T // 2, KZ // 2
alphas = [float((recon[e].ravel()*gt[e].ravel()).sum() / (recon[e].ravel()**2).sum()) for e in range(E)]

fig, axes = plt.subplots(E, 4, figsize=(16, 3 * E))
if E == 1:
    axes = axes[None, :]
col_titles = ["Model (rescaled)", "Zero-filled", "GT", "|Model·α − GT|"]
for e in range(E):
    alpha = alphas[e]
    rec = alpha * recon[e, mid_t, mid_kz, :, :]
    z = zf[e, mid_t, mid_kz, :, :]
    g = gt[e, mid_t, mid_kz, :, :]
    err = np.abs(rec - g)
    imgs = [rec.T, z.T, g.T, err.T]
    vmax = float(np.max([rec, g]))
    for c in range(4):
        im = axes[e, c].imshow(imgs[c], cmap=("hot" if c == 3 else "gray"), vmin=0,
                               vmax=(vmax if c != 3 else vmax * 0.5), aspect="auto")
        axes[e, c].axis("off")
        if e == 0:
            axes[e, c].set_title(col_titles[c])
    axes[e, 0].set_title(axes[e, 0].get_title() + f" (α={alpha:.2f})" if e == 0 else f"α={alpha:.2f}")

fig.suptitle(f"Error panel | t={mid_t}/{T}  kz={mid_kz}/{KZ}  (mid slice)")
fig.tight_layout()
panel = os.path.join(out_dir, "error_panel.png")
fig.savefig(panel, dpi=120, bbox_inches="tight")
print(f"\nsaved {panel}")
plt.close(fig)

# 4. Line plot of x-profile at (t,kz,ky)=center to see slice-wise jitter.
fig, axes = plt.subplots(E, 1, figsize=(10, 2.5 * E))
if E == 1:
    axes = [axes]
ky_mid = KY // 2
for e in range(E):
    alpha = alphas[e]
    axes[e].plot(alpha * recon[e, mid_t, mid_kz, ky_mid, :], label=f"model·{alpha:.2f}")
    axes[e].plot(zf[e, mid_t, mid_kz, ky_mid, :], label="zf", linestyle=":")
    axes[e].plot(gt[e, mid_t, mid_kz, ky_mid, :], label="gt", linestyle="--")
    axes[e].set_title(f"enc{e} x-profile at (t={mid_t}, kz={mid_kz}, ky={ky_mid})")
    axes[e].legend()
    axes[e].set_xlabel("x"); axes[e].set_ylabel("|image|")
fig.tight_layout()
prof = os.path.join(out_dir, "x_profile.png")
fig.savefig(prof, dpi=120, bbox_inches="tight")
print(f"saved {prof}")
plt.close(fig)
