import numpy as np
import sys

p = sys.argv[1]
d = np.load(p, allow_pickle=True)
recon = d["recon"]  # (enc, t, kz, ky, x)
zf = d["zf"]
gt = d["gt"]

print(f"recon {recon.shape} dtype={recon.dtype}")
print(f"zf    {zf.shape} dtype={zf.dtype}")
print(f"gt    {gt.shape} dtype={gt.dtype}" if gt is not None and gt.ndim > 0 else "gt None")

print()
print("Per-enc stats:")
print(f"{'enc':>3} {'arr':>6} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'q50':>10} {'q99':>10} {'nan':>6}")
for e in range(recon.shape[0]):
    for name, arr in [("recon", recon), ("zf", zf), ("gt", gt)]:
        if arr is None or getattr(arr, "ndim", 0) == 0:
            continue
        a = arr[e]
        fin = np.isfinite(a)
        if not fin.all():
            a = a[fin]
        print(
            f"{e:>3} {name:>6} "
            f"{a.min():>10.4g} {a.max():>10.4g} {a.mean():>10.4g} {a.std():>10.4g} "
            f"{np.quantile(a, 0.5):>10.4g} {np.quantile(a, 0.99):>10.4g} "
            f"{(~fin).sum():>6d}"
        )
    print()

# Correlation between recon and gt, per-enc, at multiple x positions.
if gt is not None and gt.ndim > 0:
    print("Correlation of recon vs gt per enc (over all t, kz, ky, x):")
    for e in range(recon.shape[0]):
        r = recon[e].ravel()
        g = gt[e].ravel()
        # Normalize since magnitudes may differ.
        r_n = (r - r.mean()) / (r.std() + 1e-8)
        g_n = (g - g.mean()) / (g.std() + 1e-8)
        corr = float((r_n * g_n).mean())
        # Scale ratio (gt / recon) at high-signal locations
        mask = g > np.quantile(g, 0.95)
        scale = float(g[mask].mean() / (r[mask].mean() + 1e-12))
        print(f"  enc{e}: pearson_r={corr:.3f}  gt/recon_scale_at_hi_signal={scale:.3f}")

    print()
    print("Mean intensity profile along x (enc0, averaged over t, kz, ky):")
    prof_r = recon[0].mean(axis=(0, 1, 2))
    prof_g = gt[0].mean(axis=(0, 1, 2))
    prof_z = zf[0].mean(axis=(0, 1, 2))
    for i in range(0, prof_r.size, max(1, prof_r.size // 10)):
        print(f"   x={i:3d}: recon={prof_r[i]:.4g} zf={prof_z[i]:.4g} gt={prof_g[i]:.4g}")

    # Check if recon time-variation resembles gt time-variation (diff across t).
    print()
    print("Temporal variability (std across t, averaged over kz, ky, x) per enc:")
    for e in range(recon.shape[0]):
        sd_r = recon[e].std(axis=0).mean()
        sd_g = gt[e].std(axis=0).mean()
        sd_z = zf[e].std(axis=0).mean()
        print(f"  enc{e}: recon_std_over_t={sd_r:.4g}  zf_std_over_t={sd_z:.4g}  gt_std_over_t={sd_g:.4g}")
