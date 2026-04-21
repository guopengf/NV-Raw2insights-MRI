import numpy as np
import sys
p = sys.argv[1]
d = np.load(p, allow_pickle=True)
for k in d.files:
    arr = d[k]
    if arr is None or getattr(arr, "ndim", 0) == 0:
        print(k, "scalar/None:", arr)
        continue
    nan = bool(np.isnan(arr).any())
    inf = bool(np.isinf(arr).any())
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"{k} shape={arr.shape} dtype={arr.dtype} ALL NaN/Inf")
        continue
    frac_nan = np.isnan(arr).mean()
    print(
        f"{k} shape={arr.shape} dtype={arr.dtype} "
        f"min={finite.min():.4g} max={finite.max():.4g} mean={finite.mean():.4g} "
        f"nan={nan} inf={inf} frac_nan={frac_nan:.3g} frac_zero={(arr == 0).mean():.3g}"
    )
