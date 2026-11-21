# Final Mask Suite Runtime

This is a byte-for-byte runtime snapshot of the final mask suite released on
cy136 on 2026-11-18. It includes the five Python modules, both fitted profiles,
and the upstream PR4D MIT license. Audit scripts, outputs, and data are excluded.

The training config resolves `code/mask_suite.py` relative to the repository
root and calls `generate()` in memory. Keep the profiles at this directory's
root. See the paper experiment README for rate and mask settings.
