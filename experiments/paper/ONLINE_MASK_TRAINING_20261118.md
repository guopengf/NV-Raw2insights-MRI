# Online 4D Flow Sampling: 2026-11-18

The experiment training loaders directly call `generate()` in the user-specified
`final_mask_suite_20261118/code/mask_suite.py`. Initially read from the cy136 share,
the unchanged runtime files and license are now bundled for cluster deployment.
The source suite and raw dataset are not modified. No mask files are needed in the dataset.

## Behavior

- All 11 experiment configs enable `paper.online_undersampling`. Disable it to retain
  the experiment loader's previous per-window kt-Gaussian behavior.
- Draw one uniform family (`gaussian`, `pr4d`, `pseudo`, `uih`) and one nominal real R
  in [10, 50] per case per epoch. `rate_mode: "integer"` uses all 41 integer settings
  with the default bounds. Seeds derive from the experiment seed, case ID and epoch.
- The time-varying methods keep native [T,Z,Y] and use absolute frame indices.
  UIH stays native [Z,Y] and is broadcast only when forming the input window.
  Four encodings, coils and the fully sampled readout share the selected plane.
  The unmasked full k-space remains the supervised target source.
- Preserve the requested fractional R for sampling and continuous conditioning.
  The Restormer label embedding and CUT table use their nearest existing bins.
  Our paper model shares its mask-conditioning slot across methods to keep checkpoint
  shapes; the real binary mask drives DC. The FlowVN adapter matches the real dtype
  of its reconstruction tensor to prevent an unintended float64 promotion.
- Cache binary masks by case within the epoch in worker RAM. Reconstructed cache
  entries are deterministic across workers and eviction. Neural training writes the
  per-case method, requested/actual R, budget, seed and source fingerprints to
  `sampling/epoch_*.json`; solver smoke reports record the same fields.
- This changes the paper experiment training and solver Train-window checks,
  not the historical `scripts/train.py` data backend or fixed whole-case evaluation protocol.

## Verification

- 46 tests passed, including both two-worker loader cases. Checks cover native shapes,
  seed changes at fixed R, exact budget, dynamic-frame differences, shared UIH,
  absolute frame/window consistency, all 41 integer settings, and the disabled-mode fallback.
- Real Train case `Center012_P000203_Aorta`, native k-space [4,25,5,19,96,88]:
  direct loader checks exercised all four families from the shared schedule. Dynamic
  masks were [25,19,96] and UIH was [19,96]. Overlapping windows and E/coil/x
  broadcasting matched the same case mask.
- All nine neural configs completed one optimizer step on that case, including CUT's
  five-window accumulation; CG-SENSE and CS-TV completed one Train reconstruction window.
  Their common epoch-0 mask was UIH at requested R=38.1089094632, with 47 points and
  actual grid R=38.8085106383. CG reached its 30-iteration cap before tolerance in
  all 60 subproblems; this is a workflow check, not a convergence result.
- No formal training or held-out metrics were run.

Reports: `outputs/paper/online_mask_check/runs/<experiment>/smoke/report.json`.
Ready-to-run NAS configs: `outputs/paper/virtual_dataset_online/configs/`.
They are separate from the historical NAS configs and outputs.
See README for configuration and existing training commands.
