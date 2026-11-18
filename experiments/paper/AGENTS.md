# Paper Training And Full-Case Pipeline

Read `README.md` before changing this experiment package. If the user has supplied
the private task prompt / fixed 2026-11-15 DOCX locally in `inputs/`, read those too;
private planning documents are not included in the published code branch.
This package was based on `pengfeig/4dflow-finetune`, baseline SHA
`9504c29e50237d96eee89f3cf47693c64482bd2f`, and is published on
`haosen/baseline_ablation` in `guopengf/NV-Raw2insights-MRI`.
The cluster handoff uses the separately requested branch spelling
`haosen/baselin_ablation`; its setup and queue guidance are in CLUSTER_TRAINING.md.
The old root `agent.md` describes an earlier phase and is not the current experiment specification.

- This stage prepares training and runs only 1-3 optimizer steps on Train data.
- The 2026-11-17 follow-up also authorizes full-case inference/assembly and integration
  with the existing official-wrapper metrics. Keep these entries isolated here.
- Validate the new pipeline on a few Train cases only. Synthetic fixtures may be used
  for integration checks but must never be reported as real MRI/test validation.
- Do not run complete training, test evaluation, or select checkpoints using test data.
- Inference/evaluation CLI defaults to Train. Supporting explicit original test labels
  in code is not authorization to execute held-out tests during development.
- Use one fixed full-time kt-Gaussian mask per case/R/seed across methods; cache only
  outside datasets for the existing evaluation pipeline. Never resample independently
  for each inference window.
- For training and solver Train-window checks, use the cy136 final mask suite
  configured in paper.online_undersampling. Do not duplicate or edit its code.
  Draw one method and nominal R per case and epoch; share the full-time pattern
  across windows, E, coils and x. UIH stays 2D until the loader selects a window.
  Continuous R in [10,50] is the default; integer mode is an explicit alternative.
- Reconstruct every time and x position, keep all four encodings and complex phase,
  restore the common acquisition scale, and publish completion metadata last.
- Our final cascade repeats its center time prediction: export only genuine center
  time estimates and slide over all center times, never treat copies as new frames.
- CS-TV must operate on the complete spatial x extent, not independent x slabs.
- Reuse scripts/tools/evaluate_4dflow_submission.py::evaluate_arrays and supplied
  corrmap.npz. Never invoke dataset-side corrmap generation or change metric formulas.
- Source and destination dataset paths are read-only, especially `/media/Elements/4DFlow/`
  and `/media/fudan22/4DFlow/`. Apply `OutputGuard` before all writes, including symlinks.
- User supplied file_mapping(3).csv on 2026-11-17. Optional source_mapping resolves
  original target_file identities to NAS source_file reads only. Preserve logical
  directory, original rows, disk and dataset_group; never rewrite the final CSV.
- Build virtual indices with build_virtual_dataset.py; generated config/cache/results
  belong in outputs/paper/virtual_dataset, not NAS or /media. No mounts/symlinks needed.
- Protect both physical NAS sources (including resolved file symlinks) and logical
  destinations. Verify coverage/size/shape against the original CSV; no guessing.
- Virtual setup indexes all original labels but checks only 1-3 Train case headers;
  do not describe all indexed sources as physically validated or evaluate Test data.
- Preserve `train`, `ordinary_test`, `focused_test`; there is no validation split.
- Do not repartition, move, rename, convert in place, or write dataset-side caches.
- Raw layout is `[E,T,C,kz,ky,kx]`; IFFT raw kx, use x as slice, sample kz/ky only.
- Read VENC directions from each patient's params.csv. Do not assume the old Aorta order.
- Comparisons belong in `comparisons/` and pinned author checkouts in `vendor/`.
  Do not insert comparison implementations into `scripts/` or our backbone.
- FlowMoDL and CUT-4DFlow are pinned Git submodules, matching vendor_lock.json.
  Use bootstrap --fetch-authors after cloning; it handles empty submodule folders,
  avoids overwriting local source edits, and sparse-checks out CUT code, not weights.
- Do not publish source-mapping CSVs, local environments, raw MRI, weights or outputs.
  The checked-in final manifest/tree contain logical pseudonymous file indices only.
- A0-A4 use 6 cascades, S=3, T=5 and disable VAA. Follow the module staircase in README.
- PASS means the recorded check passed. Report exact shape, curriculum overrides,
  solver non-convergence, missing dependencies and untested full-resolution resource needs.
