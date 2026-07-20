---
name: nv-raw2insights-4dflow-mri
description: Use when adapting, training, debugging, running inference, or evaluating NVIDIA-Medtech NV-Raw2insights-MRI for 4D Flow Aorta MRI data with 6D k-space, external coil maps, ktGaussian undersampling, Flow4d conditioning, and organized Center/Scanner/Patient outputs.
---

# NV-Raw2insights 4D Flow MRI

Use this skill for work on the 4D Flow Aorta fork of `NV-Raw2insights-MRI`.
The code is based on `NVIDIA-Medtech/NV-Raw2insights-MRI`, adapted for 4D Flow Aorta MRI.

## Data Model

4D Flow Aorta raw k-space is 6D:

```text
(enc, t, coil, kz, ky, kx)
```

Meaning:

- `enc`: velocity encoding, usually 4 encodings.
- `t`: cardiac time frame.
- `coil`: receiver coil.
- `kz`, `ky`: accelerated phase-encoding directions.
- `kx`: fully sampled readout direction.

The adaptation does:

```text
IFFT along kx
(enc, t, coil, kz, ky, kx)
-> (enc, t, x, coil, kz, ky)
```

Then each JSON selects one `encoding_idx`, keeping a singleton enc dimension:

```text
(1, t, coil, kz, ky, kx)
```

After transforms, the model sees the original 5D-style input:

```text
(t, x, coil, kz, ky)
```

So `enc` is not merged with `t`; it is split into separate JSON cases. With four encodings and five accelerations, each patient produces:

```text
4 encodings * 5 accelerations = 20 JSON files
```

## Required Patient Files

Expected patient folder:

```text
Aorta/CenterXXX/ScannerName/PYYY/
```

Required files:

```text
coilmap.mat
kdata_full.mat
kdata_ktGaussian10.mat
kdata_ktGaussian20.mat
kdata_ktGaussian30.mat
kdata_ktGaussian40.mat
kdata_ktGaussian50.mat
usmask_ktGaussian10.mat
usmask_ktGaussian20.mat
usmask_ktGaussian30.mat
usmask_ktGaussian40.mat
usmask_ktGaussian50.mat
```

Typical roots:

```text
/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/
/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta/
```

If `TrainSet/Aorta` does not contain undersampled `kdata_ktGaussian*.mat`, the generated train manifest will be empty and training will fail with `ZeroDivisionError: division by zero`.

## Raw Data Safety

Do not write anything under:

```text
/data/CMRx4DFlow2026-ChallengeData
```

Treat raw challenge data as read-only.

Safe output locations are repo-local `outputs/` or a separate checkpoint/output folder such as:

```text
outputs/4dflow/4dflow_finetune_ckpts/
```

Never save PNGs, metrics, generated JSONs, or recon outputs into patient data folders.

`scripts/check_4dflow_coilmap_combine.py` defaults to repo-local output, not patient folders.

## Important Code Conventions

### External Coil Maps

Use existing `coilmap.mat`; do not estimate coil maps inside the model for this 4D Flow path.

Key config:

```json
"use_external_csm": true,
"use_csm": true,
"use_tau_csm": false,
"use_acs_region": false
```

The model keeps coil-map-related modules for pretrained weight compatibility but uses provided `sensitivity_maps` when available.

### Flow4d Conditioning

Use `Flow4d`, not `Flow2d`, throughout:

- config `acq_types`
- generated manifests
- reader default for 4D Flow JSON
- inference wrapper default acquisition
- model default acquisition list

Use acceleration classes:

```json
[10, 20, 30, 40, 50]
```

and align:

```json
"accelerations": [10.0, 20.0, 30.0, 40.0, 50.0],
"center_fractions": [0.0, 0.0, 0.0, 0.0, 0.0],
"four_dflow_accelerations": [10, 20, 30, 40, 50],
"accelerations_for_def_model": [10, 20, 30, 40, 50]
```

Changing from old `8/16/24 + Flow2d` to `10/20/30/40/50 + Flow4d` changes label conditioning. Retest a small run before large-cluster training.

### Fast IO Reader

The important IO optimization is in `scripts/readers.py`:

- Select `encoding_idx` before loading the full `.mat` array.
- Use h5py dataset slicing:

```python
selection = (slice(enc_idx, enc_idx + 1), Ellipsis)
dataset[selection]
```

This avoids reading all encodings for every JSON.

For 4 encodings, this can reduce k-space IO roughly 4x per JSON.

### DataLoader IO Settings

Use these DataLoader settings in `scripts/train.py`:

```python
pin_memory=True
persistent_workers=args.num_workers > 0
in_order=False
```

Typical large-node config:

```json
"num_workers": 8,
"batch_size": 8
```

For debugging DataLoader hangs, set:

```json
"num_workers": 0
```

## Config Template

Main config:

```text
configs/nv_raw2insights_mri_base_4dflow.json
```

Important values:

```json
{
  "model_variant": "nv_raw2insights_mri_base",
  "data_path_train": ["/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/"],
  "data_path_val": ["/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta/"],
  "fixed_mask_types": ["ktGaussian"],
  "train_mask_types": ["fixed"],
  "val_mask_types": ["fixed"],
  "accelerations": [10.0, 20.0, 30.0, 40.0, 50.0],
  "center_fractions": [0.0, 0.0, 0.0, 0.0, 0.0],
  "acq_types": ["BlackBlood", "Cine", "Flow4d", "LGE", "Mapping", "Perfusion", "T1rho", "T1w", "T2w"],
  "is_4dflow_aorta": true,
  "use_external_csm": true,
  "four_dflow_accelerations": [10, 20, 30, 40, 50],
  "four_dflow_encodings": [0, 1, 2, 3],
  "accelerations_for_def_model": [10, 20, 30, 40, 50],
  "batch_size": 1,
  "num_workers": 1,
  "num_samples_per_case": 8,
  "ddp": true,
  "val": false
}
```

For large nodes, increase `batch_size` and `num_workers` only after a small validation run.

## Training

Use the original pretrained checkpoint for fine-tuning. The code auto-resumes from:

```text
<exp_dir>/<exp>/<model_filename>
```

If that file does not exist, it loads the base pretrained checkpoint resolved by `model_variant`.

Two-GPU local/server run on GPUs 1 and 3:

```bash
cd /localhome/zhanghs/NV-Raw2insights-MRI

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow.json
```

Eight-GPU node:

```bash
WANDB_MODE=offline torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow_pg.json
```

If W&B prompts interactively, use:

```bash
WANDB_MODE=offline
```

or disable it if supported in the environment:

```bash
WANDB_DISABLED=true
```

## Training Log Interpretation

Example:

```text
269/320, 8/8, lr: ..., train_loss: 1.6949, train_step_loss: 2.1963
```

Meaning:

- `269/320`: current JSON/case index out of total training JSONs for the rank.
- `8/8`: inner sample progress for the current JSON/case.
- `8` usually comes from `num_samples_per_case`, not from time frames.
- `train_step_loss`: current optimizer step loss, DDP-averaged when DDP is enabled.
- `train_loss`: running average over the current epoch.

If `adaptive_batch_size=true`, the display may not show `1/8, 2/8, ..., 8/8`.
It may jump by larger chunks, e.g. `4/8` or `8/8`, because one optimizer step can include multiple samples.

## DDP/NCCL Timeout Debugging

If log shows:

```text
Watchdog caught collective operation timeout: OpType=ALLREDUCE
```

and points near:

```python
dist.all_reduce(num_samples, op=dist.ReduceOp.MIN)
```

then one rank reached a sync point while another rank was stuck or slow.

Common causes:

- slow `.mat` read for one rank
- very large case shape on one rank
- DataLoader worker hang
- checkpoint or network filesystem IO stall

It is not necessarily validation. If `val_interval=40` and the timeout happens at epoch 7, it is in training, not validation.

Debug steps:

1. Resume from the latest checkpoint.
2. Set `"num_workers": 0` to expose DataLoader issues.
3. Add rank/file logging around the train loop if needed.
4. Increase NCCL timeout only after finding the slow step.

## Checkpoints

The always-latest checkpoint is:

```text
<exp_dir>/<exp>/<model_filename>
```

Periodic checkpoints look like:

```text
<model_filename>_epoch5.pt
<model_filename>_epoch10.pt
```

If training has passed epoch 10, the non-epoch-suffixed `.pt` is newer than `_epoch10.pt`.

## Inference

Use the wrapper:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python scripts/run_4dflow_inference.py \
  --case-root "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta/Center007/GE_30T_Architect" \
  --json-dir "outputs/infer/jsons" \
  --tmp-out "outputs/infer/tmp" \
  --final-out "outputs/infer/final" \
  --config configs/nv_raw2insights_mri_base_4dflow.json \
  --ckpt "outputs/4dflow/4dflow_finetune_ckpts/base_ft_4dflow_encbatch/nv_raw2insights_mri_base_ft_4dflow_encbatch.pt" \
  --encoding-idx 0 \
  --accelerations 10,20,30,40,50
```

For challenge test folders without `kdata_full.mat`, use:

```bash
--targetless
```

The wrapper creates final organized outputs like:

```text
final/Center007/GE_30T_Architect/P076/kdata_ktGaussian10_enc0_recon.mat
```

Run the wrapper once for each encoding index `0,1,2,3`. It saves coil-combined complex real/imag output by default,
which preserves phase and is directly compatible with `scripts/tools/export_4dflow_submission.py`. Pass
`--preserve-multicoil-output` only for legacy/debug coil-resolved output.

## Evaluation: 2D XY/ZY SSIM

Use:

```text
scripts/vis_tools/evaluate_4dflow_recon_planes.py
```

For an organized `final/` folder:

```bash
python scripts/vis_tools/evaluate_4dflow_recon_planes.py \
  --data-root "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta" \
  --recon-root "/path/to/final" \
  --enc-idx 0 \
  --recon-layout xyzt \
  --out-root "outputs/4dflow_eval_vis_test_enc0"
```

Metrics only:

```bash
python scripts/vis_tools/evaluate_4dflow_recon_planes.py \
  --data-root "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta" \
  --recon-root "/path/to/final" \
  --enc-idx 0 \
  --recon-layout xyzt \
  --out-root "outputs/4dflow_eval_metrics_test_enc0" \
  --no-images
```

The script computes:

- per-slice/time xy SSIM
- per-x/time zy SSIM
- summary CSV
- optional GT/recon/error PNGs under repo-local output

## Evaluation: PCMRA / 3D SSIM

Use:

```text
scripts/vis_tools/evaluate_4dflow_pcmra_folder.py
```

Command:

```bash
python scripts/vis_tools/evaluate_4dflow_pcmra_folder.py \
  --recon-root "/path/to/final" \
  --data-root "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TestSet/Aorta" \
  --out-csv "outputs/4dflow_pcmra_metrics.csv" \
  --recon-layout xyzt
```

This computes GT PCMRA following the notebook logic:

```python
kmean = temporal_mean(kdata_full)
img = direct_recon(kmean, coilmap)
pcmra = mean(abs(img), enc) * sqrt(sum(angle(img[1:] * conj(img[0])) ** 2, enc))
```

For recon, it groups:

```text
kdata_ktGaussianXX_enc0_recon.mat
kdata_ktGaussianXX_enc1_recon.mat
kdata_ktGaussianXX_enc2_recon.mat
kdata_ktGaussianXX_enc3_recon.mat
```

If recon files contain complex images, it computes true PCMRA. If they contain magnitude-only `img4ranking`, the script reports:

```text
mode=magnitude_mean_fallback
```

That fallback is only a sanity metric, not true phase-based PCMRA.

## Coilmap Sanity Check

Use:

```text
scripts/check_4dflow_coilmap_combine.py
```

The intent is to test whether `coilmap.mat` combines k-space images correctly.

Do not save outputs into patient folders. Keep outputs under repo-local `outputs/`.

## Mask / Undersampled Data Generation

Use only on copied/generated data, not original challenge data:

```text
scripts/generate_4dflow_ktgaussian_train.py
```

It can generate:

```text
usmask_ktGaussian*.mat
kdata_ktGaussian*.mat
```

It supports:

```bash
--num-shards
--shard-index
--writer h5py
--dry-run
```

It writes via temporary files and atomic replace, but still must not be run on raw challenge folders unless explicitly intended.

## Validation Philosophy

A useful pipeline sanity check:

1. Zero-shot base checkpoint on P076 gives low SSIM around 0.4-0.7.
2. Train on validation including P076.
3. Inference on P076 improves to around 0.9 SSIM after several epochs.

This proves the training code can overfit/converge and the data path is mostly correct.
It does not prove generalization.

Before large-cluster training, do a small holdout validation using cases not used in training.

Changing Flow2d/old acceleration labels to Flow4d/10-50 conditioning should be retested briefly before expensive runs.

## Common Failure Modes

### Empty Training Manifest

Symptom:

```text
ZeroDivisionError: division by zero
```

Likely cause: no training JSON files generated because the train folder lacks undersampled files.

Check:

```bash
find /path/to/Aorta -name "kdata_ktGaussian10.mat" | head
find /path/to/Aorta -name "kdata_full.mat" | head
ls <exp_dir>/<exp>/jsons_train | wc -l
```

### Shape Mismatch in Evaluation

If evaluating recon from Center008 using GT from Center007:

```text
Shape mismatch GT (...) vs recon (...)
```

Use `--data-root` so GT is inferred per case.

### 6D vs 7D Einops Error

Symptom:

```text
Wrong shape: expected 7 dims. Received 6-dim tensor.
```

Likely cause: external coilmap was passed without the expected batch dimension during rearrange.
Wrap sensitivity maps as a list/with batch dimension before `rearrange_mri_data`.

### Complex Output and PCMRA

The 4D Flow wrapper saves coil-combined complex real/imag output. Do not replace this save path with
`complex_abs(...)`; phase is required for PCMRA and the official flow metrics.

## Key Files

```text
configs/nv_raw2insights_mri_base_4dflow.json
configs/nv_raw2insights_mri_base_4dflow_pg.json
scripts/readers.py
scripts/transforms.py
scripts/train.py
scripts/train_utils.py
scripts/inference.py
scripts/run_4dflow_inference.py
scripts/create_4dflow_aorta_json.py
scripts/generate_4dflow_ktgaussian_train.py
scripts/check_4dflow_coilmap_combine.py
scripts/vis_tools/evaluate_4dflow_recon_planes.py
scripts/vis_tools/evaluate_4dflow_pcmra_folder.py
scripts/path_safety.py
```

## When Editing

- Preserve pretrained weight compatibility.
- Keep `Flow4d` conditioning.
- Keep acceleration labels aligned with actual masks.
- Use external coil maps when `use_external_csm=true`.
- Do not save to raw data paths.
- Prefer h5py slicing for large `.mat` arrays.
- Do not merge `enc` with `t`; split enc into JSON/batch instead.
- Use repo-local outputs for metrics, visualizations, and generated manifests.
