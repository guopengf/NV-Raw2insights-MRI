<!-- # NV-Raw2Insights-MRI

[![License](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-NVIDIA%20Open%20Model-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/nvidia/NV-Raw2Insights-MRI)
[![Paper](https://img.shields.io/badge/arXiv-2512.17137-red.svg)](https://arxiv.org/abs/2512.17137)

Universal MRI reconstruction from undersampled k-space. A single model handles diverse protocols, anatomies, contrasts, and acceleration factors without task-specific fine-tuning.

<p align="center">
<img width="900" alt="NV-Raw2Insights-MRI" src="https://github.com/user-attachments/assets/27ce9dea-c592-4dd5-b984-38542e1cf8e6" />
</p>

## Overview

NV-Raw2Insights-MRI is built on the Scalable Deep Unrolled Model (SDUM) framework. It combines a Restormer-based cascaded unrolled architecture with learned coil sensitivity estimation, sampling-aware weighted data consistency, and universal conditioning on protocol metadata. Trained on heterogeneous data from CMRxRecon2024, CMRxRecon2025, and fastMRI brain datasets, a single model achieves state-of-the-art results across cardiac, brain, and knee MRI reconstruction.

This project was conducted by NVIDIA in collaboration with the [CMRxRecon Team](https://github.com/CmrxRecon), [Fudan University](https://hupi.fudan.edu.cn/en/), and [Johns Hopkins University](https://profiles.hopkinsmedicine.org/provider/shanshan-jiang/2777746).

## News

- **[March 2026]** — Released NV-Raw2Insights-MRI as part of the NVIDIA MedTech Open Models
- **[February 2026]** — Achieved 1st place across all four tracks in the [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) without task-specific fine-tuning

## Model Variants

| Model | Cascades | Parameters | HuggingFace | License |
|-------|:--------:|:----------:|-------------|---------|
| [NV-Raw2Insights-MRI-Small](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 6 | 230M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Base](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 18 | 760M | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| [NV-Raw2Insights-MRI-Large](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | 34 | 1.4B | [Download](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

Checkpoints are automatically downloaded from [HuggingFace](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) when not provided locally.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i example \
  -o outputs/example_output_base
```

For multi-GPU inference:

```bash
torchrun --nproc_per_node=8 scripts/inference.py \
  -c configs/nv_raw2insights_mri_base.json \
  -i /path/to/input \
  -o /path/to/output
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup](docs/setup.md) | Full installation guide |
| [Inference](docs/inference.md) | Inference options, configs, multi-GPU |
| [Training](docs/training.md) | Training and fine-tuning guide |

## Performance

### Model Scaling (PSNR vs cascade depth)

| Cascades (T) | PSNR (dB) | Parameters |
|:------------:|:---------:|:----------:|
| 1 | 28.73 | 42M |
| 3 | 30.21 | 126M |
| 6 | 32.09 | 253M |
| 10 | 32.54 | 422M |
| 18 | 33.18 | 759M |

### Inference Compute (per slice, NVIDIA H100, T=18)

| Input Size | Time (s) | Memory (GB) |
|:----------:|:--------:|:-----------:|
| 128x128 | 0.32 | 4.78 |
| 256x256 | 1.03 | 6.07 |
| 256x512 | 2.06 | 7.98 |
| 328x512 | 2.67 | 9.26 |
| 328x640 | 3.30 | 9.62 |
| 328x768 | 3.97 | 10.83 |

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Model weights | [NVIDIA Open Model License](LICENSE.weights) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Citation

```bibtex
@article{wang2025sdum,
  title={SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction},
  author={Wang, Puyang and Guo, Pengfei and Chai, Keyi and Zhou, Jinyuan and Xu, Daguang and Jiang, Shanshan},
  journal={arXiv preprint arXiv:2512.17137},
  year={2025}
}
```

Please also cite the [CMRxRecon dataset](https://www.synapse.org/Synapse:syn59814210/wiki/) papers.

## Resources

- [SDUM Paper](https://arxiv.org/abs/2512.17137) — arXiv
- [HuggingFace Model](https://huggingface.co/nvidia/NV-Raw2Insights-MRI) — Weights and model card
- [CMRxRecon2025 Challenge](https://www.synapse.org/Synapse:syn59814210/wiki/634966) — Benchmark -->

# 4D Flow Aorta MRI Finetuning Usage

This branch adapts NV-Raw2insights-MRI for 4D Flow Aorta MRI reconstruction.

The expected raw data layout is:

```text
/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/
  Center007/
    GE_30T_Architect/
      P076/
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
run `generate_4dflow_ktgaussian_train.py` to generate undersampled kspace and masks:
```
python scripts/generate_4dflow_ktgaussian_train.py   --root path/to/TrainSet/Aorta   --accelerations 10,20,30,40,50   --overwrite
found 138 patients under /data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/TrainSet/Aorta
```

The 4D Flow k-space shape is expected to be:

```text
(enc, t, coil, kz, ky, kx)
```

This code uses a 1D centered IFFT along `kx`, then treats `x` like the slice dimension used by the original model:

```text
(enc, t, coil, kz, ky, kx)
-> IFFT along kx
(enc, t, coil, kz, ky, x)
-> transpose
(enc, t, x, coil, kz, ky)
```

Each velocity encoding `enc` is split into separate training samples, so the model still sees the original 5D-style input:

```text
(t, x, coil, kz, ky)
```

## Important

Do not write outputs into the raw data directory.

Avoid using any output path under:

```text
/data/CMRx4DFlow2026-ChallengeData/
```

Use a workspace, scratch folder, or experiment folder instead, for example:

```text
outputs/4dflow/4dflow_finetune_ckpts/
outputs/
```

## Config

The 4D Flow config is:

```text
configs/nv_raw2insights_mri_base_4dflow.json
```

Important fields:

```json
{
  "data_path_train": [
    "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/"
  ],
  "data_path_val": [
    "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/"
  ],
  "is_4dflow_aorta": true,
  "use_external_csm": true,
  "train_mask_types": ["fixed"],
  "val_mask_types": ["fixed"],
  "four_dflow_accelerations": [10, 20, 30, 40, 50],
  "four_dflow_encodings": [0, 1, 2, 3]
}
```

Currently `data_path_train` and `data_path_val` point to the same dataset. This is useful for pipeline testing, but it is not an independent validation split.

## Training

### Multi-GPU Training

Example: use physical GPUs 1 and 3.

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow.json
```

Inside PyTorch, these visible GPUs are remapped to:

```text
cuda:0 -> physical GPU 1
cuda:1 -> physical GPU 3
```

### Disable W&B Completely

If you do not want W&B at all:

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow.json
```

### Single-GPU Training

Set `"ddp": false` in the config, then run:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow.json
```

## Coilmap Check

Use this script to verify whether `coilmap.mat` produces a reasonable coil-combined image.

```bash
python scripts/check_4dflow_coilmap_combine.py \
  --patient-dir "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P076/" \
  --enc 0 \
  --frame 0
```

By default, the PNG is saved to the workspace-local folder:

```text
outputs/coilmap_checks/
```

Example output name:

```text
outputs/coilmap_checks/P076_coilmap_check_enc0_t0_x64.png
```

The script does not modify the `.mat` files.

## Optional: Generate 4D Flow JSON Manifests Manually

Training can generate manifests automatically inside the experiment folder. If you want to generate them manually:

```bash
python scripts/create_4dflow_aorta_json.py \
  --root "/data/CMRx4DFlow2026-ChallengeData/R1R2/TaskR1R2/ValidationSet/Aorta/" \
  --out "outputs/4dflow_jsons"
```

Each generated JSON contains:

```json
{
  "kspace": "path/to/kdata_ktGaussian10.mat",
  "target_kspace": "path/to/kdata_full.mat",
  "mask": ["path/to/usmask_ktGaussian10.mat"],
  "mask_type": "ktGaussian10",
  "acquisition": "Flow2d",
  "encoding_idx": 0,
  "is_4dflow": true,
  "coilmap": "path/to/coilmap.mat"
}
```

## Submission Export and Evaluation

`scripts/run_4dflow_inference.py` sensitivity-combines multi-coil complex output by default, so its organized
`final` tree is directly exportable. Use `--preserve-multicoil-output` only when coil-resolved output is needed for
debugging. Existing coil-resolved runs must first be converted with `scripts/tools/fix_4dflow_recon_coil_dim.py`.

Export the per-encoding complex `.mat` reconstructions to the official sparse NPZ submission tree with:

```bash
python scripts/tools/export_4dflow_submission.py \
  --recon-root "/path/to/inference/R1R2/final/Aorta" \
  --data-root "/path/to/ChallengeData/TaskR1&R2/ValidationSet/Aorta" \
  --out-root "/path/to/inference/R1R2/submission" \
  --task TaskR1R2 \
  --split ValidationSet \
  --anatomy Aorta \
  --recon-layout yzxt \
  --overwrite
```

The exporter expects one complex reconstruction per encoding and acceleration. It supports either flat inference files:

```text
Center007__GE_30T_Architect__P076__ktGaussian10__enc0.mat
```

or organized files:

```text
Center007/GE_30T_Architect/P076/kdata_ktGaussian10_enc0_recon.mat
```

It writes:

```text
TaskR1R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P076/img_ktGaussian10.npz
```

The output is multiplied by the official `segmask.mat`, matching the challenge sparse-submission format. By default the result is a directory tree, not a zip file. Add `--zip` only if a `Submission.zip` archive is needed.

For TaskS2, run the exporter once per anatomy folder, changing `--recon-root`, `--data-root`, and `--anatomy` accordingly.

Evaluate an exported validation submission against local GT with:

```bash
python scripts/tools/evaluate_4dflow_submission.py \
  --submission-root "/path/to/inference/R1R2/submission" \
  --gt-root "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT" \
  --eval-code-dir "/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode" \
  --out-csv "/path/to/inference/R1R2/submission/eval_metrics.csv" \
  --out-json "/path/to/inference/R1R2/submission/eval_summary.json" \
  --task TaskR1R2 \
  --include-complex-diff \
  --skip-errors
```

`--gt-root` should point to the directory that contains the task folders, for example `TaskR1R2/ValidationSet/Aorta/...`. The evaluator records per-case metrics in the CSV and summary means in the JSON.

## Modified / Added Scripts

### `scripts/readers.py`

Modified `CMRxReconReader` to support 4D Flow Aorta JSON files.

For 4D Flow data, it reads:

```text
kdata_ktGaussianXX.mat  -> undersampled input
kdata_full.mat          -> fully sampled GT
usmask_ktGaussianXX.mat -> fixed mask condition
coilmap.mat             -> external coil sensitivity maps
```

It does not generate masks from full k-space.

### `scripts/transforms.py`

Added 4D Flow conversion utilities:

```text
raw_4dflow_to_hybrid()
raw_4dflow_mask_to_hybrid()
raw_4dflow_coilmap_to_hybrid()
```

The fixed-mask branch now uses the already undersampled k-space as input and the full k-space as target.

The coilmap is converted into the same layout as the model input and passed forward as `sensitivity_maps`.

### `scripts/models/restormer/restormer.py`

Modified the Restormer MRI model to support external coil sensitivity maps.

When:

```json
"use_external_csm": true
```

the model uses `coilmap.mat` from the patient folder instead of estimating coil maps internally.

The coil sensitivity model module is kept in the architecture for pretrained weight compatibility, but its parameters are frozen when external coil maps are used.

### `scripts/models/latent_recon.py`

Modified the flow wrapper so `sensitivity_maps` can be passed through the cascade/flow model.

### `scripts/train.py`

Modified training to support `is_4dflow_aorta`.

When enabled, training scans:

```text
Center*/Scanner*/Patient*/
```

and automatically creates JSON manifests under the experiment output folder:

```text
<exp_dir>/<exp>/jsons_train/
<exp_dir>/<exp>/jsons_val/
```

It also passes external coil maps into the model during training and validation.

### `scripts/train_utils.py`

Disabled incompatible CMRxRecon k-space augmentation for 4D Flow Aorta training.

### `scripts/check_4dflow_coilmap_combine.py`

New utility script for visually checking whether `coilmap.mat` works for SENSE-style coil combination.

Default output:

```text
outputs/coilmap_checks/
```

### `scripts/create_4dflow_aorta_json.py`

New utility script to manually generate JSON manifests for 4D Flow Aorta data.

### `scripts/path_safety.py`

New helper that can prevent accidental writes into raw challenge data directories.

It is intended to prevent writing outputs under paths that look like:

```text
ChallengeData
ValidationSet
TrainingSet
Aorta
```

### `scripts/run_4dflow_inference.py`

Utility script for running inference over 4D Flow Aorta cases and organizing results.

It creates temporary JSON inputs, runs `scripts/inference.py`, sensitivity-combines multi-coil output, then copies
exporter-ready complex reconstructions into an organized output folder. Pass `--preserve-multicoil-output` to retain
the legacy coil-resolved output instead.

### `scripts/tools/fix_4dflow_recon_coil_dim.py`

Legacy utility for sensitivity-combining coil-resolved complex `.mat` files created by older inference runs or by
`run_4dflow_inference.py --preserve-multicoil-output`.

### `scripts/reorganize_4dflow_outputs.py`

Utility script for reorganizing flat inference outputs into:

```text
Center/Scanner/Patient/
```

### `scripts/tools/export_4dflow_submission.py`

Utility script for converting complex 4D Flow inference `.mat` files into the official CMRx4DFlow sparse NPZ submission format.

### `scripts/tools/evaluate_4dflow_submission.py`

Utility script for computing validation metrics from an exported submission tree using the official EvaluationCode utilities.

### `scripts/visualize_4dflow_gt_img4ranking.py`

Utility script for visualizing full 4D Flow GT k-space in an img4ranking-like style.

### `scripts/vis_tools/`

Additional visualization and evaluation helpers:

```text
save_kspace_frames.py
save_recon_frames.py
show_zf_gt_recon_grid.py
evaluate_case.py
```

These are optional debugging tools and are not required for training.

## Notes

- The original pretrained model weights can still be loaded because the model input shape is kept compatible.
- `enc` is split into the sample dimension instead of changing the model architecture.
- The model uses existing `coilmap.mat` instead of estimating coil maps.
- The fixed masks are read from `usmask_ktGaussianXX.mat`.
- The undersampled inputs are read directly from `kdata_ktGaussianXX.mat`.
- The training target is always `kdata_full.mat`.
