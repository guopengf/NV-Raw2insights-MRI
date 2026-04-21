# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NV-Raw2Insights-MRI implements the **Scalable Deep Unrolled Model (SDUM)** for universal MRI reconstruction from undersampled k-space. A single model handles diverse protocols, anatomies, contrasts, and acceleration factors. Built on MONAI and PyTorch with distributed training support (torchrun/DDP).

Three model variants: Small (6 cascades, 230M params), Base (18 cascades, 760M), Large (34 cascades, 1.4B). Checkpoints auto-download from HuggingFace (`nvidia/NV-Raw2insights-MRI`).

## Common Commands

### Inference
```bash
# Single GPU
python scripts/inference.py -c configs/nv_raw2insights_mri_base.json -i example -o outputs/example_output_base

# Multi-GPU
torchrun --nproc_per_node=8 scripts/inference.py -c configs/nv_raw2insights_mri_base.json -i /path/to/input -o /path/to/output
```

### Training
```bash
# Multi-GPU (recommended)
torchrun --nproc_per_node=8 --nnodes=1 scripts/train.py --config configs/nv_raw2insights_mri_base.json

# Validation only
torchrun --nproc_per_node=8 scripts/train.py --config configs/nv_raw2insights_mri_base.json --val
```

### Dataset Preparation
```bash
python scripts/create_cmrxrecon_dataset.py \
  --source_dir /path/to/ChallengeData \
  --mask_dir /path/to/ChallengeData \
  --output_dir dataset/CMRxRecon2025/ChallengeDataTrain \
  --training_set
```

### Visualization
```bash
python scripts/visualize_mat.py outputs/example_output_base/val_img4ranking -o outputs/figs
```

### Linting
```bash
./lint.sh           # check only (flake8, black, isort)
./lint.sh --fix     # auto-fix formatting
```

## Architecture

### Model Pipeline

The reconstruction pipeline is a **cascaded unrolled architecture** with flow-matching:

1. **Entry point**: `scripts/models/latent_recon.py::create_mri_recon_model()` — constructs the full model from a config. When `flow=True` (default for all configs), builds `Flow_SkipConnected_MRI_Recon` wrapping `Cascaded_SkipConnected_MRI_Recon`.

2. **Cascade structure**: Each cascade is a `restormer_mri` module (`scripts/models/restormer/restormer.py`) that includes:
   - Restormer-based image reconstruction network (Transformer blocks with transposed attention)
   - Learned coil sensitivity estimation (CSM via `CoilSensitivityModel_DCAE` in `scripts/models/varnet.py`, first cascade only)
   - Sampling-aware weighted data consistency
   - Universal conditioning on mask type, acceleration factor, and acquisition type via label/timestep embeddings

3. **Flow wrapper**: `Flow_SkipConnected_MRI_Recon` runs the cascade stack for `num_steps` iterations (flow-matching steps), passing skip connections and sensitivity maps between iterations.

4. **Gradient checkpointing**: Enabled during training via `torch.utils.checkpoint.checkpoint` for memory efficiency.

### Data Pipeline

- **Readers** (`scripts/readers.py`): `CMRxReconReader` (JSON descriptors → .mat files), `FastMRIReader` (.h5 files), `CestMRIReader`
- **Transforms** (`scripts/transforms.py`): k-space masking (random, equispaced, fixed, kt-uniform/gaussian/radial), data augmentation (flip, k-space shift, phase shift, contrast, resize), rearrangement and normalization
- **Data utils** (`scripts/mri_data/data_utils.py`): MRI data rearrangement, postprocessing, k-space cropping
- **kt-Sampling** (`scripts/mri_data/ktSampling.py`): Generates sampling masks (kt-Gaussian, kt-Radial, Uniform)

### Training Flow (`scripts/train.py`)

- Uses MONAI's `Dataset`/`CacheDataset` and `DataLoader` with `DistributedSampler`
- Optimizer: Muon (default) or AdamW, configured in `scripts/train_utils.py::get_optimizer()`
- LR schedule: cosine with warmup (manual via `adjust_learning_rate()` in `scripts/utils.py`)
- Loss: SSIM-based (configurable via `loss_type`)
- Logging: TensorBoard + Weights & Biases
- Auto-resume: detects existing checkpoints in `exp_dir/exp/` and continues training

### Key Data Conventions

- K-space tensors use shape `[t, z, c, y, x]` (time, slices, coils, height, width)
- Complex data stored as last-dim real/imag pairs: `[..., 2]`
- Windowed temporal input: `windowed_input()` in `scripts/utils.py` creates sliding windows of `num_frames` frames per sample
- Output format: `.mat` files with `img4ranking` key under `val_img4ranking/` directory

### Configuration

JSON configs in `configs/` control everything: model architecture, data paths, training hyperparameters, mask types, and logging. The `Config` class in `scripts/utils.py` loads JSON into attribute-accessible objects with defaults.

## Code Style

- Line length: 120 characters
- Formatting: black + isort
- Linting: flake8 with MONAI-style ignores (see `.flake8`)
- All scripts run from the repo root with `scripts/` as the working directory for imports (scripts use bare imports like `from utils import *`, `from models.latent_recon import ...`)

## Contributing

All commits require DCO sign-off (`git commit -s`).
