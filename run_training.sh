#! /bin/bash

export HF_HOME=/workspace/code/NV-Raw2insights-MRI-fork/cache

WANDB_MODE=offline torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow_pg.json
