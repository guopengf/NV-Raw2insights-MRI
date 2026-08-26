#! /bin/bash

export HF_HOME=/workspace/code/NV-Raw2insights-MRI-fork/cache

# WANDB_MODE=offline torchrun --nproc_per_node=8 scripts/train.py \
#   --config configs/nv_raw2insights_mri_base_4dflow_pg.json

WANDB_MODE=offline torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json #configs/nv_raw2insights_mri_base_4dflow_3d_flowvn_multiplane_pg.json #configs/nv_raw2insights_mri_base_4dflow_mask_vaa_pg.json # configs/nv_raw2insights_mri_base_4dflow_mask_vaa_slab.json #configs/nv_raw2insights_mri_base_4dflow_slab_no_vaa.json