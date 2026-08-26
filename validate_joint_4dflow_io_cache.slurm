#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=00:45:00
#SBATCH --job-name=joint4d-io-cache-debug
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-joint4d-io-cache-debug.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATA_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData

srun \
    --container-image "$IMAGE" \
    --container-mounts="/home/pengfeig/.netrc:/root/.netrc,${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork
        export WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        /root/miniconda3/envs/nv-raw2insights-mri/bin/torchrun \
            --standalone --nproc_per_node=8 \
            scripts/train.py --debug \
            --config configs/benchmarks/nv_raw2insights_mri_small_4dflow_joint_batch_io_cache_debug_pg.json
    "
