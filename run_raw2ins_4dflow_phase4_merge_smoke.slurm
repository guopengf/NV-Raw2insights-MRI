#!/bin/bash
#SBATCH --job-name=raw2ins_phase4_smoke
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/home/pengfeig/workspace/slurm-logs/%j_raw2ins_phase4_smoke.out

set -euo pipefail

export HF_HOME=/workspace/code/NV-Raw2insights-MRI-fork/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --container-image ~/workspace/cache/conda_raw2insights.sqsh \
    --container-mounts=${HOME}/.netrc:${CONTAINER_HOME:-/root}/.netrc,/lustre/fsw/portfolios/healthcareeng/users/pengfeig:/workspace,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData:/data/CMRx4DFlow2026-ChallengeData \
    --no-container-mount-home \
    --container-remap-root \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        source ~/miniconda3/bin/activate
        set +u
        source ~/.bashrc
        set -u
        conda activate nv-raw2insights-mri
        cd /workspace/code/NV-Raw2insights-MRI-fork

        python scripts/tools/check_4dflow_phase4_merge.py --device cuda
    '
