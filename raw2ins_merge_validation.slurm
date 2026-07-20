#!/bin/bash
#SBATCH --job-name=raw2ins-merge-validation
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/logs/%j-raw2ins-merge-validation.out

srun --kill-on-bad-exit=1 \
    --container-image ~/workspace/cache/conda_raw2insights.sqsh \
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
        python -m py_compile \
            scripts/mri_data/coil_combine.py \
            scripts/inference.py \
            scripts/run_4dflow_inference.py \
            scripts/tools/fix_4dflow_recon_coil_dim.py \
            scripts/tools/export_4dflow_submission.py
        python scripts/inference.py --help >/dev/null
        python scripts/tools/validate_4dflow_inference_export.py
    '
