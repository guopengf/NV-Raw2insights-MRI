#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --array=0-111%4
#SBATCH --job-name=4dflow-build-gt
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%A_%a-4dflow-build-gt.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
RUN_ROOT=${RUN_ROOT:-${REPO}/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z}
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python

test -f "$RUN_ROOT/gt_plan.json"
mkdir -p "$RUN_ROOT/gt" "$RUN_ROOT/logs"
export NVIDIA_VISIBLE_DEVICES=void

srun \
    --container-image "$IMAGE" \
    --container-env=NVIDIA_VISIBLE_DEVICES \
    --container-mounts="${USER_ROOT}:/workspace,${DATASET_ROOT}:/datasets" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        '$PY' scripts/tools/build_4dflow_validation_gt.py \
            --run-case \
            --plan /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/gt_plan.json \
            --gt-root /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/gt \
            --case-index '${SLURM_ARRAY_TASK_ID}' \
            --workers 8
    "

cp "/home/pengfeig/workspace/slurm-logs-agent/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-4dflow-build-gt.out" "$RUN_ROOT/logs/" 2>/dev/null || true

