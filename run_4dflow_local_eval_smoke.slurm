#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=4dflow-eval-smoke
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-eval-smoke.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
RUN_ROOT=${RUN_ROOT:-${REPO}/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z}
DATA_ROOT=/home/pengfeig/healthcareeng_monai/datasets/fudan/data/CMRx4DFlow2026-ChallengeData-20260908
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/gt"
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
        '$PY' -m unittest tests.test_4dflow_local_leaderboard
        '$PY' scripts/tools/build_4dflow_validation_gt.py \
            --prepare-plan \
            --data-root /datasets/fudan/data/CMRx4DFlow2026-ChallengeData-20260908 \
            --plan /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/gt_plan.json \
            --expected-cases 112
        '$PY' scripts/tools/build_4dflow_validation_gt.py \
            --run-case \
            --plan /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/gt_plan.json \
            --gt-root /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z/gt \
            --case-index 0 \
            --workers 8
    "

cp "/home/pengfeig/workspace/slurm-logs-agent/${SLURM_JOB_ID}-4dflow-eval-smoke.out" "$RUN_ROOT/logs/" 2>/dev/null || true

