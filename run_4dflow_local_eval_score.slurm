#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=4dflow-local-score
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-local-score.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
RUN_ROOT=${RUN_ROOT:-${REPO}/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z}
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
EVAL_CODE=${RUN_ROOT}/official/CMRx4DFlow2026/CMRx4DFlowReconDemo/Utils
METRICS=${RUN_ROOT}/metrics

mkdir -p "$METRICS" "$RUN_ROOT/corrmaps" "$RUN_ROOT/logs"
test -d "$EVAL_CODE"
test -d "$RUN_ROOT/submissions/epoch900"
test -d "$RUN_ROOT/submissions/epoch960"
export NVIDIA_VISIBLE_DEVICES=void

srun \
    --container-image "$IMAGE" \
    --container-env=NVIDIA_VISIBLE_DEVICES \
    --container-mounts="${USER_ROOT}:/workspace" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        RUN=/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z
        '$PY' scripts/tools/build_4dflow_validation_gt.py \
            --finalize --plan \"\$RUN/gt_plan.json\" --gt-root \"\$RUN/gt\" \
            --receipt \"\$RUN/gt_receipt.json\"
        '$PY' scripts/tools/evaluate_4dflow_submission.py \
            --submission-root \"\$RUN/submissions/epoch900\" \
            --gt-root \"\$RUN/gt\" \
            --eval-code-dir \"\$RUN/official/CMRx4DFlow2026/CMRx4DFlowReconDemo/Utils\" \
            --cache-dir \"\$RUN/corrmaps\" \
            --spatial-mode full \
            --use-venc \
            --include-complex-diff \
            --out-csv \"\$RUN/metrics/epoch900_full_venc.csv\" \
            --out-json \"\$RUN/metrics/epoch900_full_venc.summary.json\"
        '$PY' scripts/tools/verify_4dflow_leaderboard_parity.py \
            \"\$RUN/metrics/epoch900_full_venc.summary.json\" \
            --out \"\$RUN/metrics/epoch900_parity_receipt.json\"
        '$PY' scripts/tools/evaluate_4dflow_submission.py \
            --submission-root \"\$RUN/submissions/epoch960\" \
            --gt-root \"\$RUN/gt\" \
            --eval-code-dir \"\$RUN/official/CMRx4DFlow2026/CMRx4DFlowReconDemo/Utils\" \
            --cache-dir \"\$RUN/corrmaps\" \
            --spatial-mode full \
            --use-venc \
            --include-complex-diff \
            --out-csv \"\$RUN/metrics/epoch960.csv\" \
            --out-json \"\$RUN/metrics/epoch960.summary.json\"
    "

cp "/home/pengfeig/workspace/slurm-logs-agent/${SLURM_JOB_ID}-4dflow-local-score.out" "$RUN_ROOT/logs/" 2>/dev/null || true
