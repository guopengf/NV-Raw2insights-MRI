#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=4dflow-org-corrmap
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-org-corrmap.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
RUN_ROOT=${RUN_ROOT:-${REPO}/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z}
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
EVAL_CODE=${RUN_ROOT}/official/CMRx-evaluation-snippets/evaluation-2026
CORRMAP_ROOT=${RUN_ROOT}/inputs/organizer_corrmaps/materialized
EXPECTED=${REPO}/configs/benchmarks/4dflow_joint_channel_epoch960_synapse_expected.json

mkdir -p "$RUN_ROOT/metrics" "$RUN_ROOT/logs"
test -d "$EVAL_CODE"
test -d "$CORRMAP_ROOT"
test -f "$EXPECTED"
test "$(find "$CORRMAP_ROOT" -path '*/ValidationSet/*/corrmap.npz' -type f | wc -l)" -eq 112
test "$(find "$RUN_ROOT/submissions/joint_channel_epoch960" -type f -name '*.npz' | wc -l)" -eq 112
test "$(find "$RUN_ROOT/submissions/epoch960" -type f -name '*.npz' | wc -l)" -eq 112
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
        CORRMAP=\"\$RUN/inputs/organizer_corrmaps/materialized\"
        '$PY' scripts/tools/build_4dflow_validation_gt.py \
            --finalize --plan \"\$RUN/gt_plan.json\" --gt-root \"\$RUN/gt\" \
            --receipt \"\$RUN/gt_receipt.json\"
        '$PY' scripts/tools/evaluate_4dflow_submission.py \
            --submission-root \"\$RUN/submissions/joint_channel_epoch960\" \
            --gt-root \"\$RUN/gt\" \
            --eval-code-dir \"\$RUN/official/CMRx-evaluation-snippets/evaluation-2026\" \
            --corrmap-root \"\$CORRMAP\" \
            --spatial-mode full \
            --include-complex-diff \
            --out-csv \"\$RUN/metrics/joint_channel_epoch960_official_organizer_corrmap.csv\" \
            --out-json \"\$RUN/metrics/joint_channel_epoch960_official_organizer_corrmap.summary.json\"
        '$PY' scripts/tools/evaluate_4dflow_submission.py \
            --submission-root \"\$RUN/submissions/epoch960\" \
            --gt-root \"\$RUN/gt\" \
            --eval-code-dir \"\$RUN/official/CMRx-evaluation-snippets/evaluation-2026\" \
            --corrmap-root \"\$CORRMAP\" \
            --spatial-mode full \
            --include-complex-diff \
            --out-csv \"\$RUN/metrics/epoch960_official_organizer_corrmap.csv\" \
            --out-json \"\$RUN/metrics/epoch960_official_organizer_corrmap.summary.json\"
        '$PY' scripts/tools/verify_4dflow_leaderboard_parity.py \
            \"\$RUN/metrics/joint_channel_epoch960_official_organizer_corrmap.summary.json\" \
            --expected-json /workspace/code/NV-Raw2insights-MRI-fork/configs/benchmarks/4dflow_joint_channel_epoch960_synapse_expected.json \
            --required-metrics SSIM nRMSE RelErr AngErr \
            --out \"\$RUN/metrics/joint_channel_epoch960_organizer_corrmap_parity_receipt.json\"
    "

cp "/home/pengfeig/workspace/slurm-logs-agent/${SLURM_JOB_ID}-4dflow-org-corrmap.out" "$RUN_ROOT/logs/" 2>/dev/null || true
