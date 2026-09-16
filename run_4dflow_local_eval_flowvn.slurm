#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=4dflow-flowvn-r2
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-flowvn-r2.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
RUN_ROOT=${RUN_ROOT:-${REPO}/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z}
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python

test -f "$RUN_ROOT/metrics/epoch900_parity_receipt.json"
test -f "$RUN_ROOT/metrics/epoch960.csv"
mkdir -p "$RUN_ROOT/flowvn/TaskR1R2/ValidationSet" "$RUN_ROOT/logs"

srun \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${DATASET_ROOT}:/datasets" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        export PATH=/root/miniconda3/envs/nv-raw2insights-mri/bin:\$PATH
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        RUN=/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/local_leaderboard_eval/joint_venc_epoch960_20260916T210353Z
        OFFICIAL=\"\$RUN/official/CMRx4DFlow2026/CMRx4DFlowReconDemo\"
        cd \"\$OFFICIAL/ForEvalution\"
        '$PY' -c 'import torch, pytorch_lightning; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
        '$PY' BatchRecon_FlowVN.py \
            --flowvn_main ../FlowVN/main.py \
            --test_roots /datasets/fudan/data/CMRx4DFlow2026-ChallengeData-20260908/TaskR1R2/ValidationSet \
            --in_base_dir /datasets/fudan/data/CMRx4DFlow2026-ChallengeData-20260908/TaskR1R2/ValidationSet \
            --out_base_dir \"\$RUN/flowvn/TaskR1R2/ValidationSet\" \
            --ckpt_path ../FlowVN/weights/3-epochepoch=015.ckpt \
            --usrate 10 20 30 40 50 \
            --devices 0
        cd /workspace/code/NV-Raw2insights-MRI-fork
        '$PY' scripts/tools/evaluate_4dflow_submission.py \
            --submission-root \"\$RUN/flowvn\" \
            --gt-root \"\$RUN/gt\" \
            --eval-code-dir \"\$RUN/official/CMRx4DFlow2026/CMRx4DFlowReconDemo/Utils\" \
            --cache-dir \"\$RUN/corrmaps\" \
            --task TaskR1R2 \
            --spatial-mode full \
            --include-complex-diff \
            --out-csv \"\$RUN/metrics/flowvn_r2.csv\" \
            --out-json \"\$RUN/metrics/flowvn_r2.summary.json\"
        '$PY' scripts/tools/compare_4dflow_flowvn_baseline.py \
            --submission-csv \"\$RUN/metrics/epoch960.csv\" \
            --flowvn-csv \"\$RUN/metrics/flowvn_r2.csv\" \
            --out-csv \"\$RUN/metrics/epoch960_r2_vs_flowvn.csv\" \
            --out-json \"\$RUN/metrics/epoch960_r2_vs_flowvn.summary.json\"
    "

cp "/home/pengfeig/workspace/slurm-logs-agent/${SLURM_JOB_ID}-4dflow-flowvn-r2.out" "$RUN_ROOT/logs/" 2>/dev/null || true
