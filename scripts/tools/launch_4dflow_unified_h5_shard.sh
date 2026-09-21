#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-7%2
#SBATCH --job-name=4dflow-unified-h5-shard
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%A_%a-4dflow-unified-h5-shard.out

set -euo pipefail

: "${RUN_ID:?Submit with --export=ALL,RUN_ID=<safe-id>}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac
case "$SLURM_ARRAY_TASK_ID" in
    0 | 1 | 2 | 3 | 4 | 5 | 6 | 7) ;;
    *)
        printf 'Invalid shard index: %q\n' "$SLURM_ARRAY_TASK_ID" >&2
        exit 2
        ;;
esac

export NVIDIA_VISIBLE_DEVICES=void
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
SOURCE_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
H5_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/datasets
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
REPO=/workspace/code/NV-Raw2insights-MRI-fork
E1_HOST_ROOT=${H5_DATASETS}/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
PLAN=${CONTROL_ROOT}/conversion-plan.json

test -s "$HOST_CONTROL_ROOT/conversion-plan.json"
test ! -e "$HOST_CONTROL_ROOT/COMPLETED"
test ! -e "$E1_HOST_ROOT/COMPLETED"

srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${SOURCE_DATASETS}:/data,${H5_DATASETS}:/h5data" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd '$REPO'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --run-shard \
            --plan '$PLAN' \
            --shard-index '$SLURM_ARRAY_TASK_ID' \
            | tee '$CONTROL_ROOT/shard-${SLURM_ARRAY_TASK_ID}.log'
    "

printf -v SHARD_MARKER '%s/shards/shard-%02d.json' "$HOST_CONTROL_ROOT" "$SLURM_ARRAY_TASK_ID"
test -s "$SHARD_MARKER"
test ! -e "$E1_HOST_ROOT/COMPLETED"
