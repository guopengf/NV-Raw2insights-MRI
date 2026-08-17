#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=cpu1-00012
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --job-name=build-windowed4d-e1-shard
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%A_%a-build-windowed4d-e1-shard.out

set -euo pipefail

: "${RUN_ID:?RUN_ID must be exported by the submission wrapper}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a SLURM array}"
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac
case "$SLURM_ARRAY_TASK_ID" in
    0 | 1) ;;
    *)
        printf 'Invalid shard index: %q\n' "$SLURM_ARRAY_TASK_ID" >&2
        exit 2
        ;;
esac

IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_PARENT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
DATA_ROOT=${DATASET_PARENT}/CMRx4DFlow2026-ChallengeData
E1_HOST_ROOT=${DATA_ROOT}/windowed-e1-v2
E4_HOST_ROOT=${DATA_ROOT}/windowed-e4-v2
E1_OUTPUT_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2
E4_OUTPUT_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e4-v2
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_e1_conversion/${RUN_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_e1_conversion/${RUN_ID}
PLAN=${CONTROL_ROOT}/conversion-plan.json

test -d "$E1_HOST_ROOT"
test ! -e "$E4_HOST_ROOT"
test ! -e "$E1_HOST_ROOT/COMPLETED"
test -s "$HOST_CONTROL_ROOT/conversion-plan.json"

container_args=(
    --container-image "$IMAGE"
    --container-mounts="${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData"
    --no-container-mount-home
    --container-remap-root
)

srun --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    --output="$HOST_CONTROL_ROOT/preflight-${SLURM_ARRAY_TASK_ID}.out" \
    --error="$HOST_CONTROL_ROOT/preflight-${SLURM_ARRAY_TASK_ID}.err" \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        test -d /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        test -d '$E1_OUTPUT_ROOT'
        test ! -e '$E4_OUTPUT_ROOT'
        test -s '$PLAN'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" -c 'import h5py, scipy, torch; from monai.data.fft_utils import ifftn_centered; print(\"PYXIS_PREFLIGHT_OK\")'
        printf 'NODE_PREFLIGHT_OK host=%s shard=%s\n' \"\$(hostname)\" '$SLURM_ARRAY_TASK_ID'
    "

srun --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --kill-on-bad-exit=1 \
    --output="$HOST_CONTROL_ROOT/shard-${SLURM_ARRAY_TASK_ID}.out" \
    --error="$HOST_CONTROL_ROOT/shard-${SLURM_ARRAY_TASK_ID}.err" \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --run-shard \
            --plan '$PLAN' \
            --shard-index '$SLURM_ARRAY_TASK_ID'
    "

printf -v SHARD_MARKER '%s/shards/shard-%02d.json' "$HOST_CONTROL_ROOT" "$SLURM_ARRAY_TASK_ID"
test -s "$SHARD_MARKER"
test ! -e "$E4_HOST_ROOT"
