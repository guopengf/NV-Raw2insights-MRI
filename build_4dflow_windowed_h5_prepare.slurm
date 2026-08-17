#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=cpu1-00012
#SBATCH --time=00:30:00
#SBATCH --job-name=build-windowed4d-e1-prepare
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-build-windowed4d-e1-prepare.out

set -euo pipefail

: "${RUN_ID:?RUN_ID must be exported by the submission wrapper}"
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
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
mkdir -p "$HOST_CONTROL_ROOT"

container_args=(
    --container-image "$IMAGE"
    --container-mounts="${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData"
    --no-container-mount-home
    --container-remap-root
)

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --output="$HOST_CONTROL_ROOT/prepare.out" \
    --error="$HOST_CONTROL_ROOT/prepare.err" \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        test -d /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        test -d '$E1_OUTPUT_ROOT'
        test ! -e '$E4_OUTPUT_ROOT'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" -c 'import h5py, scipy, torch; from monai.data.fft_utils import ifftn_centered; print(\"PREPARE_PREFLIGHT_OK\")'
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --prepare-plan \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_OUTPUT_ROOT' \
            --plan-out '$PLAN' \
            --num-shards 2 \
            --expected-patients 138 \
            --forbid-path '$E4_OUTPUT_ROOT'
    "

test -s "$HOST_CONTROL_ROOT/conversion-plan.json"
test ! -e "$E4_HOST_ROOT"
