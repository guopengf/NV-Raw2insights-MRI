#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=cpu1-00012
#SBATCH --time=04:00:00
#SBATCH --job-name=build-windowed4d-e1-10n
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-build-windowed4d-e1-10n.out

set -euo pipefail

IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_PARENT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
DATA_ROOT=${DATASET_PARENT}/CMRx4DFlow2026-ChallengeData
E1_HOST_ROOT=${DATA_ROOT}/windowed-e1-v2
E4_HOST_ROOT=${DATA_ROOT}/windowed-e4-v2
E1_OUTPUT_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2
E4_OUTPUT_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e4-v2
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_e1_conversion/${SLURM_JOB_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_e1_conversion/${SLURM_JOB_ID}
PLAN=${CONTROL_ROOT}/conversion-plan.json

test ! -e "$E4_HOST_ROOT"
mkdir -p "$HOST_CONTROL_ROOT"

container_args=(
    --container-image "$IMAGE"
    --container-mounts="${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData"
    --no-container-mount-home
    --container-remap-root
)

srun --nodes=10 --ntasks=10 --ntasks-per-node=1 --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    --output="$HOST_CONTROL_ROOT/preflight-%t.out" \
    --error="$HOST_CONTROL_ROOT/preflight-%t.err" \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        test -d /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        test -d '$E1_OUTPUT_ROOT'
        test ! -e '$E4_OUTPUT_ROOT'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" -c 'import h5py, scipy, torch; from monai.data.fft_utils import ifftn_centered; print("PYXIS_PREFLIGHT_OK")'
        printf 'NODE_PREFLIGHT_OK host=%s rank=%s\n' \"\$(hostname)\" \"\${SLURM_PROCID}\"
    "

srun --nodes=1 --ntasks=1 --cpus-per-task=1 \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --prepare-plan \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_OUTPUT_ROOT' \
            --plan-out '$PLAN' \
            --num-shards 10 \
            --expected-patients 138 \
            --forbid-path '$E4_OUTPUT_ROOT'
    "

srun --nodes=10 --ntasks=10 --ntasks-per-node=1 --cpus-per-task=32 \
    --kill-on-bad-exit=1 \
    --output="$HOST_CONTROL_ROOT/shard-%t.out" \
    --error="$HOST_CONTROL_ROOT/shard-%t.err" \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --run-shard \
            --plan '$PLAN' \
            --shard-index \"\${SLURM_PROCID}\"
    "

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
    "${container_args[@]}" \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --finalize-plan \
            --plan '$PLAN' \
            --deep-verify \
            --expected-patients 138 \
            --forbid-path '$E4_OUTPUT_ROOT'
    "

test -f "$E1_HOST_ROOT/COMPLETED"
test ! -e "$E4_HOST_ROOT"
