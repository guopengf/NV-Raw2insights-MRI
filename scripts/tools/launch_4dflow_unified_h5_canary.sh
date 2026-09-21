#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=4dflow-unified-h5-canary
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-unified-h5-canary.out

set -euo pipefail

: "${RUN_ID:?Submit with --export=ALL,RUN_ID=<safe-id>}"
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac

IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATASET_PARENT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
HOST_RUN_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_canary/${RUN_ID}
RUN_ROOT=/workspace/outputs/4dflow/windowed_h5_unified_canary/${RUN_ID}
REPO=/workspace/code/NV-Raw2insights-MRI-fork
CONFIG=${REPO}/configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_batch_flow_unified_70_15_15.json
E1_ROOT=${RUN_ROOT}/e1
PLAN=${RUN_ROOT}/control/conversion-plan.json

test ! -e "$HOST_RUN_ROOT/COMPLETED"
mkdir -p "$HOST_RUN_ROOT/control" "$HOST_RUN_ROOT/validation"

srun --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${DATASET_PARENT}:/data" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd '$REPO'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

        \"\$PY\" -m py_compile \
            scripts/windowed_4dflow.py \
            scripts/tools/build_4dflow_windowed_h5.py \
            scripts/tools/prepare_4dflow_windowed_h5_canary.py \
            scripts/tools/validate_4dflow_windowed_h5_real.py

        if [[ ! -s '$PLAN' ]]; then
            \"\$PY\" scripts/tools/prepare_4dflow_windowed_h5_canary.py \
                --config '$CONFIG' \
                --e1-output-root '$E1_ROOT' \
                --plan-out '$PLAN' \
                --selection-out '$RUN_ROOT/control/selection.json' \
                | tee '$RUN_ROOT/control/prepare.log'
        fi

        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --run-shard \
            --plan '$PLAN' \
            --shard-index 0 \
            | tee '$RUN_ROOT/control/shard-00.log'

        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --finalize-plan \
            --plan '$PLAN' \
            --deep-verify \
            --expected-patients 6 \
            | tee '$RUN_ROOT/control/finalize.log'

        \"\$PY\" scripts/tools/validate_4dflow_windowed_h5_real.py \
            --config '$CONFIG' \
            --index-path '$E1_ROOT/index.json' \
            --work-dir '$RUN_ROOT/validation/work' \
            --output '$RUN_ROOT/validation/receipt.json' \
            --expected-patients 6 \
            --expected-manifests 10 \
            --expected-val-patients 63 \
            --expected-val-manifests 259 \
            | tee '$RUN_ROOT/validation/validate.log'

        touch '$RUN_ROOT/COMPLETED'
    "

test -s "$HOST_RUN_ROOT/control/conversion-plan.json"
test -s "$HOST_RUN_ROOT/validation/receipt.json"
test -f "$HOST_RUN_ROOT/e1/COMPLETED"
test -f "$HOST_RUN_ROOT/COMPLETED"
du -s --block-size=1 "$HOST_RUN_ROOT" > "$HOST_RUN_ROOT/host-du-bytes.txt"
printf 'CANARY_RUN_ROOT=%s\n' "$HOST_RUN_ROOT"
