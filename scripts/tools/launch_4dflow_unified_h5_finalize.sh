#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=4dflow-unified-h5-finalize
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-unified-h5-finalize.out

set -euo pipefail

: "${RUN_ID:?Submit with --export=ALL,RUN_ID=<safe-id>}"
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac

export NVIDIA_VISIBLE_DEVICES=void
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
SOURCE_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
H5_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
REPO=/workspace/code/NV-Raw2insights-MRI-fork
CONFIG=${REPO}/configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_batch_flow_unified_70_15_15.json
E1_HOST_ROOT=${H5_DATASETS}/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
E1_ROOT=/h5data/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
PLAN=${CONTROL_ROOT}/conversion-plan.json

test -s "$HOST_CONTROL_ROOT/conversion-plan.json"
for shard_index in 0 1 2 3 4 5 6 7; do
    printf -v marker '%s/shards/shard-%02d.json' "$HOST_CONTROL_ROOT" "$shard_index"
    test -s "$marker"
done
test ! -e "$HOST_CONTROL_ROOT/COMPLETED"
if test -e "$E1_HOST_ROOT/COMPLETED"; then
    test -s "$E1_HOST_ROOT/index.json"
fi
mkdir -p "$HOST_CONTROL_ROOT/validation"

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

        if test -e '$E1_ROOT/COMPLETED'; then
            test -s '$E1_ROOT/index.json'
            printf '%s\n' '{\"status\":\"reused_completed_finalization\"}' \
                | tee -a '$CONTROL_ROOT/finalize.log'
        else
            \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
                --finalize-plan \
                --plan '$PLAN' \
                --deep-verify \
                --expected-patients 292 \
                | tee '$CONTROL_ROOT/finalize.log'
        fi

        \"\$PY\" scripts/tools/validate_4dflow_windowed_h5_real.py \
            --config '$CONFIG' \
            --index-path '$E1_ROOT/index.json' \
            --work-dir '$CONTROL_ROOT/validation/work' \
            --output '$CONTROL_ROOT/validation/receipt.json' \
            --expected-patients 292 \
            --expected-manifests 1124 \
            --expected-val-patients 63 \
            --expected-val-manifests 259 \
            --raw-parity-patients 0 \
            | tee '$CONTROL_ROOT/validation/validate.log'
    "

test -f "$E1_HOST_ROOT/COMPLETED"
test -s "$E1_HOST_ROOT/index.json"
test -s "$HOST_CONTROL_ROOT/validation/receipt.json"
test -z "$(find "$E1_HOST_ROOT" -type f \( -name '*.tmp' -o -name '*.partial' \) -print -quit)"
du -s --block-size=1 "$E1_HOST_ROOT" > "$HOST_CONTROL_ROOT/host-du-bytes.txt"
sha256sum \
    "$E1_HOST_ROOT/index.json" \
    "$HOST_CONTROL_ROOT/conversion-plan.json" \
    /home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_batch_flow_unified_70_15_15.json \
    > "$HOST_CONTROL_ROOT/sha256sums.txt"
/cm/shared/apps/scripts/fs-quota-status > "$HOST_CONTROL_ROOT/quota-after.txt"
touch "$HOST_CONTROL_ROOT/COMPLETED"
printf 'PRODUCTION_H5_ROOT=%s\n' "$E1_HOST_ROOT"
