#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=4dflow-unified-h5-prepare
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-unified-h5-prepare.out

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
H5_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/datasets
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
REPO=/workspace/code/NV-Raw2insights-MRI-fork
CONFIG=${REPO}/configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_batch_flow_unified_70_15_15.json
E1_HOST_ROOT=${H5_DATASETS}/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
E1_ROOT=/h5data/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
PLAN=${CONTROL_ROOT}/conversion-plan.json

test ! -e "$E1_HOST_ROOT"
test ! -e "$HOST_CONTROL_ROOT/COMPLETED"
mkdir -p "$HOST_CONTROL_ROOT"
/cm/shared/apps/scripts/fs-quota-status > "$HOST_CONTROL_ROOT/quota-before.txt"
grep -q 'project=healthcareeng_isaac' "$HOST_CONTROL_ROOT/quota-before.txt"

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
            --prepare-plan \
            --config '$CONFIG' \
            --data-split train \
            --e1-output-root '$E1_ROOT' \
            --plan-out '$PLAN' \
            --num-shards 8 \
            --expected-patients 292 \
            | tee '$CONTROL_ROOT/prepare.log'

        \"\$PY\" - '$PLAN' '$CONTROL_ROOT/inventory.json' <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
profiles = Counter(tuple(sorted(int(value) for value in patient['inputs'])) for patient in plan['patients'])
expected = Counter({(10, 20, 30, 40, 50): 208, (10,): 19, (20,): 19, (30,): 16, (40,): 14, (50,): 16})
if profiles != expected:
    raise RuntimeError(f'Unexpected acceleration profiles: {profiles}')
manifest_count = sum(len(patient['inputs']) for patient in plan['patients'])
if manifest_count != 1124:
    raise RuntimeError(f'Expected 1124 training manifests, found {manifest_count}')
payload = {
    'status': 'ok',
    'patients': plan['patient_count'],
    'profiles': {','.join(map(str, key)): value for key, value in sorted(profiles.items())},
    'joint_manifests': manifest_count,
    'source_bytes': sum(int(patient['source_bytes']) for patient in plan['patients']),
    'plan_sha256': plan['plan_sha256'],
    'shards': [
        {
            'shard_index': shard['shard_index'],
            'patient_count': shard['patient_count'],
            'estimated_conversion_bytes': shard['estimated_conversion_bytes'],
        }
        for shard in plan['shards']
    ],
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY
    "

test -s "$HOST_CONTROL_ROOT/conversion-plan.json"
test -s "$HOST_CONTROL_ROOT/inventory.json"
test ! -e "$E1_HOST_ROOT"
