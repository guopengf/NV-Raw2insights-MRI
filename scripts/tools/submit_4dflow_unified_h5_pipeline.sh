#!/bin/bash
set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork
H5_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
RUN_ID=${1:-unified_e1_$(date -u +%Y%m%dT%H%M%SZ)}
case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac

CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_production/${RUN_ID}
test ! -e "$CONTROL_ROOT"
test ! -e "$H5_ROOT"
mkdir -p "$CONTROL_ROOT"

cd "$REPO"
prepare_job=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" scripts/tools/launch_4dflow_unified_h5_prepare.sh)
shard_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL,RUN_ID="$RUN_ID" scripts/tools/launch_4dflow_unified_h5_shard.sh)
finalize_job=$(sbatch --parsable --dependency="afterok:${shard_job}" --export=ALL,RUN_ID="$RUN_ID" scripts/tools/launch_4dflow_unified_h5_finalize.sh)

python3 - "$CONTROL_ROOT/submission.json" "$RUN_ID" "$prepare_job" "$shard_job" "$finalize_job" <<'PY'
import json
import sys
from pathlib import Path

path, run_id, prepare_job, shard_job, finalize_job = sys.argv[1:]
payload = {
    "run_id": run_id,
    "prepare_job": prepare_job,
    "shard_array_job": shard_job,
    "finalize_job": finalize_job,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
