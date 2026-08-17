#!/bin/bash

set -euo pipefail

REPO_ROOT=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
DATA_ROOT=/home/pengfeig/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData
E1_ROOT=${DATA_ROOT}/windowed-e1-v2
E4_ROOT=${DATA_ROOT}/windowed-e4-v2
CONTROL_BASE=/home/pengfeig/workspace/outputs/4dflow/windowed_e1_conversion
RUN_ID=${RUN_ID:-e1-array-$(date -u +%Y%m%dT%H%M%SZ)-$$}

case "$RUN_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid RUN_ID: %q\n' "$RUN_ID" >&2
        exit 2
        ;;
esac

cd "$REPO_ROOT"
test -d "$E1_ROOT"
test ! -e "$E4_ROOT"
test ! -e "$E1_ROOT/COMPLETED"

active_jobs=$(squeue -h --me --format='%j' | grep -E '^build-windowed4d-e1-(prepare|shard|finalize)$' || true)
if [[ -n "$active_jobs" ]]; then
    printf 'Refusing duplicate conversion chain; active jobs:\n%s\n' "$active_jobs" >&2
    exit 3
fi

CONTROL_ROOT=${CONTROL_BASE}/${RUN_ID}
test ! -e "$CONTROL_ROOT"
mkdir -p "$CONTROL_ROOT"

submitted_ids=()
submission_complete=0
cleanup_partial_submission() {
    if [[ "$submission_complete" -eq 0 && "${#submitted_ids[@]}" -gt 0 ]]; then
        scancel "${submitted_ids[@]}" || true
    fi
}
trap cleanup_partial_submission EXIT

PREP_JOB_ID=$(sbatch --parsable \
    --export=ALL,RUN_ID="$RUN_ID" \
    build_4dflow_windowed_h5_prepare.slurm)
PREP_JOB_ID=${PREP_JOB_ID%%;*}
submitted_ids+=("$PREP_JOB_ID")

ARRAY_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${PREP_JOB_ID}" \
    --export=ALL,RUN_ID="$RUN_ID",PREP_JOB_ID="$PREP_JOB_ID" \
    build_4dflow_windowed_h5_shard.slurm)
ARRAY_JOB_ID=${ARRAY_JOB_ID%%;*}
submitted_ids+=("$ARRAY_JOB_ID")

FINAL_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --export=ALL,RUN_ID="$RUN_ID",PREP_JOB_ID="$PREP_JOB_ID",ARRAY_JOB_ID="$ARRAY_JOB_ID" \
    build_4dflow_windowed_h5_finalize.slurm)
FINAL_JOB_ID=${FINAL_JOB_ID%%;*}
submitted_ids+=("$FINAL_JOB_ID")

printf 'RUN_ID=%s\nPREP_JOB_ID=%s\nARRAY_JOB_ID=%s\nFINAL_JOB_ID=%s\n' \
    "$RUN_ID" "$PREP_JOB_ID" "$ARRAY_JOB_ID" "$FINAL_JOB_ID" \
    > "$CONTROL_ROOT/submission.env"

submission_complete=1
trap - EXIT

printf '{"run_id":"%s","prepare_job_id":"%s","array_job_id":"%s","final_job_id":"%s"}\n' \
    "$RUN_ID" "$PREP_JOB_ID" "$ARRAY_JOB_ID" "$FINAL_JOB_ID"
