#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=4dflow-h5-migrate-monai
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-4dflow-h5-migrate-monai.out

set -euo pipefail
umask 002

: "${MIGRATION_ID:?Submit with --export=ALL,MIGRATION_ID=<safe-id>}"
case "$MIGRATION_ID" in
    *[!A-Za-z0-9._-]* | "")
        printf 'Invalid MIGRATION_ID: %q\n' "$MIGRATION_ID" >&2
        exit 2
        ;;
esac

SOURCE_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/datasets/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2
MONAI_DATASETS=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets
DEST_PARENT=${MONAI_DATASETS}/CMRx4DFlow2026-unified-70_15_15-seed20260914
DEST_ROOT=${DEST_PARENT}/windowed-e1-v2
STAGING_ROOT=${DEST_PARENT}/.windowed-e1-v2.copying-${MIGRATION_ID}
HOST_CONTROL_ROOT=/home/pengfeig/workspace/outputs/4dflow/windowed_h5_unified_migration/${MIGRATION_ID}
CONTROL_ROOT=/workspace/outputs/4dflow/windowed_h5_unified_migration/${MIGRATION_ID}
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
REPO=/workspace/code/NV-Raw2insights-MRI-fork
CONFIG=${REPO}/configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_batch_flow_unified_70_15_15.json
DEST_INDEX=/h5data/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2/index.json
EXPECTED_INDEX_SHA256=7300ab8d8014c6486826888c70671836d2e73a2a9c88e3cb92dd3a86aed6f414

mkdir -p "$HOST_CONTROL_ROOT" "$DEST_PARENT"
test -s "$SOURCE_ROOT/index.json"
test -f "$SOURCE_ROOT/COMPLETED"
test "$(find "$SOURCE_ROOT/patients" -type f -name '*.h5' | wc -l)" -eq 292
test -z "$(find "$SOURCE_ROOT" -type f \( -name '*.tmp' -o -name '*.partial' \) -print -quit)"
printf '%s  %s\n' "$EXPECTED_INDEX_SHA256" "$SOURCE_ROOT/index.json" \
    | sha256sum --check --strict

printf 'source=%s\ndestination=%s\nstaging=%s\n' \
    "$SOURCE_ROOT" "$DEST_ROOT" "$STAGING_ROOT" \
    > "$HOST_CONTROL_ROOT/paths.txt"
/cm/shared/apps/scripts/fs-quota-status > "$HOST_CONTROL_ROOT/quota-before.txt"
findmnt -T "$SOURCE_ROOT" > "$HOST_CONTROL_ROOT/source-mount.txt"
findmnt -T "$DEST_PARENT" > "$HOST_CONTROL_ROOT/destination-mount.txt"

if test -e "$DEST_ROOT" && test -e "$STAGING_ROOT"; then
    printf 'Both final and staging destinations exist; refusing ambiguous recovery.\n' >&2
    exit 3
fi

if test -e "$DEST_ROOT"; then
    VERIFY_ROOT=$DEST_ROOT
    printf 'Reusing already promoted destination for verification.\n'
else
    mkdir -p "$STAGING_ROOT"
    rsync -a --partial --append-verify --human-readable --info=progress2,stats2 \
        "$SOURCE_ROOT/" "$STAGING_ROOT/" \
        | tee "$HOST_CONTROL_ROOT/rsync-copy.log"
    VERIFY_ROOT=$STAGING_ROOT
fi

(
    cd "$SOURCE_ROOT"
    find . -type f -printf '%P\t%s\n' | LC_ALL=C sort
) > "$HOST_CONTROL_ROOT/source-inventory.tsv"
(
    cd "$VERIFY_ROOT"
    find . -type f -printf '%P\t%s\n' | LC_ALL=C sort
) > "$HOST_CONTROL_ROOT/destination-inventory.tsv"
cmp "$HOST_CONTROL_ROOT/source-inventory.tsv" \
    "$HOST_CONTROL_ROOT/destination-inventory.tsv"

rsync -a --checksum --dry-run --itemize-changes \
    "$SOURCE_ROOT/" "$VERIFY_ROOT/" \
    > "$HOST_CONTROL_ROOT/rsync-checksum-diff.txt"
test ! -s "$HOST_CONTROL_ROOT/rsync-checksum-diff.txt"

test "$(find "$VERIFY_ROOT/patients" -type f -name '*.h5' | wc -l)" -eq 292
test -z "$(find "$VERIFY_ROOT" -type f \( -name '*.tmp' -o -name '*.partial' \) -print -quit)"
printf '%s  %s\n' "$EXPECTED_INDEX_SHA256" "$VERIFY_ROOT/index.json" \
    | sha256sum --check --strict
test -f "$VERIFY_ROOT/COMPLETED"

if test "$VERIFY_ROOT" = "$STAGING_ROOT"; then
    test ! -e "$DEST_ROOT"
    mv "$STAGING_ROOT" "$DEST_ROOT"
fi
test -d "$DEST_ROOT"

mkdir -p "$HOST_CONTROL_ROOT/validation"
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${MONAI_DATASETS}:/data,${MONAI_DATASETS}:/h5data" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd '$REPO'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        \"\$PY\" scripts/tools/validate_4dflow_windowed_h5_real.py \\
            --config '$CONFIG' \\
            --index-path '$DEST_INDEX' \\
            --work-dir '$CONTROL_ROOT/validation/work' \\
            --output '$CONTROL_ROOT/validation/receipt.json' \\
            --expected-patients 292 \\
            --expected-manifests 1124 \\
            --expected-val-patients 63 \\
            --expected-val-manifests 259 \\
            --raw-parity-patients 0 \\
            | tee '$CONTROL_ROOT/validation/validate.log'
    "

test "$(jq -r '.status' "$HOST_CONTROL_ROOT/validation/receipt.json")" = ok
du -s --block-size=1 "$SOURCE_ROOT" > "$HOST_CONTROL_ROOT/source-du-bytes.txt"
du -s --block-size=1 "$DEST_ROOT" > "$HOST_CONTROL_ROOT/destination-du-bytes.txt"
sha256sum "$SOURCE_ROOT/index.json" "$DEST_ROOT/index.json" \
    > "$HOST_CONTROL_ROOT/index-sha256sums.txt"
sha256sum "$HOST_CONTROL_ROOT/source-inventory.tsv" \
    "$HOST_CONTROL_ROOT/destination-inventory.tsv" \
    > "$HOST_CONTROL_ROOT/inventory-sha256sums.txt"
/cm/shared/apps/scripts/fs-quota-status > "$HOST_CONTROL_ROOT/quota-after.txt"

SOURCE_BYTES=$(cut -f1 "$HOST_CONTROL_ROOT/source-du-bytes.txt")
DEST_BYTES=$(cut -f1 "$HOST_CONTROL_ROOT/destination-du-bytes.txt")
INVENTORY_SHA256=$(sha256sum "$HOST_CONTROL_ROOT/source-inventory.tsv" | awk '{print $1}')
RECEIPT_SHA256=$(sha256sum "$HOST_CONTROL_ROOT/validation/receipt.json" | awk '{print $1}')
jq -n \
    --arg migration_id "$MIGRATION_ID" \
    --arg source "$SOURCE_ROOT" \
    --arg destination "$DEST_ROOT" \
    --arg inventory_sha256 "$INVENTORY_SHA256" \
    --arg validation_receipt_sha256 "$RECEIPT_SHA256" \
    --arg slurm_job_id "${SLURM_JOB_ID:-unknown}" \
    --argjson source_bytes "$SOURCE_BYTES" \
    --argjson destination_bytes "$DEST_BYTES" \
    '{
        status: "ok",
        migration_id: $migration_id,
        slurm_job_id: $slurm_job_id,
        source: $source,
        destination: $destination,
        source_retained: true,
        patients: 292,
        source_bytes: $source_bytes,
        destination_bytes: $destination_bytes,
        inventory_sha256: $inventory_sha256,
        validation_receipt_sha256: $validation_receipt_sha256
    }' > "$HOST_CONTROL_ROOT/migration-receipt.json"
touch "$HOST_CONTROL_ROOT/COMPLETED"
printf 'MIGRATION_DESTINATION=%s\n' "$DEST_ROOT"
