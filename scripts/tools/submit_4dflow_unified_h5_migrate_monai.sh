#!/bin/bash

set -euo pipefail

MIGRATION_ID=${1:-monai_copy_$(date -u +%Y%m%dT%H%M%SZ)}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

sbatch --parsable \
    --export=ALL,MIGRATION_ID="$MIGRATION_ID" \
    "$SCRIPT_DIR/launch_4dflow_unified_h5_migrate_monai.slurm"
