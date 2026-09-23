#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"
sbatch --parsable scripts/slurm/build_4dflow_windowed_h5.sh
