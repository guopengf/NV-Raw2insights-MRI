#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PAPER_PYTHON:-python}"
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ours_full_final_dataset.json "$@"
