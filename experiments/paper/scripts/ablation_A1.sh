#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PAPER_PYTHON:-python}"
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A1.json "$@"
