#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PAPER_PYTHON:-python}"
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/cut_4dflow.json "$@"
