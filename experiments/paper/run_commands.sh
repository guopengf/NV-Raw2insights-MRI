#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PAPER_PYTHON:-python}"
# Formal commands below are NOT executed by this generator.
# ablation_A0
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A0.json
# ablation_A1
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A1.json
# ablation_A2
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A2.json
# ablation_A3
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A3.json
# ablation_A4
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ablation_A4.json
# ours_full_final_dataset
"$PYTHON" -m experiments.paper.ours.run --config configs/paper/ours_full_final_dataset.json
# cg_sense
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/cg_sense.json --smoke --device cpu
# cs_tv
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/cs_tv.json --smoke --device cpu
# flowvn
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/flowvn.json
# flowmodl
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/flowmodl.json
# cut_4dflow
"$PYTHON" -m experiments.paper.comparisons.run --config configs/paper/baselines/cut_4dflow.json
