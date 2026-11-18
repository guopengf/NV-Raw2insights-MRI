#!/usr/bin/env bash
set -euo pipefail

experiment="${1:?Usage: bash cluster_train.sh <experiment> <prepared-config-directory> [training options]}"
configs="$(realpath "${2:?Pass the configs directory from build_virtual_dataset}")"
shift 2
cd "$(dirname "$0")/../../.."

case "$experiment" in
  ablation_A[0-4]|ours_full_final_dataset) entry=experiments.paper.ours.run ;;
  flowvn|flowmodl|cut_4dflow) entry=experiments.paper.comparisons.run ;;
  *) printf 'Unknown neural experiment: %s\n' "$experiment" >&2; exit 2 ;;
esac

config="$configs/$experiment.json"
test -f "$config" || { printf 'Missing prepared config: %s\n' "$config" >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec "${PAPER_PYTHON:-$PWD/experiments/paper/.venv/bin/python}" -u -m "$entry" --config "$config" "$@"
