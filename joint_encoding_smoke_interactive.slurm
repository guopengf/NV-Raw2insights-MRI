#!/bin/bash
#SBATCH -A healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=raw2ins-joint4-smoke
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-joint-encodings/logs/%j-joint4-smoke.out

set -euo pipefail

echo "READY job=${SLURM_JOB_ID} node=$(hostname)"
while true; do
    sleep 60
done
