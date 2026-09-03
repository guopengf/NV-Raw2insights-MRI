#!/bin/bash
#SBATCH --job-name=raw2ins_joint_ep100_package
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=cpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/logs/%j-raw2ins-joint-ep100-package.out

set -euo pipefail
: "${RUN_NAME:?Submit with --export=ALL,RUN_NAME=<completed-full-run-name>}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

srun --container-image ~/workspace/cache/conda_raw2insights.sqsh \
    --container-mounts=${HOME}/.netrc:${CONTAINER_HOME:-/root}/.netrc,/lustre/fsw/portfolios/healthcareeng/users/pengfeig:/workspace,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData:/data/CMRx4DFlow2026-ChallengeData \
    --no-container-mount-home \
    --container-remap-root \
    --export=ALL \
    bash -s <<'INNER'
set -euo pipefail
source ~/miniconda3/bin/activate
set +u
source ~/.bashrc
set -u
conda activate nv-raw2insights-mri
cd /workspace/code/NV-Raw2insights-MRI-fork

python scripts/tools/run_4dflow_challenge_submission.py package-shards \
    --run-root "/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/challenge_submission_joint_epoch100_3d_flowvn_multiplane/runs/${RUN_NAME}" \
    --artifact-tag small_joint_epoch100 \
    --skip-combined-zip
INNER
