#!/bin/bash
#SBATCH --job-name=raw2ins_joint_ep100_recover
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/logs/%A_%a-raw2ins-joint-ep100-recover.out

set -euo pipefail
: "${RUN_NAME:?Submit with --array=<failed-shards> --export=ALL,RUN_NAME=<full-run-name>}"

export HF_HOME=/workspace/code/NV-Raw2insights-MRI-fork/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
export GIT_DIR=/workspace/code/NV-Raw2insights-MRI-fork/.git/worktrees/NV-Raw2insights-MRI-fork-joint-epoch100-inference
export GIT_WORK_TREE=/workspace/code/NV-Raw2insights-MRI-fork-joint-epoch100-inference
cd "$GIT_WORK_TREE"

python scripts/tools/recover_4dflow_challenge_shard.py \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --run-root "/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/challenge_submission_joint_epoch100_3d_flowvn_multiplane/runs/${RUN_NAME}" \
    --batch-size 4 \
    --nproc 1 \
    --num-workers 0
INNER
