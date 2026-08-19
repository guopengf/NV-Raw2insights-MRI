#!/bin/bash
#SBATCH --job-name=raw2ins_joint_ep100_smoke
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=batch_short
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/logs/%A_%a-raw2ins-joint-ep100-smoke.out

set -euo pipefail
: "${RUN_NAME:?Submit with --export=ALL,RUN_NAME=<unique-smoke-run-name>}"

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

python scripts/tools/run_4dflow_challenge_submission.py run-shard \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --mode smoke \
    --data-base /data/CMRx4DFlow2026-ChallengeData \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
    --checkpoint /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/4dflow_joint_encoding/small_ft_4dflow_joint_batch_3d_flowvn_multiplane_windowed_h5/nv_raw2insights_mri_small_ft_4dflow_joint_batch_3d_flowvn_multiplane_windowed_h5_epoch100.pt \
    --output-root "/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/challenge_submission_joint_epoch100_3d_flowvn_multiplane/runs/${RUN_NAME}" \
    --expected-checkpoint-sha256 65af7ea1d0004afa4f15fac0ceb4c1ab6a53984c40fa1fe4581d059db8a6365d \
    --expected-config-sha256 dbb324f41b3c9fafa5728a9f40d727a68cb6c596f8264719fea70ecadeb9cbb2 \
    --expected-inference-sha256 aa6fbbf3a10dc8ec01123c7dd5414812126f67486dea787e93c4b2a3a345b210 \
    --expected-exporter-sha256 45b140e492f45a24dbf972b7f44d3bb15b89be883bcd133f5b481d5da8dec06b \
    --expected-epoch 100 \
    --expected-global-step 17606 \
    --expected-wandb-run-id kqi05x9h \
    --nproc 4 \
    --num-workers 4
INNER
