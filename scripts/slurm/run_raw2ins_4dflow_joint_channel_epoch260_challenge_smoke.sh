#!/bin/bash
#SBATCH --job-name=raw2ins_joint_ch260_smoke
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition=batch,batch_short,interactive
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork/logs/%A_%a-raw2ins-joint-ch260-smoke.out

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
cd /workspace/code/NV-Raw2insights-MRI-fork

python scripts/tools/run_4dflow_challenge_submission.py run-shard \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --mode smoke \
    --data-base /data/CMRx4DFlow2026-ChallengeData \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json \
    --checkpoint /workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/4dflow_joint_encoding/small_ft_4dflow_joint_channel_3d_flowvn_multiplane/nv_raw2insights_mri_small_ft_4dflow_joint_channel_3d_flowvn_multiplane_epoch260.pt \
    --output-root "/workspace/code/NV-Raw2insights-MRI-fork/outputs/4dflow/challenge_submission_joint_channel_epoch260_3d_flowvn_multiplane/runs/${RUN_NAME}" \
    --expected-checkpoint-sha256 154ffdd3ea68512d699ce7d65448122fdc6fa7ecff8f050bcd7a6301359f3817 \
    --expected-config-sha256 5d784d2d07f10da41c7b0465f034449c061a61b26446789cdf966d1ebab9280e \
    --expected-inference-sha256 aa6fbbf3a10dc8ec01123c7dd5414812126f67486dea787e93c4b2a3a345b210 \
    --expected-exporter-sha256 45b140e492f45a24dbf972b7f44d3bb15b89be883bcd133f5b481d5da8dec06b \
    --expected-epoch 260 \
    --expected-global-step 45941 \
    --expected-wandb-run-id v9yybr91 \
    --nproc 1 \
    --batch-size 4 \
    --num-workers 0
INNER
