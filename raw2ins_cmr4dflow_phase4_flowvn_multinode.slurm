#!/bin/bash
#SBATCH --nodes=2
#SBATCH -A healthcareeng_monai
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --job-name=raw2ins-p4-flowvn
#SBATCH --dependency=singleton
#SBATCH --output=/home/pengfeig/slurm-logs-agent/%x-%j.out

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

USER=pengfeig
srun --kill-on-bad-exit=1 \
    --container-image ~/workspace/cache/conda_raw2insights.sqsh \
    --container-mounts=${HOME}/.netrc:${CONTAINER_HOME:-/root}/.netrc,/lustre/fsw/portfolios/healthcareeng/users/pengfeig:/workspace,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData:/data/CMRx4DFlow2026-ChallengeData \
    --no-container-mount-home \
    --container-remap-root \
    --export=ALL \
    bash -c '
        echo "MASTER_ADDR="$MASTER_ADDR
        CURRENT_RANK=${SLURM_PROCID:-"0"}
        n_node=${SLURM_JOB_NUM_NODES:-1}

        echo "JobID: $SLURM_JOB_ID | Full list: $worker_list | rank $CURRENT_RANK of $n_node"
        echo "number of nodes:" $n_node
        echo "node rank:" $SLURM_PROCID

        source ~/miniconda3/bin/activate
        source ~/.bashrc
        conda init
        conda activate nv-raw2insights-mri
        which python

        CONFIG="./configs/nv_raw2insights_mri_base_4dflow_3d_flowvn_multiplane.json"

        cd /workspace/code/NV-Raw2insights-MRI-fork
        export NCCL_P2P_LEVEL=NVL
        export HF_HOME=/workspace/code/NV-Raw2insights-MRI-fork/cache
        export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
        export TORCH_NCCL_DUMP_ON_TIMEOUT=1
        export TORCH_NCCL_DESYNC_DEBUG=1
        export PYTHONUNBUFFERED=1
        torchrun --nproc_per_node=8 --nnodes=$n_node  --master_port=25001 \
            --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
            scripts/train.py \
            --config $CONFIG
    '
