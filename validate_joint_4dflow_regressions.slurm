#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=joint4d-regressions
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-joint4d-regressions.out

set -euo pipefail

IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig

srun \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        /root/miniconda3/envs/nv-raw2insights-mri/bin/python -m pytest -q \
            tests/test_joint_4dflow_encodings.py \
            tests/test_flowvn_3d_backbone.py
    "
