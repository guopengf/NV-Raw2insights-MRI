#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --job-name=build-windowed4d-h5
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-build-windowed4d-h5.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATA_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData
E1_OUTPUT_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2

srun --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_OUTPUT_ROOT' \
            --deep-verify
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_OUTPUT_ROOT' \
            --verify-only --deep-verify
        touch '$E1_OUTPUT_ROOT/COMPLETED'
    "
