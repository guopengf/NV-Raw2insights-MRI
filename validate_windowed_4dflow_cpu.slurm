#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --job-name=windowed4d-cpu
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-windowed4d-cpu.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATA_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData
CANARY_ROOT=/workspace/datasets/CMRx4DFlow2026-windowed-v1-canary
RUN_ROOT=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5/outputs/4dflow/windowed_hdf5_validation/${SLURM_JOB_ID}

srun --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd /workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        mkdir -p '$RUN_ROOT'

        \"\$PY\" -m py_compile \
            scripts/windowed_4dflow.py \
            scripts/tools/build_4dflow_windowed_h5.py \
            scripts/tools/benchmark_4dflow_windowed_loader.py \
            scripts/train.py
        \"\$PY\" scripts/tools/run_direct_regression_tests.py | tee '$RUN_ROOT/regressions.log'
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --output-root '$CANARY_ROOT' \
            --limit-patients 1 \
            --deep-verify | tee '$RUN_ROOT/conversion.log'
        \"\$PY\" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --output-root '$CANARY_ROOT' \
            --verify-only --deep-verify | tee '$RUN_ROOT/verification.log'

        for MODE in joint_batch legacy; do
            if [[ \"\$MODE\" == joint_batch ]]; then
                CONFIG=configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json
            else
                CONFIG=configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json
            fi
            \"\$PY\" scripts/tools/benchmark_4dflow_windowed_loader.py \
                --config \"\$CONFIG\" \
                --index-path '$CANARY_ROOT/index.json' \
                --work-dir '$RUN_ROOT/benchmark_'\"\$MODE\" \
                --output '$RUN_ROOT/benchmark_'\"\$MODE\"'.json'
        done
        touch '$RUN_ROOT/COMPLETED'
    "
