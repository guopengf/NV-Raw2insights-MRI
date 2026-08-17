#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=validate-windowed4d-e1
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-validate-windowed4d-e1.out

set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATA_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData
E1_CANARY_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2-canary
RUN_ROOT=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5/outputs/4dflow/windowed_hdf5_e1_validation/${SLURM_JOB_ID}

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
        mkdir -p '$RUN_ROOT'
        "\$PY" -m py_compile \
            scripts/tools/build_4dflow_windowed_h5.py \
            scripts/windowed_4dflow.py \
            tests/test_windowed_4dflow.py
        "\$PY" scripts/tools/run_direct_regression_tests.py | tee '$RUN_ROOT/regressions.log'
        "\$PY" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_CANARY_ROOT' \
            --limit-patients 1 \
            --deep-verify | tee '$RUN_ROOT/e1_restart.log'
        "\$PY" scripts/tools/build_4dflow_windowed_h5.py \
            --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
            --e1-output-root '$E1_CANARY_ROOT' \
            --verify-only --deep-verify | tee '$RUN_ROOT/e1_verify.log'
        "\$PY" - '$E1_CANARY_ROOT/index.json' <<'PY'
import json
import sys

index = json.load(open(sys.argv[1]))
assert index['schema_version'] == 2
assert index['storage_profile'] == 'encoding_chunk_1'
assert len(index['patients']) == 1
print('E1_ONLY_CANARY_OK patients=1 profile=encoding_chunk_1')
PY
        test -f '$E1_CANARY_ROOT/COMPLETED'
        touch '$RUN_ROOT/COMPLETED'
    "
