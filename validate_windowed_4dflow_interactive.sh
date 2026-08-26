#!/bin/bash
set -euo pipefail

REPO=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
E1_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2-canary
E4_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e4-v2-canary
RUN_ID=${1:?usage: validate_windowed_4dflow_interactive.sh RUN_ID}
RUN_ROOT=$REPO/outputs/4dflow/windowed_hdf5_validation/$RUN_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO"
mkdir -p "$RUN_ROOT"

"$PY" scripts/tools/run_direct_regression_tests.py | tee "$RUN_ROOT/regressions.log"
"$PY" scripts/tools/build_4dflow_windowed_h5.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
    --e1-output-root "$E1_ROOT" \
    --e4-output-root "$E4_ROOT" \
    --limit-patients 1 \
    --deep-verify | tee "$RUN_ROOT/conversion.log"
"$PY" scripts/tools/build_4dflow_windowed_h5.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
    --e1-output-root "$E1_ROOT" \
    --e4-output-root "$E4_ROOT" \
    --verify-only \
    --deep-verify | tee "$RUN_ROOT/verification.log"

"$PY" scripts/tools/benchmark_4dflow_windowed_loader.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
    --index-path "$E1_ROOT/index.json" \
    --work-dir "$RUN_ROOT/benchmark_joint_batch_e1" \
    --output "$RUN_ROOT/benchmark_joint_batch_e1.json"
"$PY" scripts/tools/benchmark_4dflow_windowed_loader.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json \
    --index-path "$E4_ROOT/index.json" \
    --work-dir "$RUN_ROOT/benchmark_joint_batch_e4" \
    --output "$RUN_ROOT/benchmark_joint_batch_e4.json"
"$PY" scripts/tools/benchmark_4dflow_windowed_loader.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json \
    --index-path "$E1_ROOT/index.json" \
    --work-dir "$RUN_ROOT/benchmark_legacy_e1" \
    --output "$RUN_ROOT/benchmark_legacy_e1.json"
"$PY" scripts/tools/benchmark_4dflow_windowed_loader.py \
    --config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json \
    --index-path "$E4_ROOT/index.json" \
    --work-dir "$RUN_ROOT/benchmark_legacy_e4" \
    --output "$RUN_ROOT/benchmark_legacy_e4.json"

touch "$E1_ROOT/COMPLETED" "$E4_ROOT/COMPLETED" "$RUN_ROOT/COMPLETED"
printf 'VALIDATION_RUN_ROOT=%s\n' "$RUN_ROOT"
