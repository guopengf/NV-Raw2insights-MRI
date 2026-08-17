#!/bin/bash
set -euo pipefail

REPO=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
SOURCE_REPO=/workspace/code/NV-Raw2insights-MRI-fork
PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
TORCHRUN=/root/miniconda3/envs/nv-raw2insights-mri/bin/torchrun
E1_INDEX=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2-canary/index.json
E4_INDEX=/data/CMRx4DFlow2026-ChallengeData/windowed-e4-v2-canary/index.json
RUN_ID=${1:?usage: validate_windowed_4dflow_gpu_interactive.sh RUN_ID}
RUN_ROOT=$REPO/outputs/4dflow/windowed_hdf5_gpu_validation/$RUN_ID

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/experiments"

"$PY" - "$RUN_ROOT" "$E1_INDEX" "$E4_INDEX" "$SOURCE_REPO" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
e1_index = sys.argv[2]
e4_index = sys.argv[3]
source_repo = Path(sys.argv[4])
repo = Path("/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5")

joint_source = repo / "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json"
legacy_source = repo / "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json"
cases = {
    "joint_e4": (joint_source, e4_index, "encoding_chunk_all"),
    "joint_e1": (joint_source, e1_index, "encoding_chunk_1"),
    "legacy_e1": (legacy_source, e1_index, "encoding_chunk_1"),
}
for name, (source, index_path, profile) in cases.items():
    config = json.loads(source.read_text())
    config["four_dflow_storage"]["index_path"] = index_path
    config["four_dflow_storage"]["storage_profile"] = profile
    config["exp_dir"] = str(run_root / "experiments")
    config["exp"] = name
    config["model_filename"] = f"{name}.pt"
    config["performance_timing"]["enabled"] = True
    config["performance_timing"]["sample_interval"] = 1
    config["performance_timing"]["debug_max_train_batches"] = 1
    config["performance_timing"]["worker_timing_enabled"] = True
    config["train_num_workers"] = 1
    config["val_num_workers"] = 0
    if name.startswith("joint"):
        config["resume_ckpt"] = str(
            source_repo
            / "outputs/4dflow/4dflow_finetune_phase4/small_ft_4dflow_encbatch_phase4_3d_flowvn_multiplane"
            / "nv_raw2insights_mri_small_ft_4dflow_phase4_3d_flowvn_multiplane_epoch200.pt"
        )
    else:
        config["resume_ckpt"] = str(
            source_repo
            / "cache/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_compatible.pt"
        )
    (run_root / "configs" / f"{name}.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
PY

for mode in joint_e4 joint_e1 legacy_e1; do
    started=$SECONDS
    "$TORCHRUN" --standalone --nproc_per_node=8 \
        scripts/train.py --debug --config "$RUN_ROOT/configs/$mode.json" \
        2>&1 | tee "$RUN_ROOT/$mode.log"
    printf 'real_seconds=%d\n' "$((SECONDS - started))" > "$RUN_ROOT/$mode.time"
    if find "$RUN_ROOT/experiments/$mode" -type f -name '*.pt' -print -quit | grep -q .; then
        echo "Unexpected checkpoint written for $mode" >&2
        exit 1
    fi
    grep -q 'checkpoint_saved=False' "$RUN_ROOT/$mode.log"
    grep -Eq 'train_loss=[0-9]' "$RUN_ROOT/$mode.log"
    if grep -Eqi 'train_loss=(nan|inf)|Traceback|CUDA out of memory|NCCL.*(error|timeout)' "$RUN_ROOT/$mode.log"; then
        echo "Non-finite loss or runtime failure for $mode" >&2
        exit 1
    fi
    touch "$RUN_ROOT/$mode.COMPLETED"
done

if find "$RUN_ROOT/experiments" -type f -name '*.pt' -print -quit | grep -q .; then
    echo "Unexpected checkpoint under GPU validation run root" >&2
    exit 1
fi
touch "$RUN_ROOT/COMPLETED"
printf 'GPU_VALIDATION_RUN_ROOT=%s\n' "$RUN_ROOT"
