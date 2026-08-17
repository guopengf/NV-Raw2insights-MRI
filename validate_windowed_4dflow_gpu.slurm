#!/bin/bash
#SBATCH --account=healthcareeng_monai
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --job-name=windowed4d-gpu
#SBATCH --output=/home/pengfeig/workspace/slurm-logs-agent/%j-windowed4d-gpu.out

set -euo pipefail

IMAGE=/home/pengfeig/workspace/cache/conda_raw2insights.sqsh
USER_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pengfeig
DATA_ROOT=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData
REPO=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
SOURCE_REPO=/workspace/code/NV-Raw2insights-MRI-fork
CANARY_E1_ROOT=/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2-canary
RUN_ROOT=/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5/outputs/4dflow/windowed_hdf5_gpu_validation/${SLURM_JOB_ID}

srun --kill-on-bad-exit=1 \
    --container-image "$IMAGE" \
    --container-mounts="/home/pengfeig/.netrc:/root/.netrc,${USER_ROOT}:/workspace,${DATA_ROOT}:/data/CMRx4DFlow2026-ChallengeData" \
    --no-container-mount-home \
    --container-remap-root \
    bash -lc "
        set -euo pipefail
        cd '$REPO'
        PY=/root/miniconda3/envs/nv-raw2insights-mri/bin/python
        TORCHRUN=/root/miniconda3/envs/nv-raw2insights-mri/bin/torchrun
        export WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        mkdir -p '$RUN_ROOT/configs'
        \"\$PY\" scripts/tools/run_direct_regression_tests.py | tee '$RUN_ROOT/regressions.log'

        \"\$PY\" - <<'PY'
import json
from pathlib import Path

repo = Path('/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5')
source_repo = Path('/workspace/code/NV-Raw2insights-MRI-fork')
run_root = Path('$RUN_ROOT')
canary_indices = {
    'joint': Path('$CANARY_E1_ROOT/index.json'),
    'legacy': Path('$CANARY_E1_ROOT/index.json'),
}
configs = {
    'joint': repo / 'configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json',
    'legacy': repo / 'configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json',
}
for name, source in configs.items():
    cfg = json.loads(source.read_text())
    cfg['four_dflow_storage']['index_path'] = str(canary_indices[name])
    cfg['exp_dir'] = str(run_root / 'experiments')
    cfg['exp'] = name
    cfg['model_filename'] = f'{name}.pt'
    cfg['performance_timing']['enabled'] = True
    cfg['performance_timing']['debug_max_train_batches'] = 1
    cfg['performance_timing']['worker_timing_enabled'] = True
    cfg['train_num_workers'] = 1
    cfg['val_num_workers'] = 0
    if name == 'joint':
        cfg['resume_ckpt'] = str(source_repo / 'outputs/4dflow/4dflow_finetune_phase4/small_ft_4dflow_encbatch_phase4_3d_flowvn_multiplane/nv_raw2insights_mri_small_ft_4dflow_phase4_3d_flowvn_multiplane_epoch200.pt')
    else:
        cfg['resume_ckpt'] = str(source_repo / 'cache/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_compatible.pt')
    destination = run_root / 'configs' / f'{name}.json'
    destination.write_text(json.dumps(cfg, indent=2) + '\n')
PY

        for MODE in joint legacy; do
            \"\$TORCHRUN\" --standalone --nproc_per_node=8 \
                scripts/train.py --debug --config '$RUN_ROOT/configs/'\"\$MODE\"'.json' \
                2>&1 | tee '$RUN_ROOT/'\"\$MODE\"'.log'
            if find '$RUN_ROOT/experiments/'\"\$MODE\" -type f -name '*.pt' -print -quit | grep -q .; then
                echo \"Unexpected checkpoint written for \$MODE\" >&2
                exit 1
            fi
            grep -q 'checkpoint_saved=False' '$RUN_ROOT/'\"\$MODE\"'.log'
        done
        touch '$RUN_ROOT/COMPLETED'
    "
