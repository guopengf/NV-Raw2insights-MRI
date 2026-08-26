#!/bin/bash
set -euo pipefail

REPO=/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-windowed-hdf5
cd "$REPO"
sbatch --parsable build_4dflow_windowed_h5.slurm
