# Cluster Training: Nine Neural Configs

Branch: `haosen/baselin_ablation` in `guopengf/NV-Raw2insights-MRI`.
CG-SENSE and CS-TV are reconstruction-only baselines and are not in this job list.

## Prepare once on the training server

Use Python 3.12 with the project's GPU-enabled PyTorch environment. The destination
must have the original final dataset. The final runtime mask suite and its profiles
are tracked in the branch; no separate transfer or cy136 mount is required.
The repository does not contain the dataset or private source mapping.

```bash
git clone --branch haosen/baselin_ablation git@github.com:guopengf/NV-Raw2insights-MRI.git
cd NV-Raw2insights-MRI
conda activate 4dflow
python -m experiments.paper.bootstrap --fetch-authors --install-deps
export PAPER_PYTHON="$PWD/experiments/paper/.venv/bin/python"

export RUN_ROOT=/path/to/writable/scratch/4dflow-online
export DATA_ROOT=/path/to/final/4DFlow

"$PAPER_PYTHON" -m experiments.paper.build_virtual_dataset \
  --dataset-root "$DATA_ROOT" \
  --output "$RUN_ROOT" --verify-train-files
```

The direct root layout is `$DATA_ROOT/CenterXXX_Pxxxxxx_Anatomy/<files>` with the
unchanged final case IDs and files. For scattered raw source paths use an existing
`--mapping /path/to/source_mapping.csv` instead of `--dataset-root`; the mapping
must reflect paths visible on the training host. The checked-in final manifest
defines split membership. No Test contents are read in preparation or training.
If your configs were generated before the suite was bundled, pull this branch and
rerun setup with the original dataset root/mapping, omitting `--suite-path`.
That option remains available to explicitly override the bundled generator.

The setup checks all Train file sizes and small-case headers before queueing.
Keep `RUN_ROOT` outside the raw dataset and suite directories, with enough space
for per-experiment checkpoints. Prefer node-local SSD or fast scratch for active
checkpoints and a local dataset mirror when available. Do not launch many jobs
on the same slow NAS connection without checking aggregate bandwidth.

## One job per model

Request a single GPU per job; let the scheduler assign `CUDA_VISIBLE_DEVICES`.
Load the same environment in the job and run from the cloned repository:

```bash
export PAPER_PYTHON="$PWD/experiments/paper/.venv/bin/python"
export RUN_ROOT=/path/to/writable/scratch/4dflow-online
EXPERIMENT=flowvn
bash experiments/paper/scripts/cluster_train.sh "$EXPERIMENT" "$RUN_ROOT/configs" \
  --smoke --steps 1 --device cuda:0
# For formal training:
bash experiments/paper/scripts/cluster_train.sh "$EXPERIMENT" "$RUN_ROOT/configs" \
  --device cuda:0 --resume-latest
```

Smoke uses its own directory and does not seed the subsequent training run.
`--resume-latest` starts fresh when no formal checkpoint exists, or resumes at
the epoch boundary from that experiment's latest checkpoint. Use one live job
per experiment/output directory. Do not use torchrun or pass the cluster JSON
files to the legacy `scripts/train.py` entry. On the actual cluster, test the
largest Train grid on the assigned GPU before booking long jobs, for example add
`--smoke-case-id Center001_P000018_Cerebrovascular` to the smoke command.

All nine configs enable `paper.online_undersampling`: one of the four final
mask families and one real R in [10,50] per case per epoch; the four encodings
share points. Prior fixed whole-case evaluation settings are unchanged.

## Throughput changes

- Shuffle case order and window order within each case; sample centers, window
  counts, masks, targets and model/loss settings are unchanged. Grouping changes
  optimizer order, so this is not a bitwise replay of earlier runs.
- Open HDF5 and read coil maps, segmentation and metadata once per case block.
  Cache only transformed x slices needed by that case's windows. Crop-cache budget
  is 512 MiB; a separate HDF5 cache (up to 64 MiB with default settings) avoids
  repeatedly inflating multi-time compressed chunks. These are cache limits,
  not whole-process RAM limits; FlowMoDL still has large full-size training tensors.
- Prefetch one window in a host thread without multiprocessing /dev/shm; use
  bounded pinned transfers when the needed tensors fit the pin budget. The
  launcher defaults to two CPU threads and honors explicit OMP/MKL settings.
  Do not ship unused full-k-space arrays or log-only metadata to GPU.
- Use one nonfinite gradient norm check instead of synchronizing for each
  parameter; detailed activation/gradient probes stay in smoke. Logs report
  exposed loader wait and training time separately. Real I/O still depends on
  warm cache, compression and shared-storage bandwidth.
- `paper.checkpoint_every_epochs` defaults to 1 (and always saves the final
  epoch); increase only when accepting a longer recovery interval. Keep a
  separate output root for changed training settings.

## Queue Order

Suggested order for getting useful results early on a single queue. Use the
small GPU/time jobs first, then the 960-epoch models by measured memory tier.
On independent queues, start high-memory A2 (and A1 or FlowMoDL where they fit)
concurrently instead of leaving the largest jobs until the end.

| Order | Job | Epochs | Train s/window | Peak GPU GiB |
|---:|---|---:|---:|---:|
| 1 | `flowvn` | 100 | 0.56 | 3.43 |
| 2 | `cut_4dflow` | 100 | 1.69 | 1.54 |
| 3 | `flowmodl` | 50 | 8.19 | 3.78 |
| 4 | `ablation_A0` | 960 | 1.88 | 5.24 |
| 5 | `ablation_A3` | 960 | 2.22 | 6.07 |
| 6 | `ours_full_final_dataset` | 960 | 1.50 | 6.08 |
| 7 | `ablation_A1` | 960 | 2.16 | 9.00 |
| 8 | `ablation_A2` | 960 | 2.24 | 9.18 |
| 9 | `ablation_A4` (optional duplicate) | 960 | 1.52 | 6.07 |

These are two-step probes on the same small real Train case on a shared RTX
5880 Ada, not walltime forecasts or memory requirements for the largest case.
CUT uses eight windows across its two updates; seconds are divided by windows,
not optimizer steps. The separate data-wait time fluctuated with NAS traffic.
In particular FlowMoDL has the slowest window and uses a 64-slice input, but
its 50-epoch budget is shorter. Recheck it and the 3D jobs on large grids before
requesting resources. The full model and A4 currently differ only in the job
name/output paths; same model, objective and seed. Keep A4 as an available
config, but do not book both jobs just to duplicate one result. If an actual
repeat is needed, explicitly change its seed and record the protocol.

On the same NAS Train case, preparing eight S3/T5 windows required 160
encoding/time frame reads with individual windows vs. 88 with case reuse;
one local timed comparison was 56.1 s vs. 20.8 s. This compares data
preparation (no GPU); it is not a promised 2.7x training speedup.
54 unit/interface tests passed. All 694 Train cases passed the presence/size check. All nine GPU smoke
runs completed two steps with GT and the final suite; no full runs or held-out
evaluation were executed.
