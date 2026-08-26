# Windowed 4D-Flow HDF5 Pipeline

## Scope

This branch adds an opt-in training backend that converts each raw patient once
and writes two numerically identical, differently chunked HDF5 stores. Training
reads only the temporal and raw-x windows selected for the current sample. The
existing raw-MAT backend remains unchanged and is still used by old configs.

Both supplied training configurations keep `batch_size=8` and
`num_samples_per_case=8`. Joint four-encoding and legacy per-encoding training
both work with either storage profile when the config and index declare the same
profile.

## Production Destination

The restartable production converter is configured to write only the measured-fast E1 profile:

- E1 host root: `/home/pengfeig/healthcareeng_monai/datasets/CMRx4DFlow2026-ChallengeData/windowed-e1-v2`

Inside the training container this is mounted as
`/data/CMRx4DFlow2026-ChallengeData/windowed-e1-v2`. The converter CLI still
accepts an optional E4 output for controlled experiments, but the production
launcher intentionally omits it.

## Storage Profiles

Both profiles expose the same logical datasets:

```text
hybrid/target             [T, X, E, C, Z, Y, 2]
hybrid/input/<accel>      [T, X, E, C, Z, Y, 2]
mask/<accel>              [T, Z, Y]
coilmap                   [X, C, Z, Y, 2] or [X, E, C, Z, Y, 2]
segmask                   [X, Z, Y] (optional)
```

The profiles differ only in the hybrid dataset chunk shape:

- E1, `encoding_chunk_1`: `[1,1,1,C,Z,Y,2]`.
- E4, `encoding_chunk_all`: `[1,1,4,C,Z,Y,2]`.

During conversion, each source MAT array is opened and transformed once. The
result is written to a PID-scoped E1 temporary store. When the optional E4
output is requested, both profiles are written in the same source-read pass.
Each requested store is validated and atomically renamed only after all five acceleration
inputs, target, masks, coil map, optional segmentation, metadata, and source
provenance are complete. Restart skips verified stores; changed source size or
mtime requires explicit `--overwrite`.

## Activation

The recommended E1 configs are:

- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json`
- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json`

The controlled E4 joint alternative is:

- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_e4_pg.json`

The loader checks that `four_dflow_storage.storage_profile` matches the index
and every opened patient store. Old configs have no `four_dflow_storage` block,
so they continue through the raw-MAT reader and transform chain.

## Real-Patient Validation

The isolated canary conversion used `Center007/GE_30T_Architect/P001` with
logical shape `[16,108,4,10,30,112,2]` and accelerations 10, 20, 30, 40, and
50. Deep pair validation proved equal patient sets, source manifests, shapes,
accelerations, target/input probes, masks, coil maps, and segmentation data.

| Profile | Chunk shape | Canary size |
|---|---|---:|
| E1 | `[1,1,1,10,30,112,2]` | 11,179,576,090 bytes |
| E4 | `[1,1,4,10,30,112,2]` | 11,179,327,138 bytes |

The direct suite passes 31/31 tests. It covers exact paired-profile equality,
restart safety, config/index profile guards, joint and legacy numerical parity,
window indexing, unchanged raw-config routing, joint loss gradients, model
shape contracts, and checkpoint bootstrap/resume behavior.

## Loader Measurements

The original real-data raw-versus-windowed benchmark measured 194.055 s versus
8.722 s for five joint samples (22.25x) and 53.930 s versus 0.696 s for five
legacy samples (77.45x). Those numbers include cold raw MAT reads and the full
raw conversion path.

For the E1/E4 decision, four fresh-process runs used the same five real
acceleration samples and omitted the already measured raw baseline. The HDF5
page cache was warm for this relative comparison.

| Mode | E1 median | E4 median | E1 advantage |
|---|---:|---:|---:|
| Joint four-encoding | 1.336 s | 1.555 s | 14.1% |
| Legacy per-encoding | 0.280 s | 0.514 s | 45.5% |

An attempted h5py frame-fancy-index reader was rejected because it increased
joint time to about 43 s and legacy time to about 11 s. The retained reader uses
direct coordinate selections and reports the true read-call count.

## GPU Smoke

An isolated eight-GPU smoke on allocation `15948237` ran one real debug batch
for joint E4, joint E1, and legacy E1. All modes had finite loss, reached the
explicit debug stop with `checkpoint_saved=False`, and wrote no `.pt` files.

| Mode | Loader wait avg/max | Worker load avg/max | Wall time |
|---|---:|---:|---:|
| Joint E4 | 1.545/1.717 s | 1.149/1.180 s | 60 s |
| Joint E1 | 1.499/1.624 s | 1.107/1.161 s | 54 s |
| Legacy E1 | 0.347/0.417 s | 0.266/0.286 s | 43 s |

E1 is therefore the measured default for both modes. E4 remains fully
supported and is produced because it is a useful alternative for future reader
or compression experiments.

## Capacity And Launch

The canary stores are intentionally uncompressed to favor runtime throughput.
There are 138 eligible patients. A canary-size projection is about 1.54 TB for
the E1 production dataset; exact usage varies with geometry.

Reproducible launchers:

- `build_4dflow_windowed_h5.slurm`: E1-only restartable full conversion and deep verification.
- `submit_4dflow_windowed_h5.sh`: thin submission wrapper.
- `validate_windowed_4dflow_cpu.slurm`: paired canary and CPU benchmarks.
- `validate_windowed_4dflow_gpu.slurm`: one-batch joint/legacy GPU smoke.
- `validate_windowed_4dflow_interactive.sh`: validation inside an existing allocation.
- `validate_windowed_4dflow_gpu_interactive.sh`: E1/E4 GPU comparison inside an existing allocation.
- `validate_windowed_4dflow_e1_only_cpu.slurm`: E1-only CLI and canary verification.

The full 138-patient conversion was not launched during validation.
