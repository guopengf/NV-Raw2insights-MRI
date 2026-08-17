# Windowed 4D-Flow HDF5 Pipeline

## Scope

This branch adds an opt-in training backend that converts raw per-patient MAT
files into chunked HDF5 stores and reads only the temporal/slice windows used by
the model. The existing raw-MAT backend remains the default.

Both supplied configurations keep `batch_size=8` and
`num_samples_per_case=8`. The backend supports:

- joint four-encoding training, returning `[sample, encoding, slice, time, ...]`;
- legacy per-encoding training, selecting one encoding from the same store;
- raw-MAT validation while the v1 training backend is evaluated.

## Storage Layout

Each patient store contains:

```text
hybrid/target             [T, X, E, C, Z, Y, 2]
hybrid/input/<accel>      [T, X, E, C, Z, Y, 2]
mask/<accel>              [T, Z, Y]
coilmap                   [X, C, Z, Y, 2] or [X, E, C, Z, Y, 2]
segmask                   [X, Z, Y] (optional)
```

K-space is transformed along the readout axis once during conversion. Training
workers read only the selected `(time, raw-x slice)` chunks, perform the
remaining 2D IFFT, normalize with the same per-frame/per-encoding statistics as
the raw pipeline, and expand compact masks/maps only for those windows.

## Safety

- Stores are written to a PID-scoped temporary file and atomically renamed.
- `complete=true`, schema version, shape, finite probes, and source
  size/mtime provenance are validated before reuse.
- A source change requires an explicit `--overwrite` conversion.
- The HDF5 backend rejects data augmentation, non-fixed masks, MRA/VAA priors,
  non-slab reconstruction, and HDF5 validation in schema v1.
- Debug validation asserts that no checkpoint is written.

## Activation

The new configs are:

- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json`
- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_windowed_h5_pg.json`

Old configs have no `four_dflow_storage` block and therefore continue through
the unchanged raw-MAT reader and transform chain.

## Validation Results

CPU validation job `15953713` completed `0:0` on `cpu_interactive` in 7m08s.
It passed 28 direct regression tests, including numerical parity between the
windowed and full-volume pipelines in both joint and per-encoding modes.
Follow-up regression job `15954226` completed `0:0` in 44s with 29/29 tests,
adding explicit proof that old configs remain on `raw_mat` and only the two new
configs opt into HDF5 training.

One real patient was converted and deep-verified:

- patient: `Center007/GE_30T_Architect/P001`;
- shape: `[16,108,4,10,30,112,2]`;
- five accelerations: `10,20,30,40,50`;
- store size: 11,179,576,090 bytes;
- output was hidden until an atomic rename and then indexed with source
  size/mtime provenance.

The loader benchmark processed the five acceleration variants in one process.
It measures MAT read plus the existing full-volume hybrid conversion, mask
expansion, IFFT, tensor conversion, and normalization against the new selected
window path.

| Mode | Raw total | Windowed total | Speedup |
|---|---:|---:|---:|
| Joint four-encoding | 194.055 s | 8.722 s | 22.25x |
| Legacy per-encoding | 53.930 s | 0.696 s | 77.45x |

Joint windowed cases took 1.47-2.35 s each and materialized
`[8,4,3,5,10,30,112,2]`. Legacy cases took 0.123-0.172 s each and materialized
`[8,3,5,10,30,112,2]`. The raw path materialized all 1,728 `(time, raw-x)`
positions before training selected eight centers.

The canary store is intentionally uncompressed to favor runtime throughput. It
is much larger than the sparse/compressed source MAT files. There are 138
eligible patients; a simple canary-size projection is about 1.54 TB, although
the exact total varies with patient geometry. The user fsw quota check showed
54.2 TB used of 149 TB, so capacity is available. GPU smoke validation is still
pending.

Reproducible launchers:

- `build_4dflow_windowed_h5.slurm`: restartable full-dataset conversion into
  `/workspace/datasets/CMRx4DFlow2026-windowed-v1` after canary approval.
- `submit_4dflow_windowed_h5.sh`: thin submission wrapper for that conversion.
- `validate_windowed_4dflow_cpu.slurm`: syntax/regression tests, one real
  patient conversion, deep validation, and raw-vs-windowed loader benchmarks.
- `validate_windowed_4dflow_regressions.slurm`: short CPU-only direct test run.
- `validate_windowed_4dflow_gpu.slurm`: one debug train case for joint and
  legacy modes on eight GPUs, with no-checkpoint assertions.
