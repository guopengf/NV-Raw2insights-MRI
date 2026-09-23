# Windowed 4D-Flow HDF5 Pipeline

## Scope

This branch adds an opt-in training backend that converts each raw patient once
and can write numerically identical, differently chunked HDF5 stores. Production
writes only the measured-fast E1 profile; E4 remains available for controlled
experiments. Training reads only the temporal and raw-x windows selected for the
current sample. The existing raw-MAT backend remains unchanged and is still used
by old configs.

Both supplied training configurations keep `batch_size=8` and
`num_samples_per_case=8`. Joint four-encoding and legacy per-encoding training
both work with either storage profile when the config and index declare the same
profile.

## Production Destination

The completed unified production converter wrote only the measured-fast E1
profile. The verified production copy is now served from `healthcareeng_monai`:

- Run ID: `unified_e1_20260921T1325Z`
- Canonical E1 host root:
  `/home/pengfeig/healthcareeng_monai/datasets/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2`
- Resolved E1 host root:
  `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2`
- Retained source root:
  `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/datasets/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2`

Inside the training container this is mounted as
`/h5data/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2`.
The converter CLI still accepts an optional E4 output for controlled
experiments, but the production launcher intentionally omits it. The later
copy-verify-switch migration retained the completed Isaac source and promoted
an independently verified MONAI copy; the in-container path did not change.

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
supported and can be produced as a useful alternative for future reader or
compression experiments.

## Unified Production Completion

Production completed on 2026-09-22. The effective successful chain was prepare
job `1278628`, original shard-array tasks 0 and 1, resumed shard-array tasks 2
through 6, recovery shard-7 job `1293377`, and finalizer retry `1293475`.
The original pause cancelled the remaining original array tasks and original
finalizer. The first resumed shard 7 reached its 12-hour limit after writing
32/36 stores; its restart-safe recovery reused those stores and wrote only the
four missing patients in 9m48s.

Finalizer `1293378` completed deep HDF5 verification and real-loader validation,
but exited 1 when `tee` attempted to open `validation/validate.log` before the
validation directory existed. The launcher now creates that directory before
the container step and safely reuses an already completed HDF5 finalization.
Retry `1293475` then completed in 2m42s with exit code 0 and created the
control-root `COMPLETED` marker.

The final inventory is:

| Acceleration profile | Patients |
|---|---:|
| 10, 20, 30, 40, 50 | 208 |
| 10 only | 19 |
| 20 only | 19 |
| 30 only | 16 |
| 40 only | 14 |
| 50 only | 16 |
| **Total** | **292** |

The index contains exactly 292 unique patient stores. Their logical HDF5 byte
sum is 4,980,443,982,857 bytes; measured host usage for the production root is
4,980,445,318,144 bytes. The production root contains no `.tmp` or `.partial`
files. `index.json`, the production `COMPLETED` receipt, and the control-root
`COMPLETED` marker are all present.

Validation receipt
`outputs/4dflow/windowed_h5_unified_production/unified_e1_20260921T1325Z/validation/receipt.json`
has `status: ok`, 292 patients, 1,124 training manifests, 63 validation
patients, and 259 validation manifests. It loaded finite samples for all six
acceleration-profile classes. No compatible training checkpoint was present at
the configured `cache/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_compatible.pt`
path or elsewhere under the workspace search scope, so no checkpoint-backed
forward/backward smoke was claimed. The earlier six-profile canary had already
passed 6/6 raw-probe parity and finite DataLoader sampling.

Closure hashes:

- `index.json`: `7300ab8d8014c6486826888c70671836d2e73a2a9c88e3cb92dd3a86aed6f414`
- conversion plan file:
  `8d9d48319ad19aba672de7b8ae2075754c20629bc9fa0fef31a989d6c0b220b7`
- production config:
  `21b37949beb4ee9cde1b2da1f010f242be7aae155a0456824ffe99a1fc2f9cd1`
- validation receipt:
  `7a89f2ab7a5bd93501035616aae14d51635f47932a392f5b9c82e5d3e05395d3`

Three interrupted temporary files remain preserved outside the production root
in the two timestamped `.windowed-e1-v2-*-partials-*` quarantine directories.
They were not deleted.

## Verified MONAI Copy-Verify-Switch

Migration `monai_copy_20260922T201002Z` completed on 2026-09-22 under Slurm job
`1297822` in 5h58m35s with exit code 0 using workflow commit
`7c565c18276a1cff2cfb57b63f2f76b1398b2654`. The restartable migration copied
into a hidden staging directory without `--delete`, compared exact
relative-path and size inventories, read all 4.98 TB again with an rsync
checksum dry-run, and only then atomically promoted the staging directory to
the canonical MONAI destination.

The source and destination inventories are byte-identical and share SHA-256
`eaa14c58167d2dc1fc67b6d58406526c237feb36ed73bed186e130e101a14892`.
The checksum difference file is empty. Both roots contain exactly 292 H5
stores, zero `.tmp` or `.partial` files, a production `COMPLETED` marker, and an
`index.json` with SHA-256
`7300ab8d8014c6486826888c70671836d2e73a2a9c88e3cb92dd3a86aed6f414`.
Both have measured host usage of 4,980,445,318,144 bytes.

The real six-profile loader validation reran against the MONAI copy and wrote
`outputs/4dflow/windowed_h5_unified_migration/monai_copy_20260922T201002Z/validation/receipt.json`
with `status: ok`, 292 patients, 1,124 training manifests, 63 validation
patients, and 259 validation manifests. The migration receipt has `status: ok`
and `source_retained: true`; the migration control root has its `COMPLETED`
marker. Its path is
`outputs/4dflow/windowed_h5_unified_migration/monai_copy_20260922T201002Z/migration-receipt.json`
and its SHA-256 is
`7c28edeb4a3e1bf126c13a374d4891823b2cf28ec8b131fae13905fa6e2449b7`.
The original Isaac dataset and all quarantined interrupted partials remain
intact. No source cleanup was performed.

After these gates passed, the canonical host-side launchers were switched from
the Isaac H5 root to the MONAI H5 root. The stable container mount remains
`/h5data/CMRx4DFlow2026-unified-70_15_15-seed20260914/windowed-e1-v2`.

Reproducible launchers:

- `scripts/tools/launch_4dflow_unified_h5_prepare.sh`
- `scripts/tools/launch_4dflow_unified_h5_shard.sh`
- `scripts/tools/launch_4dflow_unified_h5_finalize.sh`
- `scripts/tools/submit_4dflow_unified_h5_pipeline.sh`
