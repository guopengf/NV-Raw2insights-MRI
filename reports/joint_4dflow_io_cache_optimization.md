# Joint 4D-Flow I/O Cache Optimization

## Scope

This change is isolated on branch `codex/joint-encoding-io-cache` in
`/home/pengfeig/workspace/code/NV-Raw2insights-MRI-fork-joint-io-cache`.
The shared production checkout, running jobs, production configs, and rolling checkpoints were not modified.

Both joint configs keep `batch_size=8` and `num_samples_per_case=8`.

## Implementation

- Group the five acceleration manifests that share one fully sampled target.
- Partition grouped manifests into equal per-rank lengths while preserving adjacent target chunks.
- Shuffle target chunks per epoch with `TargetGroupedSampler` without splitting a chunk.
- Cache one raw full-target MAT array and one coil-map MAT array in each persistent reader worker.
- Retain acceleration-specific input and mask reads for every manifest.
- Add independent train/validation worker, prefetch, and pin-memory controls.
- Use one persistent train worker per rank and lazy zero-worker validation for the joint configs.

Recommended joint-loader settings:

```json
{
  "num_workers": 1,
  "train_num_workers": 1,
  "val_num_workers": 0,
  "train_prefetch_factor": 1,
  "val_prefetch_factor": 1,
  "lazy_val_loader": true,
  "group_accelerations_by_target": true,
  "reader_target_cache_entries": 1,
  "reader_coilmap_cache_entries": 1
}
```

The one-worker setting is intentional. With multiple workers, consecutive acceleration variants are distributed across workers, so each worker may independently reread the same multi-GiB target.

## Reader Benchmark

Command: `scripts/tools/benchmark_joint_4dflow_reader_cache.py`

The benchmark used all five accelerations for one median-size patient target (`1.910 GiB`) on `pool0-01941`.

| Metric | Cache off | Cache on | Change |
|---|---:|---:|---:|
| Full-target HDF5 reads | 5 | 1 | -80% |
| Coil-map HDF5 reads | 5 | 1 | -80% |
| Total target-read time | 40.170 s | 7.675 s | -80.9% |
| Five-manifest reader wall time | 75.132 s | 39.837 s | -47.0% |
| Target cache-hit latency | n/a | 0.02-0.05 ms | effectively free |

The remaining time is expected: each acceleration has a distinct undersampled input, so five input reads cannot be removed by a target cache.

For the full 690-manifest dataset on 16 DDP ranks, equal-length rank partitioning produces 153 target-homogeneous chunks. The expected target-read count is therefore about 153 instead of 690, a 77.8% reduction. Boundary chunks are the small gap from the ideal 138 reads.

## End-to-End Diagnostic

An eight-GPU debug run loaded the read-only epoch-200 checkpoint, restored all 1,728 network tensors with zero unchanged keys, and processed three real grouped cases per rank.

- 24 worker events: exactly 3 per rank.
- 13 target/coil-map hits and 11 misses in the deliberately short rank-boundary sample.
- Finite joint losses and finite FlowVN gradients at accumulation boundaries.
- Peak CUDA allocation: approximately 22.7 GiB per GPU.
- Clean debug stop with `checkpoint_saved=False`; no `.pt` file was written.
- Worker timing, slow-loader timing, and lightweight timing JSONL files were produced under the isolated output root.

The diagnostic also exposed a missing `TargetGroupedSampler.set_epoch()` method before data loading. The sampler was fixed to use deterministic `seed + epoch` ordering, its regression test was extended, and the complete diagnostic then passed.

## Remaining Bottleneck

Raw target reads are no longer the only dominant cost. In the end-to-end run, every acceleration still repeated target complex conversion and full-volume transforms. Representative per-case averages included roughly 3.4 s target complex conversion, 18.5 s hybrid-mask processing, 3.2 s tensor conversion, 7.6 s IFFT, and 3.9 s rearrangement/normalization.

A later optimization can cache invariant post-read target preprocessing for the current patient, but it should be measured carefully because the transformed tensors are substantially larger than the raw-array cache and may increase worker RSS.

## Compatibility

Old experiment configs do not opt into grouping or reader caching:

- `group_accelerations_by_target` defaults to false.
- Both reader cache capacities default to zero.
- Separate loader controls fall back to the existing `num_workers` value.
- Validation defaults retain the former multi-epoch loader, prefetch, pin-memory, and persistent-worker behavior.
- Non-joint 4D-flow target reads retain encoding selection and do not load/cache the full four-encoding target.

## Verification

- 12 joint-encoding unit/regression tests passed, including explicit non-joint encoding-slice coverage.
- 12 pre-existing FlowVN/old-path regression tests passed.
- Python syntax checks, JSON validation, and `git diff --check` passed.
- Reproducible debug launcher: `validate_joint_4dflow_io_cache.slurm`.
