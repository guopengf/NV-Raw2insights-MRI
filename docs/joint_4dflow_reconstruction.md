# Joint Four-Encoding 4D-Flow Reconstruction

## Scope

This implementation keeps the four challenge encodings aligned from the MAT reader through training, validation, and inference. It provides two model modes without changing the production training checkout:

- `batch`: flatten `[G,E,...]` to `[G*E,...]` at the model boundary. Existing model weights are unchanged, while the loss sees all four encodings together.
- `channel`: flatten at the public model boundary, regroup inside each cascade after coil reduction, and concatenate `E*T*C*2` into the 3D Restormer channel dimension. FlowVN and data consistency remain per encoding.

Both configs keep `batch_size=8` and `num_samples_per_case=8`. Since `E=4`, the grouped center batch is `G=2`; four grouped optimizer steps process the eight selected centers in a case. The grouped manifest has one JSON per case/acceleration instead of four encoding-specific JSONs, preserving approximately the same optimizer-step count per epoch.

## Tensor Contract

The reader loads raw k-space as `[E,T,C,Kz,Ky,Kx]`. After the readout IFFT and normalization, the dataset emits:

```text
input/target       [T*X,E,C,Z,Y,2]
mask               [T*X,E,1,Z,Y,1]
mean/std           [T*X,E,1,1,1,*]
sensitivity maps   [T*X,E,C,Z,Y,2]
```

An aligned slab/window gather produces `[G,E,S,T,C,Z,Y,2]`. The model receives `[G*E,S,T,C,Z,Y,2]` in both modes, so the cascade interface, coil sensitivity handling, and soft data consistency remain encoding-local.

Channel mode regroups only the reduced-coil backbone input:

```text
[G*E,S,T,C,Z,Y,2] -> [G,E*T*C*2,S,Z,Y]
```

The backbone output is restored to `[G*E,S,T,C,Z,Y,2]` before FlowVN subtraction, sensitivity expansion, and data consistency.

## Checkpoint Migration

Batch mode is shape-identical to the encoding-by-encoding model.

Channel mode changes only two kernels per cascade:

- `embed_conv.weight`: repeat the input-channel dimension four times and divide by four.
- `output.weight`: repeat the output-channel dimension four times.

For six cascades this inflates exactly 12 tensors. All other compatible tensors load normally. With identical encoding inputs, this initialization matches the batch-mode function up to GPU accumulation order.

## Joint Loss

The loss uses encoding 0 as the reference and encodings 1-3 as velocity-sensitive channels. It combines:

- target-amplitude-normalized complex error;
- target-amplitude-normalized magnitude error;
- circular error of reference-relative phase;
- normalized speed RMSE over the three relative phase channels;
- vector direction cosine error.

The optional vessel mask is broadcast over encoding and coil axes. Undefined phase/direction terms in zero-signal regions are ignored.

## Verification Evidence

All testing used the isolated worktree and interactive job `15826488`; no production checkpoint was read while being written.

- Eight direct joint tests passed on H100, including grouped indexing, hybrid conversion equivalence, finite loss gradients, checkpoint inflation, fixed cardinality, and export orientation.
- Channel bootstrap equivalence: max absolute drift `2.745e-4`, mean absolute drift `2.814e-5` on H100.
- Real case `Center007/GE_30T_Architect/P001` loaded as `1728 x 4` for input, target, mask, normalization statistics, and sensitivity maps in both modes.
- Full six-cascade batch smoke: flat batch 8, finite forward/backward, `17.956 GiB` peak allocated.
- Full six-cascade channel smoke: flat batch 8, exactly 12 inflated kernels, finite forward/backward, normalized joint loss `1.11535`, `7.031 GiB` peak allocated.
- Actual `scripts/train.py` channel diagnostic completed four grouped optimizer steps for eight centers, logged finite joint components and FlowVN gradients, and exited with `checkpoint_saved=False`.
- Actual `scripts/inference.py` channel diagnostic reconstructed two grouped centers with all four encodings and explicitly reported `saved=False` for the partial volume.
- Six existing Restormer/FlowVN/checkpoint regression tests passed.

The single-case smoke timings are functional evidence, not a throughput benchmark. Channel mode is the better first training candidate based on this memory result, but it should still be compared with batch mode over repeated full epochs before choosing the final production strategy.

## Configs

- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_pg.json`
- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json`
- `configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_smoke.json`

The production config and production output directory are not modified. The two training configs write under `outputs/4dflow/4dflow_joint_encoding/` and bootstrap from the stable compatible checkpoint. A later migration from the running experiment should use a verified snapshot after its writer has stopped.
