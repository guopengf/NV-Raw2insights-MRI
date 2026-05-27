# VAA Implementation Log

Timestamp: 2026-05-27 22:56 Asia/Shanghai

## Implementation Content

Implemented Phase3 Vascular Attention Adapter (VAA) support for 4D Flow MRI reconstruction with config-controlled MRA generation/loading, model forwarding, freeze behavior, and optional complex phase-aware loss.

Default behavior remains baseline-compatible:

- `phase3.enable_vaa=false` does not generate/load MRA.
- `phase3.enable_vaa=false` does not instantiate/call VAA in the Restormer path.
- `phase3.enable_vaa=false` does not compute `deltaF`.
- Model behavior is directly `F'=F`.

## Modified Files

- `configs/nv_raw2insights_mri_base_4dflow.json`
- `configs/nv_raw2insights_mri_base_4dflow_pg.json`
- `scripts/inference.py`
- `scripts/mri_data/data_utils.py`
- `scripts/models/latent_recon.py`
- `scripts/models/restormer/restormer.py`
- `scripts/readers.py`
- `scripts/tools/generate_pc_mra_from_gt.py`
- `scripts/train.py`
- `scripts/train_utils.py`
- `scripts/transforms.py`
- `scripts/utils.py`

## New Files

- `scripts/models/vaa.py`
- `scripts/mra_utils.py`
- `logs/VAA_Implementation_20260527_2256.md`

## New Config

Added `phase3` config:

- `phase3.enable_vaa`: hard switch for VAA and MRA path.
- `phase3.mra.source`: supports `gt`, `zf`, `sense`, and `phase2` placeholder.
- `phase3.mra.projection`: `x_mip`.
- `phase3.mra.projection_axis`: `2`, because current 4D Flow code treats raw `x` as slice and keeps model plane as `zy`.
- `phase3.mra.vessel_map`: controls normalization, smoothing, threshold, and binary black/white output.
- `phase3.mra.sense`: controls CG-SENSE when `source=zf` or `source=sense`.
- `phase3.vaa.locations`: supports `bottleneck` and `intermediate`.
- `phase3.freeze.backbone` and `phase3.freeze.vaa`: independent freeze controls.
- `phase3.gamma`: controls zero-init learnable gamma.
- `phase3.loss`: controls phase and orthogonal SSIM loss terms.

## VAA Insertion Position

Implemented multi-layer VAA:

- `bottleneck`: after deepest encoder feature `feats[-1]`.
- `intermediate`: before refinement, using the top-level decoder/refinement input `ref_in`.

Rationale:

- Bottleneck features aggregate high-level global vascular context and are the lowest-cost first insertion point.
- Intermediate/refinement-level features retain higher spatial detail, useful for vessel boundaries and high-frequency flow patterns.
- This follows the adapter pattern from ControlNet/T2I-Adapter: condition features without replacing the backbone.

## Gamma

Implemented per-adapter learnable `gamma_raw`, initialized exactly at `0`.

Effective gamma uses shifted sigmoid mode:

- `gamma_raw=0` gives `gamma_eff=0`.
- Output is clamped to `[0,1]`.
- Forward form is `F' = F + gamma_eff * deltaF`.

The final adapter conv is also zero-initialized, so Stage1 starts from an exact baseline-equivalent behavior.

## Loss Modification

Added `Phase3ComplexLoss`:

- Magnitude loss: L1 on RSS magnitude.
- Phase loss: angular loss `1 - cos(phi_pred - phi_gt)`.
- `ssim_zy`: native model-plane SSIM on the current 2D reconstruction plane.
- `ssim_xy`: batch/window approximation of the orthogonal plane.

Important note:

The current data pipeline maps raw 4D Flow `x` to the sample/slice axis and uses `z,y` as the model image plane. Therefore:

- Native per-sample SSIM is `zy`.
- Orthogonal `xy` SSIM needs cross-slice context.
- This implementation uses a batch/window approximation for `xy`; full case-level 3D/cross-slice SSIM should be a later improvement.

## MRA Scheme

MRA prior is single-channel black/white vessel map:

```text
k-space or reconstruction
-> PC-MRA
-> x-MIP
-> normalize [0,1]
-> smoothing
-> threshold
-> M_vessel [1, z, y]
```

Source behavior:

- `source=gt`: full GT k-space -> direct coil-combined reconstruction -> PC-MRA -> x-MIP -> vessel map.
- `source=zf`: undersampled k-space -> CG-SENSE reconstruction -> PC-MRA -> x-MIP -> vessel map.
- `source=sense`: alias-style explicit SENSE path, same as `zf` implementation.
- `source=phase2`: config/interface placeholder only; not implemented in this round.

MRA is case-level and shared by all slice reconstructions from the same case.

## MRA Cache

Added cache in `scripts/mra_utils.py`.

- Cache path defaults to `/SSDHome/share/haosen/4dflow/mra_cache`.
- Cache key includes patient path, source, mask type, vessel map parameters, projection axis, and SENSE parameters.
- Cache is used only when `phase3.enable_vaa=true`.

## Checkpoint Compatibility

Old checkpoints remain compatible:

- New VAA parameters live under new adapter names.
- Existing `copy_model_state` and size-match load paths skip missing new keys safely.
- `restormer_mri.load_recon_model()` fills new keys from current initialization when old checkpoints do not contain them.
- `load_net()` continues to load compatible old weights and leaves new VAA/gamma parameters initialized.

## Inference Flow

When `phase3.enable_vaa=false`:

- No MRA generation.
- No MRA loading.
- No VAA call.
- No `deltaF` computation.

When `phase3.enable_vaa=true`:

- Reader generates/loads case-level x-MIP vessel map according to `phase3.mra.source`.
- Inference passes `mra_prior` into the model.
- VAA applies only at configured locations.

## Training Flow

When `phase3.enable_vaa=true`:

- Batch receives `mra_prior [1,z,y]`.
- The training loop expands it to `[B,1,z,y]` for each microbatch.
- Model forward receives `mra_prior`.
- Freeze policy is applied before optimizer construction.

Freeze modes supported:

- backbone frozen, VAA trainable.
- backbone trainable, VAA frozen.
- joint training.
- all frozen.

## Literature Basis

- ControlNet: zero-initialized conditional branches for safe backbone-preserving conditioning.
- T2I-Adapter: lightweight external-condition adapter pattern.
- LoRA: parameter-efficient adaptation and frozen-backbone training logic.
- Medical SAM Adapter / medical image adapters: domain-specific adapters are commonly inserted at deeper/intermediate representation stages rather than replacing the backbone.

## Validation

Completed:

- `python -m py_compile` on modified Python files.
- JSON validation for both 4D Flow configs.
- Conda `4dflow` import smoke test for `train` and `inference`.
- Lightweight Restormer+VAA forward test with `gamma_eff=0.0` at initialization.

Observed non-blocking warning:

- Matplotlib uses `/tmp` for cache because `/localhome/zhanghs/.config/matplotlib` is not writable.

## Follow-up Ablations

Recommended ablations:

- VAA off vs on.
- Bottleneck-only vs bottleneck+intermediate.
- GT prior vs ZF/SENSE prior.
- Phase loss off vs on.
- `ssim_zy` only vs `ssim_zy + ssim_xy`.
- Binary vessel map threshold sweep.
- Gamma mode `shifted_sigmoid` vs `direct_clamp`.
