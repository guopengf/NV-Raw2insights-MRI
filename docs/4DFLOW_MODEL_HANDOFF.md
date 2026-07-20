# NV-Raw2Insights-MRI 4D Flow Handoff

更新时间：2026-06-29
项目路径：`/localhome/zhanghs/NV-Raw2insights-MRI`

这个文档用于把当前 4D Flow MRI 任务交接给其他模型。读完此文件后，应该能立即理解当前代码的核心约定、已经做过的 Phase3/VAA/MRA/loss/export/evaluation 改动，以及后续修改时不能破坏的地方。

## 1. 任务背景

当前项目基于 `NV-Raw2insights-MRI`，已经适配 CMRx4DFlow2026 4D Flow MRI reconstruction。

核心目标：

- 输入 4D Flow raw k-space，输出官方 challenge 要求的 complex image reconstruction。
- 保留 complex real/imag，不要只保存 magnitude/abs，因为 phase 包含 velocity/flow 信息。
- 在 Phase3 中探索 vascular-aware adaptation：
  - VAA：Vascular Attention Adapter。
  - MRA/PC-MRA vessel prior。
  - phase/complex loss。
  - vascular loss ablation。

## 2. 绝对不能改错的数据约定

### 2.1 原始 4D Flow k-space 维度

原始 4D Flow k-space 是 6D：

```text
(enc, t, coil, kz, ky, kx)
```

含义：

- `enc`：velocity encoding，通常 4 个。
- `t`：cardiac temporal phase。
- `coil`：receiver coil。
- `kz, ky`：phase encoding directions。
- `kx`：fully sampled readout direction。

### 2.2 encoding 顺序已经确认

不要再当成未知。

```text
enc0 = reference
enc1 = RL
enc2 = AP
enc3 = HF
```

来自 metadata：

```text
venc_order = RL;AP;HF
```

PC-MRA 中：

```python
phase_diff = angle(img[1:] * conj(img[0]))
```

即 reference 到 RL/AP/HF 的 phase contrast。

日志里建议打印：

```text
reference enc = 0
velocity encodings = [RL, AP, HF]
```

### 2.3 x/z 交换是当前 4D Flow 适配的核心，不要改

当前代码对 raw k-space 做：

```text
(enc,t,coil,kz,ky,kx)
-- IFFT along raw kx/FE -->
(enc,t,x,coil,kz,ky)
```

然后每个 JSON 选择一个 `encoding_idx`，保留 singleton enc：

```text
(1,t,coil,kz,ky,kx)
```

经过 transform 后，模型看到的是：

```text
(t, x, coil, z, y)
```

也就是说：

- raw `x/FE` 被当作 slice 维度。
- 模型每次重建的空间平面是 `z-y`。
- 用户之前反复确认：加速方向是 `z-y`，输入时 `x` 和 `z` 的角色已经交换，不能改回去。

因此：

- native 2D image-plane loss 是 `zy`。
- 如果要做 MRA prior，默认应是 `zy` 方向的 vessel map。
- 如果 `use_mip=true`，应该是沿 raw `x` 做 MIP，也就是 `x-MIP`，不是 z-MIP。
- 如果 `use_mip=false`，每个 raw-x slice 使用对应的 `zy` MRA slice。

### 2.4 模型输入/输出粒度

模型输入是同一个 raw-x slice 的多个 temporal frames，不是一次输入完整 3D volume。

典型设置：

```json
"num_frames": 5
```

每个 training/inference sample：

```text
input:  around-one-slice temporal window, shape roughly [B, num_frames, coil/channel, z, y, 2]
output: center temporal frame for same raw-x slice, complex real/imag
```

推理完整 case 时，代码遍历：

- all encodings。
- all accelerations。
- all raw-x slices。
- all temporal frames。

最后再把 per-slice/per-time 结果重新组装成 4D/5D output。

## 3. 当前重要文件

### 3.1 训练/推理主流程

- `scripts/train.py`
  - training loop。
  - phase3 loss 开关。
  - VAA/MRA prior forward。
  - gamma log。
- `scripts/inference.py`
  - inference loop。
  - 现在应保留 complex output，不应只 `abs` 后保存。
- `scripts/train_utils.py`
  - transforms。
  - optimizer。
  - `apply_phase3_freeze()`。
- `scripts/utils.py`
  - `windowed_input()`。
  - `select_mra_prior_for_microbatch()`。
  - miscellaneous helpers。
- `scripts/mri_data/data_utils.py`
  - loss functions。
  - `FlowVNPhaseLoss`。
  - postprocess / crop / metric helpers。
- `scripts/readers.py`
  - 4D Flow JSON reader。
  - fast IO: select `encoding_idx` before reading full `.mat` array。

### 3.2 Model/VAA

- `scripts/models/restormer/restormer.py`
  - Restormer backbone。
  - Phase3 config is passed into Restormer。
  - VAA insertion:
    - `bottleneck`: after `feats[-1]`。
    - `intermediate`: before refinement on top-level decoder/refinement input。
- `scripts/models/vaa.py`
  - `VascularAttentionAdapter`。
  - residual adapter:

```text
F' = F + gamma * deltaF
deltaF = VAA(F, MRA)
```

### 3.3 MRA/PC-MRA

- `scripts/mra_utils.py`
  - `generate_or_load_mra_prior()`。
  - PC-MRA generation。
  - source=`gt/zf/sense/phase2 placeholder`。
  - cache。
- `scripts/tools/generate_pc_mra_from_gt.py`
  - single-case PC-MRA visual test script。
  - 用于先验证 MRA 视觉合理性。

### 3.4 Official submission / evaluation

- `scripts/tools/fix_4dflow_recon_coil_dim.py`
  - legacy repair tool，修复旧 inference `.mat` 或 `--preserve-multicoil-output` 输出里残留的 coil dimension。
  - 输入常见 shape：

```text
(PE,SPE,FE,Nt,Nc,2) = (y,z,x,t,coil,real/imag)
```

  - 输出：

```text
(PE,SPE,FE,Nt,2) = (y,z,x,t,real/imag)
```

- `scripts/tools/export_4dflow_submission.py`
  - 把 coil-combined per-enc `.mat` 转换成官方 sparse `.npz`。
  - 输入 per-encoding recon layout 默认 `yzxt`。
  - 输出官方 dense 逻辑 shape：

```text
(Nv,Nt,SPE,PE,FE) = (enc,t,z,y,x)
```

  - 保存为 sparse COO `.npz`。
  - 会乘 `segmask`，官方评测只看 mask 区域。
- `scripts/tools/evaluate_4dflow_submission.py`
  - 调用 official EvaluationCode 逻辑计算：
    - SSIM
    - nRMSE
    - RelErr
    - AngErr
    - optional ComplexDiffErr

### 3.5 Config

当前主要 4D Flow config：

- `configs/nv_raw2insights_mri_base_4dflow_pg.json`
  - 当前偏 VAA 实验。
  - `phase3.enable_vaa=true`。
  - 默认 `use_ssim_zy=true`，`use_phase=false`，`use_vascular=false`。
- `configs/nv_raw2insights_mri_base_4dflow_haosen.json`
  - 当前偏 no-VAA phase/complex loss 实验。
  - `phase3.enable_vaa=false`。
  - 默认 `use_phase=true`，`use_ssim_zy=false`，`use_vascular=false`。

## 4. Phase3 config 设计

所有 MRA/VAA/loss/freeze 行为必须通过 config 控制，不要硬编码。

典型结构：

```json
"phase3": {
  "enable_vaa": false,
  "mra": {
    "source": "gt",
    "method": "pcmra",
    "use_mip": false,
    "projection": "slice_zy",
    "projection_axis": 2,
    "use_cache": true,
    "cache_dir": "outputs/4dflow/mra_cache",
    "device": "cpu",
    "vessel_map": {
      "binary": true,
      "lower_percentile": 1.0,
      "upper_percentile": 99.5,
      "smooth_sigma": 0.75,
      "threshold": 0.35
    },
    "sense": {
      "niter": 5,
      "lam": 0.0001
    }
  },
  "vaa": {
    "locations": ["bottleneck", "intermediate"],
    "reduction": 4
  },
  "freeze": {
    "backbone": true,
    "vaa": false
  },
  "gamma": {
    "init": 0.0,
    "trainable": true,
    "mode": "shifted_sigmoid"
  },
  "loss": {
    "use_phase": true,
    "use_vascular": false,
    "use_ssim_zy": false,
    "phase": {
      "method": "flowvn_complex_l1",
      "weight": 1.0,
      "eps": 1e-8
    },
    "vascular": {
      "method": "mra_masked_phase_l1",
      "weight": 1.0,
      "normalize_by_mask": true
    }
  }
}
```

## 5. VAA 行为

### 5.1 enable_vaa=false 必须是 hard bypass

当：

```json
"phase3": {
  "enable_vaa": false
}
```

要求：

- 不生成 MRA。
- 不读取 MRA cache。
- 不 forward VAA。
- 不计算 `deltaF`。
- 直接等价：

```text
F' = F
```

不是：

```text
F' = F + 0 * deltaF
```

因为后者仍会产生额外计算。

当前代码：

- `scripts/mra_utils.py:phase3_enabled()` 控制 MRA 是否生成。
- `Restormer.apply_vaa()` 中如果 `enable_vaa=false` 或 location 不存在，直接返回原 feature。
- `apply_phase3_freeze()` 如果 `enable_vaa=false` 直接 return，不会冻结 backbone，也不会影响 loss 选择。

### 5.2 VAA 结构

文件：`scripts/models/vaa.py`

结构：

```text
(C + 1)
 -> 1x1 conv to C/reduction
 -> GELU
 -> 3x3 conv
 -> GELU
 -> 1x1 conv to C
```

输入：

- `F`: backbone feature `[B,C,H,W]`。
- `MRA/vessel_map`: `[B,1,H,W]`，会 interpolate 到 feature spatial size。

输出：

```text
F' = F + gamma * deltaF
```

### 5.3 gamma

用户要求过：gamma 数学上必须从 0 开始，baseline 初始严格等价。

当前代码支持：

- `direct_clamp`
- `shifted_sigmoid`

当前 configs 使用：

```json
"mode": "shifted_sigmoid"
```

其中 `gamma_raw=0` 时 `gamma_eff=0`，范围 clamp 到 `[0,1]`。

训练日志会打印：

- `gb` / `gamma_bottleneck`
- `gi` / `gamma_intermediate`

它们分别代表 bottleneck VAA 和 intermediate VAA 的有效 gamma 平均值。

### 5.4 VAA 插入位置

当前实现 multi-layer VAA：

1. `bottleneck`
   - deepest feature：`feats[-1]`。
   - 低成本、高层语义、适合 vascular global context。
2. `intermediate`
   - decoder/refinement 前较高分辨率 feature。
   - 更有利于 vessel boundary 和 high-frequency flow pattern。

推荐 ablation：

- VAA off baseline。
- bottleneck only。
- bottleneck + intermediate。
- GT prior vs ZF/SENSE prior。
- phase loss off/on。
- vascular loss off/on。

## 6. MRA / PC-MRA 设计

### 6.1 PC-MRA 公式

由 complex image 生成：

```python
mag = mean(abs(img), dim=enc)
phase_diff = angle(img[1:4] * conj(img[0:1]))
flow = sqrt(sum(phase_diff**2, dim=enc_velocity))
pc_mra = mag * flow
```

### 6.2 source 含义

`phase3.mra.source` 支持：

- `gt`
  - full k-space / GT reconstruction -> PC-MRA。
  - Oracle prior。
  - 不调用 CG-SENSE。
- `zf`
  - undersampled k-space -> CG-SENSE -> PC-MRA。
  - deployable prior。
  - 用户明确要求：source=`zf` 时正式 prior 默认用 SENSE enhanced image，不直接用 plain ZF。
- `sense`
  - 显式 SENSE 路径，目前实现等价于 `zf` 的 CG-SENSE prior。
- `phase2`
  - 只保留 config/interface placeholder。
  - 本轮没有实现从 Phase2 complex reconstruction 生成 z/x-MIP cache。

### 6.3 use_mip

当前用户后续改过要求：

- 不一定用 MIP。
- 可以 `use_mip=false`，每个 raw-x slice 使用对应的 `zy` vessel map。
- `select_mra_prior_for_microbatch()` 支持：
  - `[1,z,y]`：case-level shared MIP prior。
  - `[x,z,y]`：per-slice MRA prior。

### 6.4 vessel map

Vessel map 是单通道黑白/灰度图，不是 RGB。

```text
PC-MRA
 -> x-MIP or per-slice
 -> percentile normalize [0,1]
 -> optional Gaussian smoothing
 -> optional threshold to binary
 -> M_vessel
```

config 参数：

- `lower_percentile` / `upper_percentile`：归一化范围。
- `smooth_sigma`：平滑。
- `threshold`：二值化阈值。
- `binary`：是否输出黑白二值。

## 7. Loss 当前状态

### 7.1 main zy loss

`phase3.loss.use_ssim_zy`

- 这是当前模型 native plane 的默认 image loss。
- 它对应模型当前每次看到的 `z-y` 平面。
- 如果用户想只用 complex L1，可以把它关掉。

### 7.2 phase/complex loss

`phase3.loss.use_phase`

当前用 `FlowVNPhaseLoss`，位置：

```text
scripts/mri_data/data_utils.py: FlowVNPhaseLoss
```

支持 method：

- `flowvn_complex_l1` / `complex_l1`
  - 对 real/imag 两通道直接做 L1 residual。
  - 这是 complex L1，不反归一化。
  - 同时约束 magnitude 和 phase。
- `flowvn_unit_complex_l1` / `unit_complex_l1` / `phase_l1`
  - 先 normalize 到 unit complex，再做 L1。
  - 更偏 phase-only。
- `mra_masked_complex_l1`
- `mra_masked_phase_l1`

用户后续明确要求过：

- 不要反归一化后算 FlowVN loss，否则 loss 会变得非常大。
- 直接在 normalized complex real/imag 上算 complex L1。

当前 train loop 中：

```python
phase_output = output_complex_norm.float()
phase_target = target_complex_norm.float()
```

如果有 sensitivity map，会先 coil combine/reduce 再算 phase/complex loss。

### 7.3 vascular loss

`phase3.loss.use_vascular`

当前逻辑：

- 需要 `phase3.enable_vaa=true`，因为 vascular loss 要使用 `mra_prior`。
- `mra_prior` 当作 soft/binary mask。
- 当前推荐 method：

```json
"vascular": {
  "method": "mra_masked_phase_l1",
  "weight": 1.0,
  "normalize_by_mask": true
}
```

用户倾向：

- vascular loss 主要关注血管区域的 phase。
- 如果要全 complex，可通过 config method 改为 `mra_masked_complex_l1`。

### 7.4 不要随意加入 xy/3D SSIM

因为当前训练 sample 不是完整 3D volume，而是一个 raw-x slice 的 temporal window。

因此：

- `zy` SSIM 是可靠的 native 2D plane loss。
- `xy/xz` 这类 orthogonal plane loss 需要跨 raw-x slices 的 case-level context。
- 之前讨论过 batch/window 近似，但用户后续要求保留 zy，不要引入有问题的 xy loss。

## 8. Freeze / checkpoint compatibility

### 8.1 Freeze

`apply_phase3_freeze()` 位置：

```text
scripts/train_utils.py
```

行为：

- `enable_vaa=false`：直接 return，不改任何参数的 `requires_grad`。
- `enable_vaa=true`：
  - `freeze.backbone=true`：冻结非 VAA 参数。
  - `freeze.vaa=true`：冻结 VAA 参数。
  - 支持：
    - backbone freeze + VAA train。
    - backbone train + VAA freeze。
    - joint train。
    - all freeze。

### 8.2 Checkpoint

原则：

- 不改变原 backbone 参数 shape。
- VAA 是新参数，旧 checkpoint 没有这些 key。
- 加载旧权重时应允许 missing VAA keys，保持 checkpoint compatibility。
- 如果加载无 VAA checkpoint：
  - `enable_vaa=false` 最安全。
  - `enable_vaa=true` 也应能 strict=False/兼容加载，VAA 参数从初始化开始。
- 如果加载训练过 VAA 的 checkpoint：
  - config 必须 `enable_vaa=true`，否则 VAA 参数不会被模型实例化/使用。

## 9. Inference / output 重要约定

### 9.1 不要保存 abs-only

之前问题：

- inference 最后保存了 `abs/magnitude`，phase 丢失。
- 官方指标和 flow/velocity 信息需要 phase。

要求：

- 保存模型原始 complex reconstruction 的 real/imag 双通道。
- `.mat` 中推荐 key：`img4ranking`。
- shape 通常：

```text
(PE,SPE,FE,Nt,2) = (y,z,x,t,real/imag)
```

`scripts/run_4dflow_inference.py` 默认在保存前使用 sensitivity maps 做 coil combine，因此新输出通常可以直接 export。
如果使用 `--preserve-multicoil-output`，或处理旧 inference 结果，shape 可能是：

```text
(PE,SPE,FE,Nt,Nc,2)
```

此时要先用 `scripts/tools/fix_4dflow_recon_coil_dim.py`。

### 9.2 coil-combined -> official submission

官方需要：

```text
(Nv,Nt,SPE,PE,FE) = (enc,t,z,y,x)
```

`scripts/tools/export_4dflow_submission.py` 会：

1. 读取 coil-combined per-enc `.mat`。
2. `yzxt -> tzyx`。
3. stack enc0..enc3。
4. 乘 `segmask`。
5. 保存 official sparse `.npz`：

```text
Task*/ValidationSet/Anatomy/Center/Scanner/Patient/img_ktGaussianR.npz
```

## 10. 常用命令模板

### 10.1 单卡训练 GPU2

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline python scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow_haosen.json
```

### 10.2 多卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/nv_raw2insights_mri_base_4dflow_pg.json
```

如果遇到 NCCL timeout，并且同代码别人机器没问题，优先考虑本机 GPU/driver/NCCL/IO/进程状态，而不是直接改代码。

### 10.3 修复 legacy coil dim：R1R2/Aorta 示例

```bash
conda run -n 4dflow python scripts/tools/fix_4dflow_recon_coil_dim.py \
  --final-root /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/R1R2/final \
  --fixed-root /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/R1R2/fixed \
  --data-root /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskR1R2/ValidationSet/Aorta \
  --overwrite
```

### 10.4 修复 legacy coil dim：S2 多 anatomy

```bash
conda run -n 4dflow python scripts/tools/fix_4dflow_recon_coil_dim.py \
  --final-root /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/S2/final \
  --fixed-root /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/S2/fixed \
  --data-root /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskS2/ValidationSet \
  --overwrite
```

### 10.5 export official submission：S2 多 anatomy

```bash
BASE=/SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609

for ANAT in Carotid Cerebrovascular PortalVein RenalArtery; do
  conda run -n 4dflow python scripts/tools/export_4dflow_submission.py \
    --recon-root ${BASE}/S2/fixed/${ANAT} \
    --data-root /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskS2/ValidationSet/${ANAT} \
    --out-root ${BASE}/S2/submission \
    --task TaskS2 \
    --split ValidationSet \
    --anatomy ${ANAT} \
    --recon-layout yzxt \
    --encodings 0,1,2,3 \
    --overwrite \
    --skip-incomplete
done
```

### 10.6 evaluation

GT root 必须是 `ChallengeData_GT`，不是 raw `ChallengeData`。

错误示例：

```text
/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData/TaskS1/ValidationSet/Aorta
```

正确示例：

```text
/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT
```

命令：

```bash
conda run -n 4dflow python scripts/tools/evaluate_4dflow_submission.py \
  --submission-root /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/S2/submission \
  --gt-root /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT \
  --eval-code-dir /mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode \
  --out-csv /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/S2/submission/eval_metrics.csv \
  --out-json /SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/S2/submission/eval_summary.json \
  --task TaskS2 \
  --include-complex-diff \
  --skip-errors
```

## 11. 已知输出路径

当前常用推理输出 root：

```text
/SSDHome/share/haosen/4dflow/output/validation0609/raw_complex_infer_val3splits_epoch120_no_vaa_allenc_20260609/
```

里面有：

```text
R1R2/
S1/
S2/
```

每个 task 通常有：

```text
final/       # raw inference output, may still have coil dim
fixed/       # coil-combined output
submission/  # official sparse npz
jsons/       # generated inference jsons
```

最差 case 挑选结果：

```text
outputs/4dflow/worst_cases_by_ssim_validation0609_no_vaa_epoch120.csv
```

## 12. 常见错误和判断

### 12.1 `ModuleNotFoundError: h5py/numpy`

当前 shell 不是 `4dflow` conda 环境。用：

```bash
conda run -n 4dflow python ...
```

或先 activate 环境。

### 12.2 `eval_metrics.csv` 全是 missing GT

大概率 `--gt-root` 指错了。评测需要：

```text
ChallengeData_GT
```

不是：

```text
ChallengeData
```

### 12.3 inference 后还是多 coil

新 wrapper 默认保存 coil-combined complex image。只有旧结果或显式使用
`--preserve-multicoil-output` 时才会保存 coil-wise image，此时需要用：

```text
scripts/tools/fix_4dflow_recon_coil_dim.py
```

做 SENSE coil combine。

### 12.4 batch size 8 时日志出现 `10/8`、`12/8`

这不是一定表示 batch size 被改坏。训练 loop 内部按 case 的 slice/time window 和 microbatch 分块打印，且不同 case 的 raw-x slice 数可能不同。不要仅凭这个输出改 sampling 逻辑；先确认具体 `mini_dataloader/windowed_input` 行为。

### 12.5 DDP unused parameter

如果 `enable_vaa=false` 但某些 VAA 参数被实例化，或者 freeze/forward/loss 使用不一致，DDP 可能报 unused parameter。当前设计要求：

- VAA off 时不要实例化/call VAA。
- VAA loss on 时必须有 MRA prior。
- 如果确实存在条件分支 unused parameter，可考虑 DDP `find_unused_parameters=True`，但不要作为第一选择。

### 12.6 NCCL timeout

如果同代码别人机器可跑，优先排查：

- 某块 GPU 异常。
- 其他用户进程占用。
- NCCL/P2P/IB 问题。
- DataLoader worker/IO 卡住。
- NFS/NAS 短时卡顿。

调试建议：

```bash
CUDA_VISIBLE_DEVICES=2 ...   # 单卡先跑
```

必要时：

```text
num_workers=0
```

### 12.7 mask 类型识别

raw 文件名类似：

```text
kdata_ktGaussian50.mat
```

config 里可能有：

```json
"train_mask_types_for_def_model": ["uniform", "kt_gaussian", "kt_radial"]
```

`ktGaussian` 文件名和 `kt_gaussian` label 是代码里映射/规范化后的概念，不要简单按字符串相等判断错误。

## 13. 当前工作区状态提示

修改前先检查 `git status --short`，不要随意覆盖尚未提交的训练或 inference 改动。

## 14. 后续修改原则

1. 不要破坏 raw `x` as slice、model plane `zy` 的适配。
2. 不要把 inference 输出改回 `abs/magnitude` only。
3. VAA/MRA/loss/freeze 必须 config-controlled。
4. `enable_vaa=false` 必须 hard bypass，不能有额外推理开销。
5. 旧 checkpoint 必须兼容。
6. 修改 loss 前先确认当前用户想做的 ablation：
   - only complex L1。
   - zy SSIM + complex L1。
   - vascular phase-only。
   - VAA off baseline。
7. official submission 之前必须确认：
   - per-enc fixed `.mat` shape 是 `(y,z,x,t,2)`。
   - export 后 sparse `.npz` shape 是 `(enc,t,z,y,x)`。
   - `segmask` 已乘上。

## 15. 快速理解一句话

这个 fork 的核心不是普通 2D MRI：它把 4D Flow raw `x/FE` 当成 slice，模型在 `zy` 平面上用 temporal window 重建 complex real/imag；Phase3 在 Restormer bottleneck/intermediate 通过 MRA vessel prior 做 residual VAA，所有行为由 `phase3` config 控制，输出最终要转换成官方 `(enc,t,z,y,x)` sparse complex `.npz` 并用 `ChallengeData_GT` 评测。
