# 4D Flow 最终数据集：训练、全病例推理与评测

本目录按用户提供的实验要求与 2026-11-15 固定版执行文档实现。
私有任务 prompt / DOCX 仅保留在本地，不随代码分支上传，也不影响现成配置的运行。
基准代码是 `pengfeig/4dflow-finetune` 的 `9504c29e50237d96eee89f3cf47693c64482bd2f`。
2026-11-15 已完成真实 Train 样本的 1–3 步预检，未执行完整训练、测试集推理、测试指标或 checkpoint 选择。
2026-11-17 按追加要求新增全病例推理/拼接与原评测接口，入口和命令见下文。
新增流程的验证状态单独记录在 `FULL_CASE_PIPELINE_20261117.md`，不能把旧窗口预检等同于全病例验收。
2026-11-18 训练配置增加最终版在线采样；旧预检报告及其 registry 中的配置指纹是历史记录。

## 发布分支与首次克隆

集群版本发布到 `git@github.com:guopengf/NV-Raw2insights-MRI.git` 的
`haosen/baselin_ablation` 分支（按本次指定拼写），原 `haosen/baseline_ablation` 保留。
九个训练配置的交付命令、I/O 优化与排队顺序见 [CLUSTER_TRAINING.md](CLUSTER_TRAINING.md)。

```bash
git clone --branch haosen/baselin_ablation \
  git@github.com:guopengf/NV-Raw2insights-MRI.git
cd NV-Raw2insights-MRI
conda activate 4dflow
python -m experiments.paper.bootstrap --fetch-authors --install-deps
export PAPER_PYTHON="$PWD/experiments/paper/.venv/bin/python"
```

FlowVN 源码沿用本仓库已有版本；FlowMoDL 与 CUT-4DFlow 在 `vendor/` 下作为
**固定提交的 Git 子模块**一并注册，对应 `.gitmodules` 和 `vendor_lock.json`。
GitHub 中这些目录链接到作者的准确提交，不是遗漏模型代码。
推荐上面的 bootstrap：按锁定版本准备源码，CUT 仅取 src/scripts/configs，避免下载作者权重/数据。
它也支持首次 clone 后留下的空子模块目录；已有正确版本不重复 fetch，不覆盖本地源码改动。
CG-SENSE、CS-TV 适配实现就在 `comparisons/solvers.py`，额外依赖由 requirements-extra.txt 安装。

本次发布包含 A0–A4、完整模型和五种基线配置、训练入口、全病例推理/评测、启动脚本和测试。
清单/tree 是匿名逻辑文件索引，不含 MRI 数组；原始文件、source mapping、环境、缓存、
实验输出和新训练权重不提交。NAS 路径映射需按后文用自己的本地 mapping 重新生成。

## 目录隔离

```text
configs/paper/                  A0–A4、完整模型配置
configs/paper/baselines/        五种对比方法配置
configs/paper/evaluation.json  共同采样、全病例输出与指标设置
experiments/paper/common/       只读清单、共同测量算子、训练窗口与输出保护
experiments/paper/ours/         本模型的独立训练入口，只导入现有模型与 loss
experiments/paper/comparisons/  对比方法的适配入口
experiments/paper/vendor/       固定版本的作者代码 Git 子模块，不改成我们的 backbone
experiments/paper/inputs/       匿名逻辑清单/tree；私有任务书仅在本地保留
experiments/paper/infer.py      全病例、全时间、全编码复数重建
experiments/paper/evaluate.py   复用原官方评测核心的批量指标入口
experiments/paper/scripts/      每个实验的独立启动脚本
outputs/paper/                 索引、缓存、日志、checkpoint、预检和导出文件
```

没有修改原有 `scripts/train.py`、模型、loss、reader 或 inference。
本目录配置使用下文的 `python -m experiments.paper...` 入口，不能直接交给旧的
`scripts/train.py`，因为最终数据集没有旧训练入口要求的预先欠采样文件/目录层级。

## 数据与只读保护

- 以原始 CSV 的 `destination_path`、`target_file`、`disk`、`dataset_group` 为准，不移动或重分数据。
- 清单包含 694 个 `train`、157 个 `ordinary_test`、193 个 `focused_test` 病例。
- 最后一行是两个数据盘 README 的元数据映射，不是病例，原样保留在清单副本和报告中。
- 没有 `val`，因此验证、模型选择均关闭；不拿任何 Test 病例补成验证集。
- `/media/Elements/4DFlow/`、`/media/fudan22/4DFlow/`、原始 Source/Destination 路径严格只读。
- `OutputGuard` 在写入前解析 realpath，保护数据根、清单中所有病例目录及输入清单目录，拒绝通过符号链接写回数据。
- 新配置从全采 k-space 在内存生成欠采样输入，直接调用 cy136 最终 mask suite。没有在数据目录中创建欠采样文件、缓存或链接。

原始 k-space 是 `[encoding,time,coil,kz,ky,kx]`；只沿 raw kx 先做 IFFT，
之后以 raw x 为 slice/slab，欠采样平面仍是 z-y。本包的共同张量约定：

```text
测量: [B,E,T,C,X_window,Z,Y] complex64 (kx 已 IFFT，z/y 仍在 k-space)
coil: [B,C,X_window,Z,Y] complex64
mask: [B,T,Z,Y]，所有编码和 raw x 共用同一采样模式
重建: [B,E,T,X_window,Z,Y] complex64
```

每次逐 frame 读取完整 kx 并 IFFT，再取 raw-x 窗口，避免把全部时间/编码的巨大体数据同时读入内存。
coilmap 做 RSS 归一化，所有方法使用相同的外部 sensitivity maps 和测量。
欠采样测量及 GT 共用从已采集 k-space 算出的 RMS scale，保留 real/imag，不对 complex L1 反归一化。
`params.csv` 逐病例读取 VENC、venc_order、spatial_order。原生编码索引不变：enc0 为 reference，
enc1–3 的方向按该病例 metadata；不能套用旧 Aorta 固定的 `RL;AP;HF`。
若 params 的 matrix_size 与 tensor 不同，以实际 tensor 为准，并在运行报告中记录差异。

训练用 `segmask.mat` 的静态 `[Z,Y,X]` ROI，转成对应 `[X_window,Z,Y]`。
`segmask_new.mat` 的 `[T,Z,Y,X]` 也完成文件头检查，但本轮不替换原完整配置的静态 ROI 方案。
`corrmap.npz` 保持只读，当前训练不进行 BGC/evaluation。

## A0–A4

所有消融都是 6 cascades、3 slices × 5 frames、seed=2026、独立随机初始化，VAA 均关闭。
主干、DC、外部 coilmap 沿用现有代码。

| 配置 | Backbone | Mixer | Joint-VENC | Loss |
|---|---|---|---|---|
| A0 | 2D，S×T channel stack | 关闭 | 编码独立 | 基础 complex L1 |
| A1 | Spatial Conv3D，X 为显式空间维 | 关闭 | 编码独立 | 同 A0 |
| A2 | 同 A1 | xyz/xyt/yzt/xzt | 编码独立 | 同 A0 |
| A3 | 同 A2 | 开启 | channel 四编码融合 | 同 A0 |
| A4 | 同 A3 | 开启 | 开启 | 基础 complex L1 + official_aligned ROI/flow loss |

`ours_full_final_dataset.json` 与 A4 使用同一完整模型结构与 loss，输出目录独立。
原生模型输入仍为 `[B*E,S,T,C,Z,Y,2]`；A3/A4 在现有主干内部再融合 E，
没有将 VENC 当成时间。训练只监督中心 temporal frame，保留 S=3 的全部 slice，
其余 temporal frames 提供上下文。最后一个 cascade 的原生时间聚合逻辑没有改变。

A0–A3 基础 loss 是 `FlowVNPhaseLoss(method="flowvn_complex_l1")`，即 real/imag 差的 L1，
并不是单独的 angular phase loss。A4 在其上复用现有 `JointEncodingLoss(profile="official_aligned")`：
ROI complex、cross-VENC circular phase、velocity、RelErr、angular 各权重 0.1；
非 complex flow 项按原配置 epoch 5–20 ramp。没有重新定义这些 loss，也未开启 SSIM。

正式训练默认 960 epochs、AdamW、lr=5e-5、weight_decay=0.001、5 epoch warmup、cosine 到 1e-5。
当前统一配置禁用数据增强；每病例每 epoch 8 个随机窗口，每窗口四编码，共 4 encoding images，
不等同于旧数据集每个加速倍数一个 JSON 的 epoch 预算。
没有自动下载 pretrained 权重或从旧实验目录隐式恢复。正式恢复用 `paper.checkpoint` 指定本入口保存的 `latest.pt`。

## 对比方法

- CG-SENSE：SciPy complex CG，每个编码、时间和 raw-x slice 单独求解 SENSE 正规方程。
  当前 30 iterations、rtol=1e-6、L2=1e-4；用完预算未收敛会明确报告，不把流程 PASS 当成重建质量证明。
- CS-TV：SigPy `TotalVariationRecon`，外部多线圈 SENSE + 三空间方向 TV，100 iterations，TV weight=0.002。
  对 hybrid x 做一次酉 FFT 后交给原生 3D SENSE/TV 算子，与共同测量等价。
- FlowVN：直接导入仓库已有 PyTorch `FlowVN`，10 stages、24 filters、kernel 7；编码独立，
  使用作者非 exp supervised complex L1 和 Adam/cosine。
  源码的 `D_size` 实际控制 depth padding，设为 7 与 kernel 7 一致，输入窗口仍是 3 slices。
- FlowMoDL：固定作者 SHA，使用原 Hydra model/loss、CG DC、composite loss 和 `train.py::train_step`，
  并使用作者 inference helper。正式和 smoke 均使用 64 readout × 6 cardiac frames，
  完整 8 cascades、30 CG iterations、64-channel denoiser 不缩减。
- CUT-4DFlow：固定作者 Final 配置（15 stages、d=160、10 heads、unshared MLP、EMA），
  调用原 `train_cmrx.training_step`、原 k-space velocity-difference loss 和 EMA，逐 raw-x slice 重建。
  正式窗口为 1 slice × 5 frames。
  按作者原设置累计 5 个窗口的梯度，再执行一次 optimizer step 和 EMA 更新；
  `paper.gradient_accumulation=5`，因此 1-step smoke 实际会读取 5 个 Train 窗口。

作者源码和原 `train.sh/recon.sh` 保存在 vendor；固定版本见 `vendor_lock.json`。
本包不直接运行作者 CLI 的自动 train/val 划分或 dataset-side 预处理。
适配入口只替换病例来源和 tensor 接线，使用原网络/训练步骤/损失，并关闭无独立清单的 validation。
三个深度基线均使用作者网络，不使用本模型的 Restormer 代替。

## 在 GPU 服务器上准备

先激活已有 `4dflow`/本项目环境，再在仓库根目录执行：

```bash
python -m experiments.paper.bootstrap --fetch-authors --install-deps
export PAPER_PYTHON="$PWD/experiments/paper/.venv/bin/python"

"$PAPER_PYTHON" -m experiments.paper.common.manifest \
  --manifest experiments/paper/inputs/final_dataset_manifest.csv \
  --tree experiments/paper/inputs/final_dataset_tree.txt \
  --headers

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.smoke --device cuda:0
```

复制本项目和 `configs/paper/`、`experiments/paper/` 到服务器；不要复制本机 `.venv`，在目标环境重建。
两块数据盘仍需能按清单路径只读访问，不要创建新数据划分或搬动数据。
若已把两个作者仓库整体复制过去，可省略 `--fetch-authors`，但应保持固定 SHA。
额外依赖只装进本目录 `.venv`，不升级原 conda 环境。基础项目依赖见根目录 `requirements.txt`。

正式配置与逻辑清单/tree 已随分支提供，无须在服务器重新生成。
只有需要从用户完整的本地任务书输入目录重新生成配置时，再显式执行：

```bash
python -m experiments.paper.generate_configs --input-root /path/to/后续实验
```

这只复制任务书/清单副本和生成实验配置，不复制、修改或重分 MRI 数据。

## 正式训练命令（本轮没有执行）

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash experiments/paper/scripts/ours_full_final_dataset.sh

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash experiments/paper/scripts/ablation_A0.sh

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  bash experiments/paper/scripts/flowmodl.sh
```

每组有独立启动脚本，全部命令见 `run_commands.sh`，不要无意中一次执行所有完整训练。
默认单卡；当前独立实验入口没有实现多卡 DDP，不要加 torchrun，否则会重复启动多个实验进程写同一路径。
上述传统方法旧启动脚本仍用于单 Train 窗口预检；完整重建请使用下文统一 `infer` 入口。

正式结果保存为 `outputs/paper/<experiment>/train/latest.pt`；最小预检为
`outputs/paper/<experiment>/smoke/latest.pt`，两者不会互相恢复。
模型、optimizer、scheduler 和 CUT EMA 一并保存；验证集未指定时只保存 latest，不声称 best。

## 预检覆盖与边界

- 选择体积较小的真实 Train 病例，使用静态血管 mask 非空的 raw-x 窗口。
- A0/A1 和三个深度对比方法运行 1 step；带 Mixer 的 A2–A4/完整模型运行 2 steps，
  确认 zero-init scale 移动后 regularizer 也获得非零梯度。
- A4/完整模型用 `--probe-flow-loss` 临时令 ramp=1，FlowMoDL 临时使用 curriculum epoch=32，
  仅用于覆盖真实血管的 phase/flow 梯度。正式配置中的 ramp 计划未修改。
- 检查实际 DataLoader、真实输入、forward、finite loss、backward、非零梯度、optimizer 更新、
  `eval/no_grad` 推理、complex 输出、checkpoint 写入及回读 shape/optimizer state。
- 旧 smoke 输出为训练窗口，并非完整病例/官方 submission；全病例接口验证和质量验收须另行区分。
- 大病例、完整 960 epochs 的资源/收敛、DDP、多服务器吞吐尚未验证。
- 所有测试用 Train，不拿 ordinary/focused test 做调试、模型选择或指标。

逐项状态、实际 loss/gradient/shape/GPU 峰值见 `preflight_report.md` 和各 `smoke/report.json`。
复数导出接口示例：

```bash
"$PAPER_PYTHON" -m experiments.paper.export_train_window --experiment ours_full_final_dataset
```

生成 MAT/NPZ 于 `outputs/paper/test_output/`，并验证 complex roundtrip，明确标注为 Train window。
我们的模型只导出中心 temporal frame：原生 forward 的 T 轴在最后 cascade 是中心估计的重复，
不能把这 T 份复制误当成 T 个独立时间重建。新增全病例入口逐中心帧滑窗实现完整时间序列。
基线保留其窗口内实际重建的时间帧。导出保留归一化域的 complex 数值，不代表扫描仪原始幅值单位。
该文件不能直接当作完整病例的 challenge submission 提交。

## 在线欠采样训练

11 份实验配置默认开启 `paper.online_undersampling.enabled`。
`suite_path` 默认指向随仓库提交的最终 `code/mask_suite.py`，源码和两个 profile
与 cy136 最终版逐字节一致。训练直接调用 `generate()`，不挂载 cy136、不运行 CLI，
也不创建中间 mask 文件。已生成的旧配置不会随 Git 更新：请重新运行
`build_virtual_dataset`，并省略 `--suite-path`，以使用仓库内默认版本。

| 配置项 | 用途 |
|---|---|
| `enabled` | `true` 使用最终套件；`false` 保留原实验入口的窗口级 kt-Gaussian 行为 |
| `methods` | 从 `gaussian` / `pr4d` / `pseudo` / `uih` 中等概率选择；只留一个可做方法对照 |
| `rate_mode` | `continuous` 随机实数倍率；`integer` 在范围内均匀抽整数，默认范围对应 41 档 |
| `rate_range` | 倍率范围，默认 `[10, 50]`；可设 `[27.5, 27.5]` 做固定实数倍率对照 |
| `acs_zy` / `pseudo_profile` | 动态三种方法默认 `3x3` ACS；pseudo 默认拟合几何。UIH 由套件自行缩放分层 |
| `cache_cases` | 单 worker 内存缓存的病例数，默认 1024，二值图案按位存储；跨 worker 或淘汰后仍能按固定 seed 重建同一图案 |

每个病例每个 epoch 只选一组方法、倍率和 seed；由 `paper.seed`、epoch 和 case ID
稳定确定，同一 epoch 不受窗口大小、读取顺序或 worker 影响；下个 epoch 重新抽样。
Gaussian、PR4D e0、pseudo 原生为 `[T,Z,Y]` 并随 T 变化，窗口按绝对时间索引取对应帧；
UIH 原生为 `[Z,Y]`，只在读取窗口时沿 T 广播。所有方法沿四编码、线圈和完整 kx 共用采样。
读取全采 `kdata_full.mat`，在 kx IFFT 后应用 mask（与 IFFT 前应用等价，因为 mask 不依赖 kx），
GT 仍取自未掩蔽的全采数据。每帧点数为 `floor(Nz*Ny/R)`，请求 R 与实际网格 R 分开记。
训练输出的 `sampling/epoch_*.json` 记录每例方法、倍率、实际点数、seed 和套件源码/配置指纹。

FlowVN 和 FlowMoDL 的连续条件输入接收请求 R；本模型的离散倍率标签和 CUT 的原生条件表
选择最近档，本模型的 FlowVN mixer 仍使用实数 R。本模型四种 mask 共用既有 mask-conditioning
槽位以保留 checkpoint 形状；传给 DC 的始终是真实二值 mask，不是 Gaussian 替代图案。

本次接入的是独立实验入口的训练和传统方法 Train 窗口预检；旧 `scripts/train.py` 预生成数据入口未迁移。
正式整病例推理/评测的固定采样协议仍保持原样：`FullCase.fixed_mask` 使用旧 kt-Gaussian，
按 case/R/seed 固定完整 Nt 并共享，不会自动改用训练随机样式。
两种阶段都不会在原数据盘写入缓存或 mask。

## 全病例推理与拼接

激活本目录环境后，在仓库根目录运行。以下仅选一个 Train 病例验接口，正式训练权重需先训练得到；
不要把只有 1–2 步的 smoke 权重当作最终结果权重。

```bash
export PAPER_PYTHON="$PWD/experiments/paper/.venv/bin/python"

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.infer \
  --config configs/paper/ours_full_final_dataset.json \
  --checkpoint outputs/paper/ours_full_final_dataset/train/latest.pt \
  --groups train --case-ids Center012_P000203_Aorta \
  --accelerations 10 --device cuda:0

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.infer \
  --config configs/paper/baselines/flowmodl.json \
  --checkpoint outputs/paper/flowmodl/train/latest.pt \
  --groups train --case-ids Center012_P000203_Aorta \
  --accelerations 10 --device cuda:0

"$PAPER_PYTHON" -m experiments.paper.infer \
  --config configs/paper/baselines/cg_sense.json \
  --groups train --case-ids Center012_P000203_Aorta --accelerations 10 --device cpu
```

其他方法只替换 config 和该方法的 checkpoint；每个方法在单独进程运行，禁止 torchrun。
去掉 `--case-ids` 即遍历指定 group 的所有病例，去掉 `--accelerations` 默认 R10/20/30/40/50。
后续独立测试可显式指定 `--groups ordinary_test focused_test`，本次开发未运行这些分组。
未指定 checkpoint 时只查对应实验 `train/latest.pt`，不存在就报错，不自动下载或使用随机权重。
支持本实验入口保存的 `model`、本项目旧 `net_state_dict` 以及同构纯 state_dict；推理严格核对键和 shape。
CUT 使用保存的 EMA。作者其他训练平台的 checkpoint 需先明确转换成同构 state_dict，不能静默跳过权重。

输出：

```text
outputs/paper/full_case/<experiment>/<原dataset_group>/<case_id>/R10/
  reconstruction.npy    complex64 [4,Nt,Nz,Ny,Nx]，原空间顺序
  reconstruction.json   mask/权重/config/source 指纹、VENC 顺序、覆盖率、窗口策略
```

只交换 tensor 轴，不做空间翻转，不丢编码，不取 abs；恢复采样 k-space 的 RMS scale，
幅值域为统一 unit-RSS 外部 coilmap 合成，不是训练归一化域。
缓存按病例逐 frame 构建、结果用磁盘 memmap 拼接，大病例需要足够实验盘空间。
一个病例 cache 可被多个方法/R 复用；保留外部 coilmap 和原始 phase。

默认策略：本模型中心 slice/中心 time 滑窗，确保每个输出 time 是真实中心估计；
FlowVN/FlowMoDL/CUT 以原生窗口 tile，尾窗利用真实邻居，只写未覆盖位置。
时间 wrap、x 边界 replicate，与本独立训练入口一致。
可用 `--merge center|tile|overlap` 明确覆盖策略；overlap 是复数加权次数平均，不是幅值平均，
实验之间须记录并保持选定策略。CG-SENSE/CS-TV 固定全 x、逐帧求解，CS-TV 保持完整 3D 空间 TV。

完成标记最后发布；中断结果不会被评测。相同权重、mask、config、窗口、输入指纹的完整结果可以跳过，
变化则报错，显式用另一输出目录或 `--overwrite`。锁文件残留时先确认旧进程已停止，不自动强行接管。

## 接入原来的评测方式

```bash
"$PAPER_PYTHON" -m experiments.paper.evaluate \
  --recon-root outputs/paper/full_case \
  --groups train \
  --out-csv outputs/paper/metrics/train_metrics.csv \
  --out-json outputs/paper/metrics/train_summary.json
```

递归处理目录下所有实验、病例和 R，按原 dataset_group / experiment / R 分组汇总。
CSV 每行一个实验×病例×R，包含 SSIM、nRMSE、RelErr、AngErr、ComplexDiffErr。
默认异常立即停止；加 `--skip-errors` 则错误单列记录，失败病例不混入均值。
采样 SHA 不一致直接报错，不能把不同随机采样结果作为同测量对比。

复用原 `scripts/tools/evaluate_4dflow_submission.py::evaluate_arrays` 和指定 EvaluationCode，
不重写指标。GT 由 full k-space 和同一 coilmap 生成；读取现有 segmask/corrmap，绝不在数据目录生成 corrmap。
`configs/paper/evaluation.json` 默认沿用历史 wrapper：血管 mask、bbox、phase radians，
`metrics.use_venc=false`；设 true 则按每病例 VENC 转成速度单位，所有方法必须统一。
这是原评测的血管 ROI 指标，不代表额外的全组织幅值评测。

### 我们已有的 MAT/NPY/NPZ 输出

通过显式 CSV 映射到新最终清单，避免猜测轴或旧目录命名。比如四个编码各一个 `.mat`：

```csv
case_id,R,experiment,encoding,recon_path,layout,key,mask_sha256
Center012_P000203_Aorta,10,legacy_ours,0,/path/enc0.mat,yzxt,img4ranking,
Center012_P000203_Aorta,10,legacy_ours,1,/path/enc1.mat,yzxt,img4ranking,
Center012_P000203_Aorta,10,legacy_ours,2,/path/enc2.mat,yzxt,img4ranking,
Center012_P000203_Aorta,10,legacy_ours,3,/path/enc3.mat,yzxt,img4ranking,
```

`layout` 必须写实际存储顺序：`yzxt` 是 [Ny,Nz,Nx,Nt]，已经回到 [Nx,Ny,Nz,Nt] 则写 `xyzt`。
单文件含四编码可写 `etzyx` 且 encoding 留空。支持原生 complex 或最后一维 real/imag=2；
拒绝 magnitude-only、多 coil 残留、重复 append 维度和窗口冒充全病例，不自动猜方向或修复数据。
旧结果必须已经恢复正确幅值尺度；不能把训练窗口的归一化输出直接作为全病例结果。
若旧结果无 mask SHA 仍可计算，但 CSV/JSON 标记 sampling_unverified，不声称采样公平性已验证。

```bash
"$PAPER_PYTHON" -m experiments.paper.evaluate \
  --prediction-index /path/predictions.csv --groups train \
  --out-csv outputs/paper/metrics/legacy.csv \
  --out-json outputs/paper/metrics/legacy.json
```

## 独立接口验证

```bash
"$PAPER_PYTHON" -m pytest \
  experiments/paper/test_contracts.py experiments/paper/test_full_case.py \
  experiments/paper/test_path_mapping.py experiments/paper/test_bootstrap.py \
  experiments/paper/test_online_sampling.py -q

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.validate_full_case --run --device cuda:0
```

后者生成独立合成 phantom，逐个加载已有 smoke checkpoint，跑完整推理和原指标核心，
报告在 `outputs/paper/full_case_check/report.json`。它不读取真实 Train/Test，
仅验证网络接口、全覆盖与评测接线，数值不能作为 MRI 质量结果。

## 硬盘离线时：NAS 只读虚拟路径

用户提供的 `file_mapping(3).csv` 含 `source_file,target_file,size_bytes,shape`。
新增入口按 `target_file` 与原最终清单严格关联，读取时把逻辑目标路径解析到 NAS，
**不在 `/media` 建目录，不创建数据软链接，不改原 CSV，不复制或搬动原始 MRI 文件**。
推理需要的派生 hybrid/GT 数组仍只缓存在独立实验输出目录。
原 case_id / dataset_group / disk / 逻辑 directory 全部保留；仅 `case.files` 指向物理来源，
原文件位置保存在 `logical_files`，映射表另存 SHA。

```bash
"$PAPER_PYTHON" -m experiments.paper.build_virtual_dataset \
  --mapping "/path/to/file_mapping(3).csv"
```

默认输出 `outputs/paper/virtual_dataset/`：

```text
inputs/source_mapping.csv  用户映射表的字节一致副本
cases_all_labels.json      1044 个病例的逻辑身份与物理读取位置
train_index.json           原 694 个 Train 病例，不重新划分
mapping_report.json        6264 文件关联、CSV 指纹、只读保护和实际检查范围
checked_train_cases.json   默认 1 个小 Train 病例的六文件大小与 MAT 头检查
pipeline.json              inference/evaluate 使用的 NAS 配置
configs/*.json             11 个独立 NAS 训练/传统方法配置
cache/                     数据集之外的 hybrid/GT/采样 cache
full_case/                 全病例复数结果
runs/                      NAS 训练配置的独立训练/smoke 输出
validation/                本次真实 Train 验证记录和指标
```

生成配置只改数据/输出地址和显式指定的 `suite_path`，
模型结构、loss、seed、S/T 和所有训练超参数与原 config 一致，原 config 本身不改。
`--check-train-cases 1|2|3` 控制只读检查的小 Train 病例数；索引完整不等于全部 NAS 文件已物理验收。
映射文件有缺失、多余、重复 target/source 或 size/shape 不一致时直接报错，不尝试按 shape 猜病例。
同一 output 目录已有另一份映射则拒绝覆盖，应指定独立 `--output`。

使用已经生成的映射配置，例如：

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.infer \
  --config configs/paper/ours_full_final_dataset.json \
  --pipeline outputs/paper/virtual_dataset/pipeline.json \
  --checkpoint outputs/paper/ours_full_final_dataset/smoke/latest.pt \
  --groups train --case-ids Center012_P000203_Aorta \
  --accelerations 10 --merge tile --device cuda:0

"$PAPER_PYTHON" -m experiments.paper.evaluate \
  --pipeline outputs/paper/virtual_dataset/pipeline.json \
  --recon-root outputs/paper/virtual_dataset/full_case \
  --groups train --case-ids Center012_P000203_Aorta \
  --out-csv outputs/paper/virtual_dataset/validation/metrics.csv \
  --out-json outputs/paper/virtual_dataset/validation/metrics.json
```

上例特意使用 smoke checkpoint 验流程，不代表正式训练效果；tile 保留原生 S=3 预测，
但本模型仍逐中心 time 滑窗，完整覆盖全部时相。正式实验须显式固定聚合策略和正式 checkpoint。
传统/深度基线只需替换 config、对应 checkpoint（CG/TV 不需要权重）。
也可在 infer/evaluate 上使用 `--source-mapping /path/mapping.csv` 覆盖映射，不改原 pipeline。

未来在 NAS 上训练时使用生成的独立 config，而不是原本指向离线硬盘索引的 config：

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
  "$PAPER_PYTHON" -m experiments.paper.ours.run \
  --config outputs/paper/virtual_dataset/configs/ours_full_final_dataset.json \
  --smoke --steps 1 --device cuda:0
```

去掉 smoke/steps 才是正式训练，本次未执行。原 checkpoint 目录不会被覆盖。
映射构建与真实 Train 验证结果见 `VIRTUAL_DATASET_MAPPING_20261117.md`。
