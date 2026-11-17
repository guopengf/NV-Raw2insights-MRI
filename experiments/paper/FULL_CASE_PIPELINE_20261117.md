# 全病例推理与评测接入记录：2026-11-17

## 本轮范围

按用户截图中的两项补齐功能，不改原模型、loss、训练或数据集目录：

1. A0–A4、完整模型、CG-SENSE、CS-TV、FlowVN、FlowMoDL、CUT-4DFlow 统一全病例推理和拼接。
2. 接入原 `evaluate_4dflow_submission.py::evaluate_arrays`，批量计算指标并输出 CSV/JSON。

工作分支 `pengfeig/4dflow-finetune`；未进行 commit/push。
11 个正式训练 config 和旧 smoke report 不因本轮改写。

## 文件

- `configs/paper/evaluation.json`：共同采样、缓存、输出和评测设置。
- `experiments/paper/infer.py`：全病例 CLI、输出锁、完整性标记、已有结果一致性检查。
- `experiments/paper/predictor.py`：各网络 target-free forward；严格核对权重、CUT EMA。
- `experiments/paper/common/case_data.py`：只读原清单病例、完整 hybrid/GT cache、固定完整时间 mask。
- `experiments/paper/common/windows.py`：center/tile/overlap 调度、复数原尺度拼接与覆盖率检查。
- `experiments/paper/common/prediction_io.py`：新输出自动发现，旧 MAT/NPY/NPZ 显式 CSV 轴/编码映射。
- `experiments/paper/evaluate.py`：调用原指标核心，检查采样一致性、读取已有 corrmap、分组统计。
- `experiments/paper/validate_full_case.py`：独立 synthetic-only 全流程检查。
- `experiments/paper/test_full_case.py`：完整覆盖、phase/scale/axis、旧 MAT、锁/断点和指标接线测试。
- `experiments/paper/README.md`、`AGENTS.md`：更新使用方式和本轮边界。

## 推理不变量

原始 `[E,T,C,kz,ky,kx]`；IFFT raw kx，raw x 是 slice，重建平面 z-y。
固定采样应用于 kz/ky，完整保留 kx；不交换 enc 与 t。
enc0 是 reference，enc1–3 的方向严格来自逐病例 params.csv。

保存 complex64 `[4,Nt,Nz,Ny,Nx]`，不 abs、不翻转空间，不残留 coil/window 维。
把训练入口的采样 RMS scale 乘回，幅值域为 unit-RSS 外部 coilmap 合成。
所有 t/x 必须覆盖，未完成结果不发布 `reconstruction.json`，不进入评测。

本模型最后 cascade 对 T 轴输出中心估计的重复，必须逐中心 t 滑动，不能输出重复项冒充其他心动时相。
默认中心 t/中心 x，基线默认输出原生窗口的真实预测并 tile；可显式选择 overlap 复数次数平均。
时间 wrap、x 边界 replicate，沿用本独立实验训练入口。
CS-TV 始终在完整 x 空间上求解 3D TV，不将它改成独立 slab TV。

## 欠采样方案

不是新造的采样算法：调用现有
`scripts/generate_4dflow_ktgaussian_train.py::generate_ktgaussian_mask_2d`。
方案为可变密度 kt-Gaussian，Gaussian sigma 为 Ny/5、Nz/5，中心默认一个点，
帧内最小间距因子 3，采样点跨时间权重衰减 0.5，目标点数 floor(Nz*Ny/R)。
默认 R10/20/30/40/50，同时记录 actual_R，不声称等同官方 MATLAB 的逐点结果。

- 训练：内存动态生成随机窗口 mask，原训练代码不变。
- 推理：每 case/R/seed 固定一次完整 Nt 序列；所有方法及不同窗口长度共享对应绝对时刻的采样。
- mask 的身份包含病例、R、seed、矩阵、生成代码 SHA；保存 SHA 防止比较不同采样。
- 缓存只写实验输出目录。欠采样 mask 与分割 segmask/VAA mask 是不同对象。

## 评测

使用原 EvaluationCode 和原 wrapper 的 SSIM、nRMSE、RelErr、AngErr、ComplexDiffErr。
默认 blood-vessel mask + bbox，use_venc=false 沿用历史 wrapper；如需真实速度单位，
统一设置 use_venc=true 并使用逐病例 VENC，不能不同方法各用一套。
只读取现有 corrmap，不调用会写回 GT 目录的生成函数。
输出按 experiment / 原 dataset_group / R 汇总，异常默认中止；skip-errors 明确列出失败行。

已有输出通过 CSV 指定 case_id、R、encoding、路径、实际轴序和 key，不猜 x/z；
支持四编码分文件和尾维 real/imag。拒绝 magnitude-only、多 coil、重复 append 维度和非全病例窗口。
缺少 mask SHA 的旧输出可以计算，但标为 sampling_unverified，不能宣称同测量公平性已验证。

## 实际验证

1. 单元测试：16 passed（含原 5 项）；覆盖中心/平铺/重叠、超大 S64/T6 边界、复数幅值恢复、轴序、编码顺序和中断保护。
2. 11 个实验全部加载原结构/本地 smoke checkpoint，完成独立合成病例全推理，无模型缩减。
3. 合成原始 k-space `[4,3,2,16,16,4]`，全部输出 `[4,3,16,16,4]`，全部覆盖 12/12 t-x，未遗漏。
4. 11 个结果使用同一个 mask SHA，原评测核心全部得到 finite 指标，CSV/JSON 成功。
5. 已知 GT 自比：SSIM=1，nRMSE=0，RelErr=0，ComplexDiffErr=0。
   AngErr≈0.00759 度来自原公式 eps，未修改原公式强制归零。
6. 首次真实 Train 全病例运行因 /media 数据盘拔掉而未完成；随后用户提供 source mapping，
   已通过 NAS 只读虚拟路径补跑单 Train 病例的完整模型、FlowMoDL、CG-SENSE 与原指标核心。
   具体覆盖和限制见 `VIRTUAL_DATASET_MAPPING_20261117.md`；不等价于全部真实病例/模型的质量验收。

合成报告：`outputs/paper/full_case_check/report.json`。
合成指标：`outputs/paper/full_case_check/metrics/metrics.csv` 和 `metrics.json`。
这些数字仅用于接口测试，禁止作为论文或实际 MRI 重建效果。
没有运行真实 ordinary_test/focused_test，也没有完整训练或 checkpoint 选择。

## 数据盘拔掉后的验证选择

不改原 CSV、不改变原 split、不修改任何原数据目录。若 NAS 上有这些病例的真实来源副本，
可另建实验目录内的显式文件映射，在 loader 读取时把逻辑路径解析到 NAS 的只读物理路径。
必须先有可信的最终病例 ID ↔ 来源文件对应关系，再检查六文件、shape、metadata、coil/seg/corr 对齐；
不能仅凭 shape 相同断定同一病例，也不能改拿 Test 补 Train。

软链接只可建在实验 scratch 目录中，而且仍须能访问真实目标；它不能替代缺失硬盘上的数据。
最终 CSV 只有 destination_path，没有 source_path；用户随后提供 `file_mapping(3).csv`，
6264 个数据文件与最终清单一一对应。已用独立 resolver 构建虚拟索引，不改 CSV，
未创建软链接或挂载操作。实现与验证见 `VIRTUAL_DATASET_MAPPING_20261117.md`。

## 下一步

- 已用只读 NAS 映射补跑 1 个小 Train 病例的完整 Nt/X；后续扩大真实训练数据与各方法检查范围。
- 检查大病例实验盘空间、吞吐、solver 收敛和真实视觉质量，不能用小 phantom 推断资源或精度。
- 正式训练得到权重后再运行独立 held-out 评测，先固定 VENC 单位、窗口聚合、采样 seed 和评测 ROI。

完整命令与外部结果 CSV 示例见同目录 README。
