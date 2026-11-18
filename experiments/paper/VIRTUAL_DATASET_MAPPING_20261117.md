# NAS 只读虚拟数据路径：2026-11-17

## 用户授权与边界

用户提供 `file_mapping(3).csv`，授权使用它构建虚拟路径。所有 Source/Destination 数据仍严格只读。
本轮仅修改独立实验目录，不改变原模型、loss、数据划分或原始 CSV；没有挂载操作，没有创建数据软链接。
工作分支仍为 `pengfeig/4dflow-finetune`；没有 commit/push。

## 映射核对

- 用户映射和最终清单各 6265 行，其中 6264 行数据 + 1 行共享 README 元数据。
- 6264 个数据 target_file 全部唯一匹配；无缺失、多余、重复 target/source；记录的 size_bytes/shape 一致。
- 保留原 694 Train / 157 ordinary_test / 193 focused_test；README 不作为病例。
- mapping SHA256：`b8302cbf1adeb398121afbc4650a70e093e3896b6f81a706b9910e8ff9291ad1`。
- 原 manifest SHA256：`8d90b6cb36275633e8d9f336949d6275b2cd50e8a41d7bd90c7ef90602c9bf75`。
- 只对一个真实 Train 病例检查了六文件大小、MAT header 和实际加载，不声称所有 NAS 文件已物理检查。
- 未读取真实 Test 样本内容、未运行 Test 指标或 checkpoint 选择。

## 虚拟路径是什么

它是读取时的路径解析，不是重建一套文件系统目录。

```text
原 destination_path / target_file / case_id / dataset_group
  -> 显式 source mapping（按 target_file join，不按 shape 猜测）
  -> NAS source_file，以只读方式打开
```

例如 `Center012_P000203_Aorta` 的逻辑目录仍是
`/media/fudan22/4DFlow/Center012_P000203_Aorta`，实际 k-space 读取：
`/mnt/nas/nas3/openData/rawdata/4dFlow/Aorta/Center012/Philips_30T_Ingenia/vn2/kdata_full.mat`。

`case.files` 保存物理路径，`logical_files` 保存原路径，`rows` 和逻辑 directory 不改。
写保护覆盖原目的地、NAS 实际来源和文件软链接的真实目标；保护根会去重，避免几千条重复校验。
原 `read_manifest()` 不传 mapping 时保持原解析行为。

## 新增与修改

- 新增 `common/path_mapping.py`：严格解析、验证、关联 mapping，保留逻辑身份；同时保护物理和逻辑路径。
- 新增 `build_virtual_dataset.py`：保存 mapping 字节一致副本、虚拟索引、11 个独立运行配置和报告。
- `common/manifest.py`：可选 `source_mapping` / `--source-mapping`，映射样本检查实际文件大小。
- `common/safety.py`：加入 NAS 数据根保护；去除被外层根覆盖的重复保护路径。
- `common/case_data.py`：选择病例可解析映射；实际读取再核对文件大小、k-space shape；cache 仅写实验输出。
- `common/execution.py`、`comparisons/solvers.py`：训练/传统方法读取 `paper.source_mapping`；默认原行为不变。
- `infer.py`、`evaluate.py`：读取 `pipeline.source_mapping`，支持 CLI override，记录映射 SHA。
- `evaluate.py`：兼容本病例 complex64 存储但虚部严格为 0 的 corrmap；若虚部非零则报错，不静默丢弃。
  原 `evaluate_arrays`、指标公式和原训练/模型代码均未改。
- 新增 `test_path_mapping.py`，补充 `test_full_case.py`；更新 README、AGENTS 与全病例记录。

## 配置位置

仅生成独立输出，不覆盖原 `configs/paper`：

```text
outputs/paper/virtual_dataset/
  inputs/source_mapping.csv
  cases_all_labels.json
  train_index.json
  checked_train_cases.json
  mapping_report.json
  pipeline.json
  configs/*.json
  cache/
  full_case/
  runs/
  validation/
```

训练/传统方法：`paper.source_mapping`。
完整推理/评测：`pipeline.source_mapping`，或 CLI `--source-mapping`。
生成的 11 个 config 仅调整 `paper.source_mapping/train_index/output_dir`，其余配置与原文件一致。
原训练 config、原权重目录、原清单和原始 dataset_group 均不改变。

## 已执行的真实 Train 验证

病例：`Center012_P000203_Aorta`，原始 `[4,25,5,19,96,88]`。
enc0 reference，后续方向来自该病例 metadata：`FH;AP;RL`，未套用旧固定顺序。

1. 六文件大小/header 检查通过，DataLoader 读取真实 k-space/coilmap/GT/segmask 成功。
2. 训练输入 `[1,4,5,5,3,19,96]`，GT `[1,4,5,3,19,96]`，均保留 complex 且 finite。
3. 完整模型使用本地 smoke checkpoint、R10、S3/T5，显式 tile 空间窗口，逐中心时间滑动。
   750 次窗口，全部 25×88=2200 个 time-slice 位置覆盖。
4. FlowMoDL 使用原生 S64/T6 和本地 smoke checkpoint；CG-SENSE 使用原配置，均完成完整病例。
5. 三者结果均是 complex64 `[4,25,19,96,88]`，保留编码、原 x/z 顺序和复数信息；覆盖率 min/max=1。
6. 同一固定完整时间 kt-Gaussian mask；原评测核心成功输出三行 finite 指标及分组汇总。
7. 文件大小/mtime 前后检查、原 CSV SHA、原 config 指纹检查记录在机器报告中。
8. 单元/接口测试共 33 passed；覆盖映射缺失/重复/错误、跨 split、逻辑身份保持、软链接写保护和 corrmap 格式。

这些是流程验收，不是训练效果对比；神经模型只加载先前 1–2 step 的 smoke 权重。
当前 CG-SENSE 的 30 次迭代预算中，8800/8800 个二维子问题未达到 tolerance；
这被记录为迭代预算耗尽，而不是流程失败。没有擅自调整迭代数或将流程通过当作收敛证明。
本轮没有在 NAS 上重新训练或更新任何模型权重，没有跑完所有真实病例或所有方法的完整重建。

## 验证产物

- `outputs/paper/virtual_dataset/mapping_report.json`：关联与头信息检查。
- `outputs/paper/virtual_dataset/validation/loader.json`：真实训练窗口。
- `outputs/paper/virtual_dataset/validation/metrics.csv`、`metrics.json`：原指标核心结果。
- `outputs/paper/virtual_dataset/validation/report.json`：完成状态、指纹、覆盖率和限制。
- `outputs/paper/virtual_dataset/full_case/<method>/train/Center012_P000203_Aorta/R10/`：完整复数结果。

复现命令见 README 的“硬盘离线时：NAS 只读虚拟路径”。
