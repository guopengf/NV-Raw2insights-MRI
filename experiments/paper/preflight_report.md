# 训练就绪性预检报告

生成时间（UTC）：2026-11-15T14:42:47.155361+00:00

基准分支：`pengfeig/4dflow-finetune`；Git SHA：`9504c29e50237d96eee89f3cf47693c64482bd2f`。
本轮没有运行完整训练、任何 Test 推理/指标或测试集模型选择。

## 数据检查

Train：694；ordinary_test：157；focused_test：193。
已读 Training 文件头：694 个病例；缺失/异常：0。
保留 CSV 全部原始标签和 README 元数据行；无独立 val，验证关闭。
CSV SHA256：`8d90b6cb36275633e8d9f336949d6275b2cd50e8a41d7bd90c7ef90602c9bf75`。
目录树 SHA256：`1b75eb3ad5c11cbafc73a7d2d8a9e22d0e47f618a4b4c818d905cd3b8dc81b0a`。

实际 smoke 病例：`/media/fudan22/4DFlow/Center012_P000203_Aorta/`；原生 `[4,25,5,19,96,88]`。
使用非空血管 Train 窗口，常规 S=3/T=5/Z=19/Y=96；CUT 使用 S=1；FlowMoDL 使用原生 S=64/T=6。
原始 z-y 尺寸和 coil 数未缩减。只读取了 Test 标签/路径元数据，未读 Test 数组内容。

## 逐项结果

PASS 仅表示列出的训练/求解接口和数值检查通过，不表示收敛或重建质量通过。

| 实验 | 状态 | optimizer steps | 最后 loss | grad norm | GPU GiB | 说明 |
|---|---|---:|---:|---:|---:|---|
| ablation_A0 | PASS | 1 | 0.486282 | 1.27511 | 4.793 |  |
| ablation_A1 | PASS | 1 | 0.511798 | 1.92856 | 6.656 |  |
| ablation_A2 | PASS | 2 | 0.425081 | 2.69028 | 10.378 |  |
| ablation_A3 | PASS | 2 | 0.482603 | 1.53289 | 7.242 |  |
| ablation_A4 | PASS | 2 | 0.818794 | 9.4429 | 7.242 |  |
| ours_full_final_dataset | PASS | 2 | 0.818794 | 9.4429 | 7.242 |  |
| cg_sense | PASS | solver | n/a | n/a | CPU | CG budget exhausted before tolerance |
| cs_tv | PASS | solver | n/a | n/a | CPU |  |
| flowvn | PASS | 1 | 0.815245 | 0.816876 | 3.428 |  |
| flowmodl | PASS | 1 | 10.5474 | 9.17904 | 3.736 | native 64x6 window verified on this Train case |
| cut_4dflow | PASS | 1 | 1.74776 | 2.80569 | 1.439 |  |

深度方法检查 DataLoader、forward、loss、backward、非零有限梯度、optimizer step、eval/no_grad 推理、
四编码 complex 输出、checkpoint 文件及回读的 model shape/optimizer state/scheduler。
记录的 GPU 峰值仅适用于上述小训练病例，不能用作大病例显存承诺。

## 额外覆盖

A2/A3/A4/Ours 两步测试检查 Mixer：首步 zero scale 允许 regularizer 梯度为零；
第二步 scale 移动后，`mixer_regularizer_grad_abs_sum` 应非零。
A4/Ours 临时 ramp=1 并检查 joint loss 对重建输出的非零有限梯度；
FlowMoDL 临时 curriculum epoch=32，确认 vel/ang 权重全开。正式训练的 ramp 没有改变。
CUT 按作者设置累计 5 个 Train 窗口后才更新一次 optimizer，并使用 EMA 权重做推理检查。
PathGuard、符号链接保护、测量算子 adjoint、消融开关与 split 标签测试见 `outputs/paper/unit_tests.xml`。

完整模型额外 checkpoint 恢复并再更新一步：PASS；详见 `outputs/paper/ours_full_resume_smoke/smoke/report.json`。

## 数据安全与代码隔离

所有 MRI/metadata 输入以只读方式打开。深度 smoke 前后核对所用 6 个数据文件 size/mtime 未变化。
CSV/任务书原件与项目内输入快照逐字节 SHA256 一致；不做原地转换、重分、改名或 dataset-side 缓存。
原有已跟踪文件改动：`[]`。对比适配和作者代码均位于 experiments/paper 下。

## 尚未覆盖

完整 epoch/收敛、最终测试指标、大病例显存/吞吐、多卡 DDP、全病例所有 raw-x/time 的推理均未验收。
传统求解器只验收 Train 窗口流程；CG 当前预算未达到目标容差，不能声称已收敛。
本包正式训练入口当前为单卡；不要用 torchrun 多次启动同一输出目录。
正式训练和后续评测由有足够资源的服务器执行，命令见 README 和 run_commands.sh。

## 环境

```json
{
  "torch": "2.6.0",
  "monai": "1.5.2",
  "numpy": "1.26.4",
  "scipy": "1.17.1",
  "h5py": "3.16.0",
  "einops": "0.8.0",
  "sigpy": "0.1.27",
  "hydra-core": "1.3.8",
  "zarr": "2.18.7",
  "numcodecs": "0.15.2",
  "pytest": "9.2.0"
}
```
