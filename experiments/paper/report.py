"""Build a source-backed readiness report; a missing/stale check is never PASS."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common.safety import OutputGuard, ROOT, sha256
from .smoke import NAMES


def main():
    guard = OutputGuard()
    registry_path = ROOT / "experiments/paper/experiment_registry.json"
    registry = json.loads(registry_path.read_text())
    availability = json.loads((ROOT / "outputs/paper/index/availability.json").read_text())["summary"]
    packages = ("torch", "monai", "numpy", "scipy", "h5py", "einops", "sigpy", "hydra-core", "zarr", "numcodecs", "pytest")
    versions = {name: importlib.metadata.version(name) for name in packages}
    source_diffs = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()
    lock = json.loads((ROOT / "experiments/paper/vendor_lock.json").read_text())
    for item in lock.values():
        if "url" in item:
            item["actual_commit"] = subprocess.check_output(["git", "-C", str(ROOT / item["path"]), "rev-parse", "HEAD"], text=True).strip()
            if item["actual_commit"] != item["commit"]:
                raise RuntimeError(f"Author commit mismatch: {item}")
    rows, failures = [], []
    for item in registry["experiments"]:
        name = item["name"]
        report_path = ROOT / "outputs/paper" / name / "smoke/report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "NOT_RUN"}
        status = report["status"]
        if report.get("config_sha256") and report["config_sha256"] != sha256(ROOT / item["config"]):
            status = "STALE_CONFIG"
        if name in {"ablation_A2", "ablation_A3", "ablation_A4", "ours_full_final_dataset"}:
            steps = report.get("steps", [])
            if len(steps) < 2 or steps[-1].get("mixer_regularizer_grad_abs_sum", 0.0) <= 0:
                status = "MIXER_COVERAGE_MISSING"
        if name in {"ablation_A4", "ours_full_final_dataset"} and report.get("steps"):
            if report["steps"][-1]["components"].get("joint_output_gradient_norm", 0.0) <= 0:
                status = "FLOW_GRADIENT_COVERAGE_MISSING"
        if status != "PASS":
            failures.append(name)
        steps = report.get("steps", [])
        last = steps[-1] if steps else {}
        loss = f"{last['loss']:.6g}" if last else "n/a"
        gradient = f"{last['gradient_norm']:.6g}" if last else "n/a"
        gpu = f"{report['peak_gpu_gib']:.3f}" if "peak_gpu_gib" in report else "CPU"
        caveat = "CG budget exhausted before tolerance" if report.get("warnings") else ""
        if name == "flowmodl":
            caveat = "native 64x6 window verified on this Train case"
        item.update(smoke_status=status, smoke_report=str(report_path.relative_to(ROOT)),
                    smoke_steps=len(steps), peak_gpu_gib=report.get("peak_gpu_gib"),
                    measured_native_shape=last.get("native_shape"), test_data_used=report.get("test_data_used"),
                    notes=caveat)
        rows.append(f"| {name} | {status} | {len(steps) if steps else 'solver'} | {loss} | {gradient} | {gpu} | {caveat} |")
    registry.update(environment=versions, author_versions=lock, original_tracked_files_changed=source_diffs,
                    report_generated_utc=datetime.now(timezone.utc).isoformat())
    guard.json(registry_path, registry)
    resume_path = ROOT / "outputs/paper/ours_full_resume_smoke/smoke/report.json"
    resume_report = json.loads(resume_path.read_text()) if resume_path.exists() else {"status": "NOT_RUN"}
    if resume_report["status"] != "PASS":
        failures.append("full_model_resume")
    unit_path = guard.check(ROOT / "outputs/paper/unit_tests.xml")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "experiments/paper/test_contracts.py", f"--junitxml={unit_path}"], cwd=ROOT)
    if result.returncode:
        failures.append("unit_tests")
    text = ["# 训练就绪性预检报告", "", f"生成时间（UTC）：{registry['report_generated_utc']}", "",
            "基准分支：`pengfeig/4dflow-finetune`；Git SHA：`9504c29e50237d96eee89f3cf47693c64482bd2f`。",
            "本轮没有运行完整训练、任何 Test 推理/指标或测试集模型选择。", "",
            "## 数据检查", "",
            f"Train：{availability['case_counts']['train']}；ordinary_test：{availability['case_counts']['ordinary_test']}；focused_test：{availability['case_counts']['focused_test']}。",
            f"已读 Training 文件头：{availability['train_available']} 个病例；缺失/异常：{availability['train_unavailable']}。",
            "保留 CSV 全部原始标签和 README 元数据行；无独立 val，验证关闭。",
            f"CSV SHA256：`{registry['manifest_sha256']}`。",
            f"目录树 SHA256：`{registry['tree_sha256']}`。", "",
            "实际 smoke 病例：`/media/fudan22/4DFlow/Center012_P000203_Aorta/`；原生 `[4,25,5,19,96,88]`。",
            "使用非空血管 Train 窗口，常规 S=3/T=5/Z=19/Y=96；CUT 使用 S=1；FlowMoDL 使用原生 S=64/T=6。",
            "原始 z-y 尺寸和 coil 数未缩减。只读取了 Test 标签/路径元数据，未读 Test 数组内容。", "",
            "## 逐项结果", "",
            "PASS 仅表示列出的训练/求解接口和数值检查通过，不表示收敛或重建质量通过。", "",
            "| 实验 | 状态 | optimizer steps | 最后 loss | grad norm | GPU GiB | 说明 |",
            "|---|---|---:|---:|---:|---:|---|", *rows, "",
            "深度方法检查 DataLoader、forward、loss、backward、非零有限梯度、optimizer step、eval/no_grad 推理、",
            "四编码 complex 输出、checkpoint 文件及回读的 model shape/optimizer state/scheduler。",
            "记录的 GPU 峰值仅适用于上述小训练病例，不能用作大病例显存承诺。", "",
            "## 额外覆盖", "",
            "A2/A3/A4/Ours 两步测试检查 Mixer：首步 zero scale 允许 regularizer 梯度为零；",
            "第二步 scale 移动后，`mixer_regularizer_grad_abs_sum` 应非零。",
            "A4/Ours 临时 ramp=1 并检查 joint loss 对重建输出的非零有限梯度；",
            "FlowMoDL 临时 curriculum epoch=32，确认 vel/ang 权重全开。正式训练的 ramp 没有改变。",
            "CUT 按作者设置累计 5 个 Train 窗口后才更新一次 optimizer，并使用 EMA 权重做推理检查。",
            "PathGuard、符号链接保护、测量算子 adjoint、消融开关与 split 标签测试见 `outputs/paper/unit_tests.xml`。", "",
            f"完整模型额外 checkpoint 恢复并再更新一步：{resume_report['status']}；详见 `outputs/paper/ours_full_resume_smoke/smoke/report.json`。", "",
            "## 数据安全与代码隔离", "",
            "所有 MRI/metadata 输入以只读方式打开。深度 smoke 前后核对所用 6 个数据文件 size/mtime 未变化。",
            "CSV/任务书原件与项目内输入快照逐字节 SHA256 一致；不做原地转换、重分、改名或 dataset-side 缓存。",
            f"原有已跟踪文件改动：`{source_diffs}`。对比适配和作者代码均位于 experiments/paper 下。", "",
            "## 尚未覆盖", "",
            "完整 epoch/收敛、最终测试指标、大病例显存/吞吐、多卡 DDP、全病例所有 raw-x/time 的推理均未验收。",
            "传统求解器只验收 Train 窗口流程；CG 当前预算未达到目标容差，不能声称已收敛。",
            "本包正式训练入口当前为单卡；不要用 torchrun 多次启动同一输出目录。",
            "正式训练和后续评测由有足够资源的服务器执行，命令见 README 和 run_commands.sh。", "",
            "## 环境", "", "```json", json.dumps(versions, indent=2), "```", ""]
    destination = guard.check(ROOT / "experiments/paper/preflight_report.md")
    destination.write_text("\n".join(text), encoding="utf-8")
    print(f"Report: {destination}; incomplete={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
