"""Sequential train-only checks; no full training or test-set evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .common.safety import OutputGuard, ROOT


NAMES = [*(f"ablation_A{i}" for i in range(5)), "ours_full_final_dataset",
         "cg_sense", "cs_tv", "flowvn", "flowmodl", "cut_4dflow"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    guard = OutputGuard()
    results = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    for name in args.names:
        baseline = name in NAMES[6:]
        cfg = ROOT / "configs/paper" / ("baselines" if baseline else "") / f"{name}.json"
        output = guard.directory(ROOT / "outputs/paper" / name / "smoke")
        log = guard.check(output / "console.log")
        module = "experiments.paper.comparisons.run" if baseline else "experiments.paper.ours.run"
        command = [sys.executable, "-m", module, "--config", str(cfg), "--smoke", "--device", args.device]
        command.extend(["--steps", str(json.loads(cfg.read_text())["paper"]["smoke_steps"])])
        if name in {"ablation_A4", "ours_full_final_dataset", "flowmodl"}:
            command.append("--probe-flow-loss")
        print(f"START {name}", flush=True)
        env = dict(os.environ, MPLCONFIGDIR="/tmp/paper_mpl", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", WANDB_MODE="disabled")
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "FAIL", "error": "No report written; see console.log"}
        record = {"name": name, "exit_code": completed.returncode, "report": str(report_path),
                  "status": report["status"], "command": command}
        results.append(record)
        guard.json(ROOT / "outputs/paper/smoke_runs" / f"{run_id}.json", results)
        print(f"{report['status']} {name} exit={completed.returncode} log={log}", flush=True)
    if any(r["exit_code"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
