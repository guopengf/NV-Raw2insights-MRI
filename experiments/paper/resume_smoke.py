"""Verify one extra real train update from the full-model readiness checkpoint."""

import argparse
import json
import subprocess
import sys

from .common.safety import OutputGuard, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    guard = OutputGuard()
    cfg = json.loads((ROOT / "configs/paper/ours_full_final_dataset.json").read_text())
    cfg["paper"].update(name="ours_full_resume_smoke", output_dir="outputs/paper/ours_full_resume_smoke",
                        checkpoint=str(ROOT / "outputs/paper/ours_full_final_dataset/smoke/latest.pt"))
    config = guard.json(ROOT / "outputs/paper/resume_smoke_config.json", cfg)
    command = [sys.executable, "-m", "experiments.paper.ours.run", "--config", str(config),
               "--smoke", "--steps", "1", "--device", args.device, "--probe-flow-loss"]
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads((ROOT / "outputs/paper/ours_full_resume_smoke/smoke/report.json").read_text())
    if report["resume_epoch"] != 1 or report["resume_step"] != 2:
        raise AssertionError(f"Wrong resume cursor: {report}")
    if report["steps"][0]["small_parameters_changed"] <= 0:
        raise AssertionError("No parameter changed after restoring checkpoint")
    print("PASS actual full-model checkpoint resume: epoch=1 step=2 -> one extra optimizer update")


if __name__ == "__main__":
    main()
