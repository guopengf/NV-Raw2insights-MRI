"""Generate independent experiment configs from the pinned full-model template."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

from .common.safety import OutputGuard, ROOT, sha256


TEMPLATE = ROOT / "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_flow_unified_70_15_15.json"
INPUT_ROOT = Path("/SSDHome/share/haosen/4dflow/后续实验")
BASE_SHA = "9504c29e50237d96eee89f3cf47693c64482bd2f"
ONLINE_UNDERSAMPLING = {
    "enabled": True,
    "suite_path": "experiments/paper/final_mask_suite_20261118/code/mask_suite.py",
    "methods": ["gaussian", "pr4d", "pseudo", "uih"],
    "rate_mode": "continuous",
    "rate_range": [10, 50],
    "acs_zy": [3, 3],
    "pseudo_profile": "aorta_c001_R10",
    "cache_cases": 1024,
}


def configuration(name, manifest, tree):
    cfg = copy.deepcopy(json.loads(TEMPLATE.read_text()))
    cfg.update(seed=2026, data_path_train=[], data_path_val=[], resume_ckpt=None,
               pretrained_csm=None, pretrained_recon=None, resume_weights_only=True,
               val=False, num_workers=0, train_num_workers=0, ddp=False,
               data_aug=False, batch_size=4, exp=name, exp_dir="outputs/paper/",
               model_filename="latest.pt", group_accelerations_by_target=False,
               four_dflow_storage={"backend": "raw_mat", "train_backend": "raw_mat", "val_backend": "raw_mat"})
    p = cfg["phase3"]
    p["enable_vaa"] = False
    p["freeze"] = {"backbone": False, "vaa": True}
    p["validation"]["enabled"] = False
    p["validation"]["flow_metrics"]["enabled"] = False
    p["mra"]["cache_dir"] = "outputs/paper/cache/mra_disabled"
    cfg["paper"] = {"method": "ours", "name": name, "seed": 2026,
                    "manifest": "experiments/paper/inputs/final_dataset_manifest.csv",
                    "tree": "experiments/paper/inputs/final_dataset_tree.txt",
                    "train_index": "outputs/paper/index/train_index.json",
                    "validation_manifest": None, "validation_enabled": False,
                    "output_dir": f"outputs/paper/{name}",
                    "initialization": "independent_random", "smoke_steps": 1,
                    "num_slices": 3, "num_frames": 5, "samples_per_case": 8,
                    "loader": {"case_cache_mib": 512, "prefetch": True, "pin_limit_mib": 128},
                    "checkpoint_every_epochs": 1,
                    "online_undersampling": copy.deepcopy(ONLINE_UNDERSAMPLING),
                    "accelerations": [10, 20, 30, 40, 50],
                    "protected_roots": ["/media/Elements/4DFlow", "/media/fudan22/4DFlow"],
                    "checkpoint": None, "baseline_commit": BASE_SHA}
    if name.startswith("ablation_A"):
        stage = int(name[-1])
        cfg["paper"]["smoke_steps"] = 2 if stage >= 2 else 1
        p["backbone"]["spatial_dims"] = 2 if stage == 0 else 3
        p["flowvn_mixer"]["enabled"] = stage >= 2
        p["joint_encoding"]["enabled"] = stage >= 3
        p["loss"]["joint"]["enabled"] = stage >= 4
        if stage < 4:
            p["loss"]["profile"] = "pengfei_joint"
            for key in ("roi", "complex_roi", "cross_venc_phase", "velocity", "relerr", "angular", "flow_loss_ramp"):
                p["loss"][key]["enabled"] = False
    elif name == "ours_full_final_dataset":
        cfg["paper"]["smoke_steps"] = 2
    return cfg


def baseline(name, common):
    cfg = {"paper": copy.deepcopy(common["paper"]), "optimizer": {"lr": 1e-4, "weight_decay": 0.0},
           "num_epochs": 100, "num_workers": 0, "batch_size": 1}
    p = cfg["paper"]
    p.update(name=name, method=name, output_dir=f"outputs/paper/{name}", smoke_steps=1)
    if name in {"cg_sense", "cs_tv"}:
        p.pop("loader")
        p.pop("checkpoint_every_epochs")
    if name == "cg_sense":
        cfg["solver"] = {"iterations": 30, "tolerance": 1e-6, "l2": 1e-4}
    elif name == "cs_tv":
        cfg["solver"] = {"iterations": 100, "tolerance": 1e-5, "tv_weight": 0.002}
    elif name == "flowvn":
        cfg["author"] = {"path": "scripts/tools/FlowVN", "entrypoint": "main.py"}
        cfg["network"] = {"D_size": 7, "T_size": 5, "num_stages": 10, "features_in": 1,
                          "features_out": 24, "kernel_size": 7, "act": "linear", "num_act_weights": 71,
                          "grid": 0.25, "weight": 0.025, "vmin": -3.5, "vmax": 3.5,
                          "sgd_momentum": True, "exp_loss": False, "mode": "train"}
    elif name == "flowmodl":
        cfg["author"] = {"path": "experiments/paper/vendor/FlowMoDL", "url": "https://github.com/tgottwald/FlowMoDL.git",
                         "entrypoint": "train.py", "config_name": "config_flowmodl"}
        cfg["hydra"] = {"model": "flowmodl", "loss": "composite", "optimizer": "adamw"}
        cfg["num_epochs"], cfg["optimizer"] = 50, {"lr": 0.001, "weight_decay": 0.01}
        p.update(num_slices=64, num_frames=6)
    elif name == "cut_4dflow":
        cfg["author"] = {"path": "experiments/paper/vendor/CUT-4DFlow", "url": "https://github.com/Abdelrahman912/CUT-4DFlow.git",
                         "train_entrypoint": "scripts/train.sh", "recon_entrypoint": "scripts/recon.sh"}
        p.update(num_slices=1, num_frames=5, gradient_accumulation=5)
        cfg["optimizer"] = {"lr": 0.0005, "weight_decay": 0.001}
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    args = parser.parse_args()
    manifest = args.input_root / "dataset_info/final_dataset_manifest.csv"
    tree = args.input_root / "dataset_info/final_dataset_tree.txt"
    guard = OutputGuard([args.input_root])
    inputs = guard.directory(ROOT / "experiments/paper/inputs")
    source_files = [manifest, tree, args.input_root / "消融实验对比试验prompt.md",
                    args.input_root / "4DFlow_Comparison_Ablation_Codex_Execution_Fixed_20261115.docx"]
    for source in source_files:
        destination = guard.check(inputs / source.name)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Input snapshot checksum mismatch: {source}")
    generated = []
    for name in [*(f"ablation_A{i}" for i in range(5)), "ours_full_final_dataset"]:
        cfg = configuration(name, manifest, tree)
        path = guard.json(ROOT / "configs/paper" / f"{name}.json", cfg)
        generated.append((name, path, cfg))
    full_configuration = generated[-1][2]
    for name in ("cg_sense", "cs_tv", "flowvn", "flowmodl", "cut_4dflow"):
        cfg = baseline(name, full_configuration)
        path = guard.json(ROOT / "configs/paper/baselines" / f"{name}.json", cfg)
        generated.append((name, path, cfg))
    registry = {"baseline_git_sha": BASE_SHA, "current_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "manifest_sha256": sha256(manifest), "tree_sha256": sha256(tree), "seed": 2026,
                "source_snapshots": {p.name: sha256(p) for p in source_files},
                "validation": "disabled; no val label provided", "experiments": []}
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "$0")/../.."',
                'PYTHON="${PAPER_PYTHON:-python}"', "# Formal commands below are NOT executed by this generator."]
    for name, path, cfg in generated:
        rel = path.relative_to(ROOT)
        module = "experiments.paper.ours.run" if cfg["paper"]["method"] == "ours" else "experiments.paper.comparisons.run"
        command = f'"$PYTHON" -m {module} --config {rel}'
        if name in {"cg_sense", "cs_tv"}:
            command += " --smoke --device cpu"
        registry["experiments"].append({"name": name, "config": str(rel), "config_sha256": sha256(path),
                                        "output_dir": cfg["paper"]["output_dir"], "seed": 2026,
                                        "resource_note": "GPU memory/native-resolution runtime require smoke measurement"})
        script = guard.check(ROOT / "experiments/paper/scripts" / f"{name}.sh")
        guard.directory(script.parent)
        script.write_text('\n'.join(["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "$0")/../../.."',
                                    'PYTHON="${PAPER_PYTHON:-python}"', command + ' "$@"', ""]))
        commands.extend([f"# {name}", command])
    guard.json(ROOT / "experiments/paper/experiment_registry.json", registry)
    guard.check(ROOT / "experiments/paper/run_commands.sh").write_text("\n".join(commands) + "\n")
    print(f"Generated {len(generated)} isolated experiment configs and launch scripts")


if __name__ == "__main__":
    main()
