#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from path_safety import assert_outputs_not_in_data


ENCODINGS = [0, 1, 2, 3]
ACCEL_RE = re.compile(r"^kdata_ktGaussian(?P<acc>\d+)\.mat$")
EXPECTED_CHECKPOINT_SHA256 = "e1604419858be0592059458e2cdb186e101f02baf362457d847e6fd642a8dec8"
EXPECTED_CONFIG_SHA256 = "2f7936aac4e2d43a4b86795ad0cf22099fc38ef70bfc55ddf728de475aa22df2"
EXPECTED_INFERENCE_SHA256 = "3f5d291abcbea649de8b4f5fa6b7b850424738b7cd19d29f7aad0bf4410e67f7"
EXPECTED_EXPORTER_SHA256 = "45b140e492f45a24dbf972b7f44d3bb15b89be883bcd133f5b481d5da8dec06b"

TASKS = {
    0: {
        "family": "R1R2",
        "task": "TaskR1R2",
        "anatomies": ["Aorta"],
        "full_cases": 32,
    },
    1: {
        "family": "S1",
        "task": "TaskS1",
        "anatomies": ["Aorta"],
        "full_cases": 40,
    },
    2: {
        "family": "S2",
        "task": "TaskS2",
        "anatomies": ["Cerebrovascular", "Carotid", "PortalVein", "RenalArtery"],
        "full_cases": 40,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.strip()


def checkpoint_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        operations = list(pickletools.genops(archive.read(pickle_name)))

    wanted = {"epoch", "global_step", "epoch_finished", "wandb_run_id"}
    result = {}
    for index, (_opcode, argument, _position) in enumerate(operations):
        if argument not in wanted:
            continue
        for opcode, value, _next_position in operations[index + 1 : index + 8]:
            if opcode.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                continue
            if opcode.name in {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}:
                result[argument] = int(value)
            elif opcode.name == "NEWTRUE":
                result[argument] = True
            elif opcode.name == "NEWFALSE":
                result[argument] = False
            elif opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                result[argument] = str(value)
            break
    return result


def verify_provenance(config: Path, checkpoint: Path) -> dict:
    paths = {
        "checkpoint": checkpoint,
        "config": config,
        "inference": REPO_ROOT / "scripts" / "inference.py",
        "exporter": REPO_ROOT / "scripts" / "tools" / "export_4dflow_submission.py",
        "orchestrator": Path(__file__).resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "config": EXPECTED_CONFIG_SHA256,
        "inference": EXPECTED_INFERENCE_SHA256,
        "exporter": EXPECTED_EXPORTER_SHA256,
    }
    mismatches = {
        name: {"expected": expected[name], "actual": hashes[name]}
        for name in expected
        if hashes[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Provenance hash mismatch: {json.dumps(mismatches, indent=2)}")

    metadata = checkpoint_metadata(checkpoint)
    required_metadata = {
        "epoch": 85,
        "global_step": 14720,
        "epoch_finished": True,
        "wandb_run_id": "99f9z029",
    }
    if metadata != required_metadata:
        raise RuntimeError(
            f"Checkpoint metadata mismatch: expected={required_metadata}, actual={metadata}"
        )

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "checkpoint_metadata": metadata,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
    }


def discover_cases(split_root: Path, anatomies: list[str], mode: str) -> list[dict]:
    records = []
    for anatomy in anatomies:
        anatomy_root = split_root / anatomy
        if not anatomy_root.is_dir():
            raise FileNotFoundError(anatomy_root)

        anatomy_records = []
        for kspace in sorted(anatomy_root.rglob("kdata_ktGaussian*.mat")):
            match = ACCEL_RE.fullmatch(kspace.name)
            if match is None:
                raise ValueError(f"Unexpected k-space filename: {kspace}")

            case_dir = kspace.parent
            relative = case_dir.relative_to(anatomy_root)
            if len(relative.parts) != 3:
                raise ValueError(
                    f"Expected Center/Scanner/Patient below {anatomy_root}, got {case_dir}"
                )
            center, scanner, patient = relative.parts
            acceleration = int(match.group("acc"))
            mask = case_dir / f"usmask_ktGaussian{acceleration}.mat"
            coilmap = case_dir / "coilmap.mat"
            segmask = case_dir / "segmask.mat"
            missing = [str(path) for path in (mask, coilmap, segmask) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing required challenge inputs: {missing}")

            anatomy_records.append(
                {
                    "anatomy": anatomy,
                    "center": center,
                    "scanner": scanner,
                    "patient": patient,
                    "acceleration": acceleration,
                    "kspace": str(kspace),
                    "mask": str(mask),
                    "coilmap": str(coilmap),
                    "segmask": str(segmask),
                }
            )

        if not anatomy_records:
            raise RuntimeError(f"No challenge inputs found under {anatomy_root}")
        if mode == "smoke":
            anatomy_records = anatomy_records[:1]
        records.extend(anatomy_records)

    keys = [
        (item["anatomy"], item["center"], item["scanner"], item["patient"], item["acceleration"])
        for item in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate challenge case/acceleration records discovered")
    return records


def build_inference_manifest(records: list[dict], json_root: Path, task: str) -> list[dict]:
    json_root.mkdir(parents=True)
    manifest = []
    for record in records:
        for encoding in ENCODINGS:
            stem = "__".join(
                [
                    task,
                    record["anatomy"],
                    record["center"],
                    record["scanner"],
                    record["patient"],
                    f"ktGaussian{record['acceleration']}",
                    f"enc{encoding}",
                ]
            )
            json_path = json_root / f"{stem}.json"
            payload = {
                "kspace": record["kspace"],
                "target_kspace": record["kspace"],
                "mask": [record["mask"]],
                "mask_type": f"ktGaussian{record['acceleration']}",
                "acquisition": "Flow4d",
                "encoding_idx": encoding,
                "is_4dflow": True,
                "targetless": True,
                "coilmap": record["coilmap"],
                "segmask": record["segmask"],
            }
            json_path.write_text(json.dumps(payload, indent=2) + "\n")
            manifest.append({**record, "encoding": encoding, "stem": stem, "json": str(json_path)})
    return manifest


def write_effective_config(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text())
    payload["num_workers"] = 0
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def run_inference(
    config: Path,
    checkpoint: Path,
    json_root: Path,
    temporary_root: Path,
    nproc: int,
) -> None:
    inference = REPO_ROOT / "scripts" / "inference.py"
    command = [sys.executable]
    if nproc > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={nproc}",
            ]
        )
    command.extend(
        [
            str(inference),
            "--config",
            str(config),
            "--input_path",
            str(json_root),
            "--output_path",
            str(temporary_root),
            "--model_ckpt",
            str(checkpoint),
            "--save-coil-combined-output",
        ]
    )
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def organize_reconstructions(manifest: list[dict], temporary_root: Path, final_root: Path) -> list[Path]:
    source_root = temporary_root / "val_img4ranking"
    outputs = []
    missing = []
    for item in manifest:
        source = source_root / f"{item['stem']}.mat"
        destination = (
            final_root
            / item["anatomy"]
            / item["center"]
            / item["scanner"]
            / item["patient"]
            / f"kdata_ktGaussian{item['acceleration']}_enc{item['encoding']}_recon.mat"
        )
        if not source.is_file():
            missing.append(str(source))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        outputs.append(destination)
    if missing:
        raise FileNotFoundError("Missing inference outputs:\n" + "\n".join(missing[:20]))
    if len(outputs) != len(manifest):
        raise RuntimeError(f"Reconstruction count mismatch: {len(outputs)} vs {len(manifest)}")
    return outputs


def export_submission(
    records: list[dict],
    final_root: Path,
    submission_root: Path,
    split_root: Path,
    task: str,
    anatomies: list[str],
) -> list[Path]:
    exporter = REPO_ROOT / "scripts" / "tools" / "export_4dflow_submission.py"
    for anatomy in anatomies:
        command = [
            sys.executable,
            str(exporter),
            "--recon-root",
            str(final_root / anatomy),
            "--data-root",
            str(split_root / anatomy),
            "--out-root",
            str(submission_root),
            "--task",
            task,
            "--split",
            "ValidationSet",
            "--anatomy",
            anatomy,
            "--recon-layout",
            "yzxt",
            "--overwrite",
        ]
        print("[EXPORT]", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    expected = {
        submission_root
        / task
        / "ValidationSet"
        / item["anatomy"]
        / item["center"]
        / item["scanner"]
        / item["patient"]
        / f"img_ktGaussian{item['acceleration']}.npz"
        for item in records
    }
    actual = set((submission_root / task).rglob("img_ktGaussian*.npz"))
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(f"Submission path mismatch: missing={missing[:20]}, extra={extra[:20]}")
    return sorted(actual)


def run_task(args: argparse.Namespace) -> None:
    started = time.time()
    spec = TASKS[args.task_index]
    data_base = args.data_base.resolve()
    split_root = data_base / spec["family"] / spec["task"] / "ValidationSet"
    output_root = assert_outputs_not_in_data([args.output_root.resolve()], [data_base])[0]
    task_root = output_root / "work" / spec["task"]
    if task_root.exists():
        raise FileExistsError(f"Task output already exists: {task_root}")
    task_root.mkdir(parents=True)

    provenance = verify_provenance(args.config.resolve(), args.checkpoint.resolve())
    records = discover_cases(split_root, spec["anatomies"], args.mode)
    expected_cases = len(spec["anatomies"]) if args.mode == "smoke" else spec["full_cases"]
    if len(records) != expected_cases:
        raise RuntimeError(
            f"{spec['task']} {args.mode} case count mismatch: {len(records)} vs {expected_cases}"
        )

    json_root = task_root / "jsons"
    temporary_root = task_root / "temporary"
    final_root = task_root / "reconstructions"
    submission_root = output_root / "submission"
    manifest = build_inference_manifest(records, json_root, spec["task"])
    expected_reconstructions = expected_cases * len(ENCODINGS)
    if len(manifest) != expected_reconstructions:
        raise RuntimeError(
            f"Inference manifest count mismatch: {len(manifest)} vs {expected_reconstructions}"
        )

    (task_root / "case_inventory.json").write_text(json.dumps(records, indent=2) + "\n")
    (task_root / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    effective_config = task_root / "effective_config_num_workers0.json"
    write_effective_config(args.config.resolve(), effective_config)

    run_inference(effective_config, args.checkpoint.resolve(), json_root, temporary_root, args.nproc)
    reconstructions = organize_reconstructions(manifest, temporary_root, final_root)
    submission_files = export_submission(
        records,
        final_root,
        submission_root,
        split_root,
        spec["task"],
        spec["anatomies"],
    )

    summary = {
        "status": "complete",
        "mode": args.mode,
        "family": spec["family"],
        "task": spec["task"],
        "split": "ValidationSet",
        "anatomies": spec["anatomies"],
        "data_root": str(split_root),
        "output_root": str(output_root),
        "case_count": len(records),
        "inference_manifest_count": len(manifest),
        "reconstruction_count": len(reconstructions),
        "submission_count": len(submission_files),
        "expected_submission_relpaths": [
            str(path.relative_to(submission_root)) for path in submission_files
        ],
        "provenance": provenance,
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST")
        },
        "nproc": args.nproc,
        "elapsed_seconds": time.time() - started,
    }
    (task_root / "task_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[TASK_COMPLETE]", json.dumps(summary, indent=2), flush=True)


def validate_sparse_npz(path: Path) -> dict:
    with np.load(path) as payload:
        names = set(payload.files)
        if names != {"coords", "data", "shape"}:
            raise ValueError(f"Unexpected NPZ fields for {path}: {sorted(names)}")
        coords = payload["coords"]
        data = payload["data"]
        shape = tuple(int(value) for value in payload["shape"])
    if len(shape) != 5 or shape[0] != len(ENCODINGS):
        raise ValueError(f"Expected submission shape (4,Nt,SPE,PE,FE), got {shape}: {path}")
    if coords.ndim != 2 or coords.shape[1] != len(shape):
        raise ValueError(f"Invalid COO coordinates in {path}: {coords.shape}")
    if data.ndim != 1 or data.shape[0] != coords.shape[0]:
        raise ValueError(f"COO coordinate/data mismatch in {path}: {coords.shape}, {data.shape}")
    if not np.iscomplexobj(data) or not np.isfinite(data.real).all() or not np.isfinite(data.imag).all():
        raise ValueError(f"Submission data must be finite complex values: {path}")
    if coords.size:
        if (coords < 0).any() or any(int(coords[:, axis].max()) >= bound for axis, bound in enumerate(shape)):
            raise ValueError(f"Out-of-bounds COO coordinates: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(shape),
        "nnz": int(data.shape[0]),
    }


def write_zip(destination: Path, root: Path, members: list[Path]) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    expected_names = [str(path.relative_to(root)) for path in members]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, name in zip(members, expected_names):
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        actual_names = archive.namelist()
        bad = archive.testzip()
    if actual_names != expected_names or bad is not None:
        raise RuntimeError(f"ZIP verification failed for {destination}: bad={bad}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "member_count": len(actual_names),
    }


def package_submission(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    submission_root = run_root / "submission"
    expected_paths = set()
    summaries = {}
    for spec in TASKS.values():
        summary_path = run_root / "work" / spec["task"] / "task_summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("mode") != "full":
            raise RuntimeError(f"Task is not a completed full run: {summary_path}")
        if summary.get("case_count") != spec["full_cases"]:
            raise RuntimeError(f"Case count mismatch in {summary_path}")
        summaries[spec["task"]] = summary
        expected_paths.update(submission_root / path for path in summary["expected_submission_relpaths"])

    actual_paths = set(submission_root.rglob("img_ktGaussian*.npz"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(f"Final submission mismatch: missing={missing[:20]}, extra={extra[:20]}")
    if len(actual_paths) != 112:
        raise RuntimeError(f"Expected 112 submission files, found {len(actual_paths)}")

    validated = [validate_sparse_npz(path) for path in sorted(actual_paths)]
    artifacts_root = run_root / "artifacts"
    zip_reports = []
    for spec in TASKS.values():
        task_members = sorted((submission_root / spec["task"]).rglob("*.npz"))
        zip_reports.append(
            write_zip(
                artifacts_root / f"{spec['task']}_ValidationSet_epoch85.zip",
                submission_root,
                task_members,
            )
        )
    zip_reports.append(
        write_zip(
            artifacts_root / "Submission.zip",
            submission_root,
            sorted(actual_paths),
        )
    )

    report = {
        "status": "complete",
        "run_root": str(run_root),
        "submission_root": str(submission_root),
        "submission_count": len(actual_paths),
        "task_counts": {
            task: summary["submission_count"] for task, summary in summaries.items()
        },
        "npz_files": validated,
        "zip_artifacts": zip_reports,
        "elapsed_at_unix": time.time(),
    }
    report_path = artifacts_root / "artifact_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[PACKAGE_COMPLETE]", json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and package epoch-85 CMRx4DFlow2026 validation submissions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-task")
    run_parser.add_argument("--task-index", type=int, choices=sorted(TASKS), required=True)
    run_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    run_parser.add_argument("--data-base", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--nproc", type=int, required=True)
    run_parser.set_defaults(handler=run_task)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--run-root", type=Path, required=True)
    package_parser.set_defaults(handler=package_submission)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
