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

SHARDS = {
    0: {
        "family": "R1R2",
        "task": "TaskR1R2",
        "anatomies": ["Aorta"],
        "full_cases": 32,
        "work_name": "TaskR1R2__Aorta",
        "smoke_acceleration": 10,
    },
    1: {
        "family": "S1",
        "task": "TaskS1",
        "anatomies": ["Aorta"],
        "full_cases": 40,
        "work_name": "TaskS1__Aorta",
        "smoke_acceleration": 20,
    },
    2: {
        "family": "S2",
        "task": "TaskS2",
        "anatomies": ["Cerebrovascular"],
        "full_cases": 10,
        "work_name": "TaskS2__Cerebrovascular",
        "smoke_acceleration": 10,
    },
    3: {
        "family": "S2",
        "task": "TaskS2",
        "anatomies": ["Carotid"],
        "full_cases": 10,
        "work_name": "TaskS2__Carotid",
        "smoke_acceleration": 40,
    },
    4: {
        "family": "S2",
        "task": "TaskS2",
        "anatomies": ["PortalVein"],
        "full_cases": 10,
        "work_name": "TaskS2__PortalVein",
        "smoke_acceleration": 30,
    },
    5: {
        "family": "S2",
        "task": "TaskS2",
        "anatomies": ["RenalArtery"],
        "full_cases": 10,
        "work_name": "TaskS2__RenalArtery",
        "smoke_acceleration": 50,
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


def verify_provenance(
    config: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    expected_config_sha256: str = EXPECTED_CONFIG_SHA256,
    expected_inference_sha256: str = EXPECTED_INFERENCE_SHA256,
    expected_exporter_sha256: str = EXPECTED_EXPORTER_SHA256,
    expected_epoch: int = 85,
    expected_global_step: int = 14720,
    expected_wandb_run_id: str = "99f9z029",
) -> dict:
    paths = {
        "checkpoint": checkpoint,
        "config": config,
        "inference": REPO_ROOT / "scripts" / "inference.py",
        "exporter": REPO_ROOT / "scripts" / "tools" / "export_4dflow_submission.py",
        "orchestrator": Path(__file__).resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "checkpoint": expected_checkpoint_sha256,
        "config": expected_config_sha256,
        "inference": expected_inference_sha256,
        "exporter": expected_exporter_sha256,
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
        "epoch": expected_epoch,
        "global_step": expected_global_step,
        "epoch_finished": True,
        "wandb_run_id": expected_wandb_run_id,
    }
    if metadata != required_metadata:
        raise RuntimeError(
            f"Checkpoint metadata mismatch: expected={required_metadata}, actual={metadata}"
        )

    git_head = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    if git_status:
        raise RuntimeError(f"Inference worktree must be clean, found:\n{git_status}")

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "checkpoint_metadata": metadata,
        "git_head": git_head,
        "git_status": git_status,
    }


def verify_checkpoint_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        member_count = len(archive.infolist())
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Checkpoint ZIP integrity failed at member: {bad_member}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "member_count": member_count,
        "bad_member": bad_member,
    }


def provenance_from_args(args: argparse.Namespace) -> dict:
    return verify_provenance(
        config=args.config.resolve(),
        checkpoint=args.checkpoint.resolve(),
        expected_checkpoint_sha256=getattr(
            args, "expected_checkpoint_sha256", EXPECTED_CHECKPOINT_SHA256
        ),
        expected_config_sha256=getattr(
            args, "expected_config_sha256", EXPECTED_CONFIG_SHA256
        ),
        expected_inference_sha256=getattr(
            args, "expected_inference_sha256", EXPECTED_INFERENCE_SHA256
        ),
        expected_exporter_sha256=getattr(
            args, "expected_exporter_sha256", EXPECTED_EXPORTER_SHA256
        ),
        expected_epoch=getattr(args, "expected_epoch", 85),
        expected_global_step=getattr(args, "expected_global_step", 14720),
        expected_wandb_run_id=getattr(args, "expected_wandb_run_id", "99f9z029"),
    )


def discover_cases(
    split_root: Path,
    anatomies: list[str],
    mode: str,
    smoke_acceleration=None,
) -> list[dict]:
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
            if smoke_acceleration is not None:
                anatomy_records = [
                    record
                    for record in anatomy_records
                    if record["acceleration"] == smoke_acceleration
                ]
                if not anatomy_records:
                    raise RuntimeError(
                        f"No acceleration-{smoke_acceleration} smoke case under {anatomy_root}"
                    )
            anatomy_records = anatomy_records[:1]
        records.extend(anatomy_records)

    keys = [
        (item["anatomy"], item["center"], item["scanner"], item["patient"], item["acceleration"])
        for item in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate challenge case/acceleration records discovered")
    return records


def joint_encoding_order(config: Path) -> list[int] | None:
    payload = json.loads(config.read_text())
    joint = payload.get("phase3", {}).get("joint_encoding", {})
    if not bool(joint.get("enabled", False)):
        return None
    mode = str(joint.get("mode", "batch")).lower()
    if mode != "batch":
        raise ValueError(f"Challenge inference requires joint batch mode, got {mode!r}")
    count = int(joint.get("count", len(ENCODINGS)))
    order = [int(value) for value in joint.get("order", ENCODINGS)]
    if count != len(ENCODINGS) or order != ENCODINGS:
        raise ValueError(
            f"Challenge inference requires four ordered encodings {ENCODINGS}, "
            f"got count={count}, order={order}"
        )
    return order


def build_inference_manifest(
    records: list[dict],
    json_root: Path,
    task: str,
    joint_order: list[int] | None,
) -> tuple[list[dict], list[dict]]:
    json_root.mkdir(parents=True)
    inference_manifest = []
    reconstruction_manifest = []
    for record in records:
        base_parts = [
            task,
            record["anatomy"],
            record["center"],
            record["scanner"],
            record["patient"],
            f"ktGaussian{record['acceleration']}",
        ]
        encoding_groups = (
            [joint_order]
            if joint_order is not None
            else [[encoding] for encoding in ENCODINGS]
        )
        for encoding_group in encoding_groups:
            joint = len(encoding_group) > 1
            suffix = "joint4" if joint else f"enc{encoding_group[0]}"
            stem = "__".join(base_parts + [suffix])
            json_path = json_root / f"{stem}.json"
            payload = {
                "kspace": record["kspace"],
                "target_kspace": record["kspace"],
                "mask": [record["mask"]],
                "mask_type": f"ktGaussian{record['acceleration']}",
                "acquisition": "Flow4d",
                "is_4dflow": True,
                "targetless": True,
                "coilmap": record["coilmap"],
                "segmask": record["segmask"],
            }
            if joint:
                payload["joint_encodings"] = True
                payload["encoding_indices"] = encoding_group
            else:
                payload["encoding_idx"] = encoding_group[0]
            json_path.write_text(json.dumps(payload, indent=2) + "\n")
            inference_manifest.append(
                {
                    **record,
                    "stem": stem,
                    "json": str(json_path),
                    "joint_encodings": joint,
                }
            )
            for encoding in encoding_group:
                reconstruction_manifest.append(
                    {
                        **record,
                        "encoding": encoding,
                        "stem": f"{stem}__enc{encoding}" if joint else stem,
                        "input_stem": stem,
                        "json": str(json_path),
                        "joint_encodings": joint,
                    }
                )
    return inference_manifest, reconstruction_manifest


def write_effective_config(
    source: Path,
    destination: Path,
    num_workers: int,
    batch_size: int,
) -> None:
    if num_workers < 0:
        raise ValueError(f"num_workers must be nonnegative, got {num_workers}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    payload = json.loads(source.read_text())
    payload["num_workers"] = num_workers
    payload["batch_size"] = batch_size
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
    actual = set()
    for anatomy in anatomies:
        actual.update(
            (submission_root / task / "ValidationSet" / anatomy).rglob(
                "img_ktGaussian*.npz"
            )
        )
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(f"Submission path mismatch: missing={missing[:20]}, extra={extra[:20]}")
    return sorted(actual)


def run_spec(
    args: argparse.Namespace,
    spec: dict,
    work_name: str,
    summary_name: str,
) -> None:
    started = time.time()
    data_base = args.data_base.resolve()
    split_root = data_base / spec["family"] / spec["task"] / "ValidationSet"
    output_root = assert_outputs_not_in_data([args.output_root.resolve()], [data_base])[0]
    task_root = output_root / "work" / work_name
    if task_root.exists():
        raise FileExistsError(f"Task output already exists: {task_root}")
    task_root.mkdir(parents=True)

    encoding_order = joint_encoding_order(args.config.resolve())
    provenance = provenance_from_args(args)
    records = discover_cases(
        split_root,
        spec["anatomies"],
        args.mode,
        spec.get("smoke_acceleration"),
    )
    expected_cases = len(spec["anatomies"]) if args.mode == "smoke" else spec["full_cases"]
    if len(records) != expected_cases:
        raise RuntimeError(
            f"{spec['task']} {args.mode} case count mismatch: {len(records)} vs {expected_cases}"
        )

    json_root = task_root / "jsons"
    temporary_root = task_root / "temporary"
    final_root = task_root / "reconstructions"
    submission_root = output_root / "submission"
    inference_manifest, reconstruction_manifest = build_inference_manifest(
        records, json_root, spec["task"], encoding_order
    )
    expected_inputs = (
        expected_cases if encoding_order is not None else expected_cases * len(ENCODINGS)
    )
    expected_reconstructions = expected_cases * len(ENCODINGS)
    if (
        len(inference_manifest) != expected_inputs
        or len(reconstruction_manifest) != expected_reconstructions
    ):
        raise RuntimeError(
            "Manifest count mismatch: "
            f"inputs={len(inference_manifest)}/{expected_inputs}, "
            f"reconstructions={len(reconstruction_manifest)}/{expected_reconstructions}"
        )

    (task_root / "case_inventory.json").write_text(json.dumps(records, indent=2) + "\n")
    (task_root / "inference_manifest.json").write_text(
        json.dumps(inference_manifest, indent=2) + "\n"
    )
    (task_root / "reconstruction_manifest.json").write_text(
        json.dumps(reconstruction_manifest, indent=2) + "\n"
    )
    effective_config = task_root / (
        f"effective_config_batch{args.batch_size}_num_workers{args.num_workers}.json"
    )
    write_effective_config(
        args.config.resolve(),
        effective_config,
        args.num_workers,
        args.batch_size,
    )

    run_inference(effective_config, args.checkpoint.resolve(), json_root, temporary_root, args.nproc)
    reconstructions = organize_reconstructions(
        reconstruction_manifest, temporary_root, final_root
    )
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
        "work_name": work_name,
        "data_root": str(split_root),
        "output_root": str(output_root),
        "case_count": len(records),
        "joint_encoding_order": encoding_order,
        "inference_manifest_count": len(inference_manifest),
        "reconstruction_manifest_count": len(reconstruction_manifest),
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
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "elapsed_seconds": time.time() - started,
    }
    (task_root / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print("[TASK_COMPLETE]", json.dumps(summary, indent=2), flush=True)


def run_task(args: argparse.Namespace) -> None:
    spec = TASKS[args.task_index]
    run_spec(args, spec, spec["task"], "task_summary.json")


def run_shard(args: argparse.Namespace) -> None:
    spec = SHARDS[args.shard_index]
    run_spec(args, spec, spec["work_name"], "shard_summary.json")


def preflight_shards(args: argparse.Namespace) -> None:
    started = time.time()
    data_base = args.data_base.resolve()
    output_root = assert_outputs_not_in_data([args.output_root.resolve()], [data_base])[0]
    report_path = output_root / "preflight.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    encoding_order = joint_encoding_order(args.config.resolve())
    provenance = provenance_from_args(args)
    checkpoint_zip = verify_checkpoint_zip(args.checkpoint.resolve())
    shard_inventories = {}
    all_keys = []
    task_counts = {}
    acceleration_counts = {}
    for shard_index, spec in SHARDS.items():
        split_root = data_base / spec["family"] / spec["task"] / "ValidationSet"
        records = discover_cases(split_root, spec["anatomies"], "full")
        if len(records) != spec["full_cases"]:
            raise RuntimeError(
                f"{spec['work_name']} case count mismatch: {len(records)} vs {spec['full_cases']}"
            )
        shard_inventories[str(shard_index)] = {
            "work_name": spec["work_name"],
            "task": spec["task"],
            "anatomies": spec["anatomies"],
            "case_count": len(records),
            "inference_manifest_count": (
                len(records) if encoding_order is not None else len(records) * len(ENCODINGS)
            ),
            "reconstruction_manifest_count": len(records) * len(ENCODINGS),
        }
        task_counts[spec["task"]] = task_counts.get(spec["task"], 0) + len(records)
        for record in records:
            all_keys.append(
                (
                    spec["task"],
                    record["anatomy"],
                    record["center"],
                    record["scanner"],
                    record["patient"],
                    record["acceleration"],
                )
            )
            acceleration = str(record["acceleration"])
            acceleration_counts[acceleration] = acceleration_counts.get(acceleration, 0) + 1

    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("Duplicate records found across challenge shards")
    if len(all_keys) != 112 or task_counts != {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40}:
        raise RuntimeError(f"Unexpected challenge inventory: total={len(all_keys)}, tasks={task_counts}")

    report = {
        "status": "complete",
        "data_base": str(data_base),
        "output_root": str(output_root),
        "case_count": len(all_keys),
        "joint_encoding_order": encoding_order,
        "inference_manifest_count": (
            len(all_keys) if encoding_order is not None else len(all_keys) * len(ENCODINGS)
        ),
        "reconstruction_manifest_count": len(all_keys) * len(ENCODINGS),
        "task_counts": task_counts,
        "acceleration_counts": acceleration_counts,
        "shards": shard_inventories,
        "provenance": provenance,
        "checkpoint_zip": checkpoint_zip,
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[PREFLIGHT_COMPLETE]", json.dumps(report, indent=2), flush=True)


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


def package_shards(args: argparse.Namespace) -> None:
    if re.fullmatch(r"[A-Za-z0-9._-]+", args.artifact_tag) is None:
        raise ValueError(f"Unsafe artifact tag: {args.artifact_tag!r}")

    run_root = args.run_root.resolve()
    submission_root = run_root / "submission"
    preflight_path = run_root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "complete" or preflight.get("case_count") != 112:
        raise RuntimeError(f"Invalid preflight report: {preflight_path}")

    expected_paths = set()
    summaries = {}
    task_counts = {}
    reference_provenance = None
    for shard_index, spec in SHARDS.items():
        summary_path = run_root / "work" / spec["work_name"] / "shard_summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("mode") != "full":
            raise RuntimeError(f"Shard is not a completed full run: {summary_path}")
        if summary.get("case_count") != spec["full_cases"]:
            raise RuntimeError(f"Case count mismatch in {summary_path}")
        if summary.get("task") != spec["task"] or summary.get("anatomies") != spec["anatomies"]:
            raise RuntimeError(f"Shard identity mismatch in {summary_path}")
        provenance = summary.get("provenance")
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise RuntimeError(f"Cross-shard provenance mismatch in {summary_path}")
        summaries[str(shard_index)] = summary
        task_counts[spec["task"]] = task_counts.get(spec["task"], 0) + summary["submission_count"]
        expected_paths.update(submission_root / path for path in summary["expected_submission_relpaths"])

    if reference_provenance != preflight.get("provenance"):
        raise RuntimeError("Shard provenance does not match preflight provenance")
    if task_counts != {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40}:
        raise RuntimeError(f"Unexpected per-task submission counts: {task_counts}")

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
                artifacts_root
                / f"{spec['task']}_ValidationSet_{args.artifact_tag}.zip",
                submission_root,
                task_members,
            )
        )
    combined_zip_included = not getattr(args, "skip_combined_zip", False)
    if combined_zip_included:
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
        "artifact_tag": args.artifact_tag,
        "combined_zip_included": combined_zip_included,
        "submission_count": len(actual_paths),
        "task_counts": task_counts,
        "shards": summaries,
        "provenance": reference_provenance,
        "npz_files": validated,
        "zip_artifacts": zip_reports,
        "elapsed_at_unix": time.time(),
    }
    report_path = artifacts_root / "artifact_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[PACKAGE_COMPLETE]", json.dumps(report, indent=2), flush=True)


def add_expected_provenance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-inference-sha256", required=True)
    parser.add_argument("--expected-exporter-sha256", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--expected-wandb-run-id", required=True)


def add_run_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and package CMRx4DFlow2026 validation submissions."
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
    run_parser.add_argument("--batch-size", type=int, required=True)
    run_parser.add_argument("--num-workers", type=int, required=True)
    run_parser.set_defaults(handler=run_task)

    preflight_parser = subparsers.add_parser("preflight-shards")
    add_run_paths(preflight_parser)
    add_expected_provenance_args(preflight_parser)
    preflight_parser.set_defaults(handler=preflight_shards)

    shard_parser = subparsers.add_parser("run-shard")
    shard_parser.add_argument("--shard-index", type=int, choices=sorted(SHARDS), required=True)
    shard_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    add_run_paths(shard_parser)
    add_expected_provenance_args(shard_parser)
    shard_parser.add_argument("--nproc", type=int, required=True)
    shard_parser.add_argument("--batch-size", type=int, required=True)
    shard_parser.add_argument("--num-workers", type=int, required=True)
    shard_parser.set_defaults(handler=run_shard)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--run-root", type=Path, required=True)
    package_parser.set_defaults(handler=package_submission)

    shard_package_parser = subparsers.add_parser("package-shards")
    shard_package_parser.add_argument("--run-root", type=Path, required=True)
    shard_package_parser.add_argument("--artifact-tag", required=True)
    shard_package_parser.add_argument("--skip-combined-zip", action="store_true")
    shard_package_parser.set_defaults(handler=package_shards)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
