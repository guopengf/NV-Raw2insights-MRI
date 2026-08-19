#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path


ORCHESTRATOR_PATH = Path(__file__).with_name("run_4dflow_challenge_submission.py")
RECOVERABLE_SHARDS = tuple(range(6))


def load_orchestrator():
    spec = importlib.util.spec_from_file_location("challenge_orchestrator", ORCHESTRATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load orchestrator: {ORCHESTRATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_provenance(orchestrator, preflight: dict) -> dict:
    provenance = preflight["provenance"]
    for name, raw_path in provenance["paths"].items():
        path = Path(raw_path)
        actual = orchestrator.sha256_file(path)
        expected = provenance["sha256"][name]
        if actual != expected:
            raise RuntimeError(
                f"Frozen provenance mismatch for {name}: expected={expected}, actual={actual}"
            )

    checkpoint = Path(provenance["paths"]["checkpoint"])
    metadata = orchestrator.checkpoint_metadata(checkpoint)
    if metadata != provenance["checkpoint_metadata"]:
        raise RuntimeError(
            "Checkpoint metadata drift: "
            f"expected={provenance['checkpoint_metadata']}, actual={metadata}"
        )
    return provenance


def completed_shard_provenance(
    orchestrator, run_root: Path, failed_index: int
) -> dict:
    summaries = []
    for shard_index, shard_spec in orchestrator.SHARDS.items():
        if shard_index == failed_index:
            continue
        summary_path = (
            run_root / "work" / shard_spec["work_name"] / "shard_summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing completed peer shard summary: {summary_path}"
            )
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("mode") != "full":
            raise RuntimeError(f"Invalid completed peer shard summary: {summary_path}")
        summaries.append((summary_path, summary["provenance"]))

    reference_path, reference = summaries[0]
    for summary_path, provenance in summaries[1:]:
        if provenance != reference:
            raise RuntimeError(
                "Completed peer shard provenance mismatch: "
                f"reference={reference_path}, mismatch={summary_path}"
            )
    return reference


def recovery_provenance(
    orchestrator, args: argparse.Namespace, run_root: Path
) -> tuple[dict, Path]:
    if args.skip_preflight:
        if args.data_base is None:
            raise ValueError("--data-base is required with --skip-preflight")
        provenance = completed_shard_provenance(
            orchestrator, run_root, args.shard_index
        )
        verified = verify_frozen_provenance(
            orchestrator, {"provenance": provenance}
        )
        return verified, args.data_base.resolve()

    preflight_path = run_root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "complete" or preflight.get("case_count") != 112:
        raise RuntimeError(f"Invalid preflight report: {preflight_path}")
    provenance = verify_frozen_provenance(orchestrator, preflight)
    return provenance, Path(preflight["data_base"])


def recover(args: argparse.Namespace) -> None:
    started = time.time()
    orchestrator = load_orchestrator()
    spec = orchestrator.SHARDS[args.shard_index]
    run_root = args.run_root.resolve()
    provenance, data_base = recovery_provenance(orchestrator, args, run_root)

    task_root = run_root / "work" / spec["work_name"]
    summary_path = task_root / "shard_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to recover completed shard: {summary_path}")

    records = json.loads((task_root / "case_inventory.json").read_text())
    inference_manifest = json.loads((task_root / "inference_manifest.json").read_text())
    reconstruction_manifest = json.loads(
        (task_root / "reconstruction_manifest.json").read_text()
    )
    expected_reconstructions = spec["full_cases"] * len(orchestrator.ENCODINGS)
    if (
        len(records) != spec["full_cases"]
        or len(inference_manifest) != spec["full_cases"]
        or len(reconstruction_manifest) != expected_reconstructions
    ):
        raise RuntimeError(
            "Failed-shard inventory mismatch: "
            f"cases={len(records)}, inputs={len(inference_manifest)}, "
            f"reconstructions={len(reconstruction_manifest)}"
        )

    temporary_root = task_root / "temporary"
    temporary_outputs = temporary_root / "val_img4ranking"
    expected_names = {f"{item['stem']}.mat" for item in reconstruction_manifest}
    existing_names = {path.name for path in temporary_outputs.glob("*.mat")}
    extra = sorted(existing_names - expected_names)
    if extra:
        raise RuntimeError(f"Unexpected existing reconstruction outputs: {extra[:20]}")
    if len(existing_names) >= expected_reconstructions:
        raise RuntimeError("No missing reconstruction outputs to recover")

    missing_names = expected_names - existing_names
    missing_input_stems = {
        item["input_stem"]
        for item in reconstruction_manifest
        if f"{item['stem']}.mat" in missing_names
    }
    recovery_inputs = [
        item for item in inference_manifest if item["stem"] in missing_input_stems
    ]
    if len(recovery_inputs) != len(missing_input_stems):
        raise RuntimeError(
            f"Unable to resolve all missing joint inputs: {len(recovery_inputs)} vs {len(missing_input_stems)}"
        )
    recovery_expected_names = {
        f"{item['stem']}.mat"
        for item in reconstruction_manifest
        if item["input_stem"] in missing_input_stems
    }

    final_root = task_root / "reconstructions"
    if final_root.exists() and any(final_root.rglob("*.mat")):
        raise RuntimeError(f"Final reconstructions already exist: {final_root}")
    submission_root = run_root / "submission"
    anatomy_submission = (
        submission_root / spec["task"] / "ValidationSet" / spec["anatomies"][0]
    )
    if anatomy_submission.exists() and any(anatomy_submission.rglob("*.npz")):
        raise RuntimeError(f"Submission outputs already exist: {anatomy_submission}")

    source_config = Path(provenance["paths"]["config"])
    config_payload = json.loads(source_config.read_text())
    original_batch_size = int(config_payload["batch_size"])
    if not 0 < args.batch_size < original_batch_size:
        raise ValueError(
            f"Retry batch size must be smaller than {original_batch_size}: {args.batch_size}"
        )
    config_payload["batch_size"] = args.batch_size
    config_payload["num_workers"] = args.num_workers
    effective_config = task_root / f"effective_config_retry_batch{args.batch_size}.json"
    recovery_json_root = task_root / f"recovery_jsons_batch{args.batch_size}"
    recovery_temporary = task_root / f"recovery_temporary_batch{args.batch_size}"
    for path in (effective_config, recovery_json_root, recovery_temporary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite previous recovery artifact: {path}")

    recovery_preflight = {
        "status": "ready",
        "shard_index": args.shard_index,
        "work_name": spec["work_name"],
        "expected_reconstructions": expected_reconstructions,
        "existing_reconstructions": len(existing_names),
        "missing_reconstructions": len(missing_names),
        "joint_cases_to_rerun": len(recovery_inputs),
        "recovery_outputs_to_compute": len(recovery_expected_names),
        "original_batch_size": original_batch_size,
        "retry_batch_size": args.batch_size,
        "nproc": args.nproc,
        "num_workers": args.num_workers,
        "provenance_sha256": provenance["sha256"],
    }
    print("[RECOVERY_PREFLIGHT]", json.dumps(recovery_preflight, indent=2), flush=True)
    if args.preflight_only:
        return

    effective_config.write_text(json.dumps(config_payload, indent=2) + "\n")
    recovery_json_root.mkdir()
    for item in recovery_inputs:
        source = Path(item["json"])
        shutil.copy2(source, recovery_json_root / source.name)
    checkpoint = Path(provenance["paths"]["checkpoint"])
    orchestrator.run_inference(
        effective_config,
        checkpoint,
        recovery_json_root,
        recovery_temporary,
        args.nproc,
    )

    recovery_outputs = recovery_temporary / "val_img4ranking"
    recovered_names = {path.name for path in recovery_outputs.glob("*.mat")}
    if recovered_names != recovery_expected_names:
        missing = sorted(recovery_expected_names - recovered_names)
        extra = sorted(recovered_names - recovery_expected_names)
        raise RuntimeError(
            f"Fresh recovery output mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )
    temporary_outputs.mkdir(parents=True, exist_ok=True)
    for source in recovery_outputs.glob("*.mat"):
        destination = temporary_outputs / source.name
        if destination.exists():
            if orchestrator.sha256_file(destination) != orchestrator.sha256_file(source):
                raise RuntimeError(f"Recovery conflicts with existing output: {destination}")
            continue
        shutil.move(str(source), str(destination))

    completed_names = {path.name for path in temporary_outputs.glob("*.mat")}
    if completed_names != expected_names:
        missing = sorted(expected_names - completed_names)
        extra = sorted(completed_names - expected_names)
        raise RuntimeError(f"Retry output mismatch: missing={missing[:20]}, extra={extra[:20]}")

    reconstructions = orchestrator.organize_reconstructions(
        reconstruction_manifest, temporary_root, final_root
    )
    split_root = data_base / spec["family"] / spec["task"] / "ValidationSet"
    submission_files = orchestrator.export_submission(
        records,
        final_root,
        submission_root,
        split_root,
        spec["task"],
        spec["anatomies"],
    )
    validated = [orchestrator.validate_sparse_npz(path) for path in submission_files]

    summary = {
        "status": "complete",
        "mode": "full",
        "family": spec["family"],
        "task": spec["task"],
        "split": "ValidationSet",
        "anatomies": spec["anatomies"],
        "work_name": spec["work_name"],
        "data_root": str(split_root),
        "output_root": str(run_root),
        "case_count": len(records),
        "joint_encoding_order": orchestrator.ENCODINGS,
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
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST",
            )
        },
        "nproc": args.nproc,
        "num_workers": args.num_workers,
        "elapsed_seconds": time.time() - started,
        "recovery": {
            "reason": "retry_after_failed_or_incomplete_shard",
            "existing_reconstructions_reused": len(existing_names),
            "reconstructions_generated": len(missing_names),
            "joint_cases_rerun": len(recovery_inputs),
            "recovery_outputs_computed": len(recovery_expected_names),
            "original_batch_size": original_batch_size,
            "retry_batch_size": args.batch_size,
            "nproc": args.nproc,
            "recovery_script": str(Path(__file__).resolve()),
            "recovery_script_sha256": orchestrator.sha256_file(Path(__file__).resolve()),
            "validated_submission_count": len(validated),
        },
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary_summary.replace(summary_path)
    print("[RECOVERY_COMPLETE]", json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a failed 4D-flow challenge shard from existing MAT outputs."
    )
    parser.add_argument("--shard-index", type=int, choices=RECOVERABLE_SHARDS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-base", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    recover(parse_args())
