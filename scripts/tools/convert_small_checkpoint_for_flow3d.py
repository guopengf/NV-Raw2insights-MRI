#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from models.latent_recon import create_mri_recon_model
from utils import load_config, load_shape_compatible_state_dict


EXPECTED_SOURCE_SHA256 = "8ba66defef7c66738c1eac25f2ea6a5d164fb9e547c36d90b469d8d5287ed624"
EXPECTED_SOURCE_TENSORS = 1452
EXPECTED_TARGET_TENSORS = 1728
EXPECTED_EXACT_TENSORS = 918
EXPECTED_INFLATED_TENSORS = 534
EXPECTED_NEW_TARGET_TENSORS = 276
EXPECTED_CASCADES = 6
EXPECTED_TENSORS_PER_CASCADE = 242


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the public small checkpoint for the six-cascade Flow 3D model."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
        raise ValueError(f"Nonfinite tensor: {name}")


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_state_dict(source_state: dict, target_state: dict):
    converted = {}
    exact_keys = []
    inflated_keys = []

    for original_key, source_tensor in source_state.items():
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError(f"Checkpoint state value is not a tensor: {original_key}")

        source_key = original_key.removeprefix("module.")
        if not source_key.startswith("cascades."):
            raise KeyError(f"Unexpected source key prefix: {original_key}")
        target_key = f"recon_model.{source_key}"
        if target_key in converted:
            raise KeyError(f"Duplicate converted key: {target_key}")
        if target_key not in target_state:
            raise KeyError(f"Converted key is absent from target model: {target_key}")

        target_tensor = target_state[target_key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            converted[target_key] = source_tensor
            exact_keys.append(target_key)
        elif (
            source_tensor.ndim == 4
            and target_tensor.ndim == 5
            and tuple(source_tensor.shape[:2]) == tuple(target_tensor.shape[:2])
            and tuple(source_tensor.shape[-2:]) == tuple(target_tensor.shape[-2:])
        ):
            inflated = source_tensor.new_zeros(target_tensor.shape)
            center = inflated.shape[2] // 2
            inflated[:, :, center] = source_tensor
            if not torch.equal(inflated[:, :, center], source_tensor):
                raise AssertionError(f"Center-slice inflation mismatch: {target_key}")
            if center and torch.count_nonzero(inflated[:, :, :center]).item() != 0:
                raise AssertionError(f"Nonzero leading inflated slices: {target_key}")
            if center + 1 < inflated.shape[2] and torch.count_nonzero(inflated[:, :, center + 1 :]).item() != 0:
                raise AssertionError(f"Nonzero trailing inflated slices: {target_key}")
            converted[target_key] = inflated
            inflated_keys.append(target_key)
        else:
            raise ValueError(
                f"Incompatible tensor shape for {target_key}: "
                f"source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )

        require_finite(target_key, converted[target_key])

    return converted, exact_keys, inflated_keys


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output.resolve()
    manifest = (args.manifest or output.with_suffix(output.suffix + ".manifest.json")).resolve()

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    if output == source:
        raise ValueError("Output must not replace the source checkpoint")

    source_sha256 = sha256_file(source)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError(
            f"Source SHA256 mismatch: expected {args.expected_source_sha256}, got {source_sha256}"
        )
    config_sha256 = sha256_file(config_path)

    config = load_config(config_path)
    if config is None:
        raise ValueError(f"Could not load config: {config_path}")
    if not config.flow or config.num_cascades != EXPECTED_CASCADES:
        raise ValueError(
            f"Expected flow=true and num_cascades={EXPECTED_CASCADES}, "
            f"got flow={config.flow!r}, num_cascades={config.num_cascades!r}"
        )

    torch.manual_seed(0)
    model = create_mri_recon_model(config)
    target_state = model.state_dict()
    if len(target_state) != EXPECTED_TARGET_TENSORS:
        raise ValueError(
            f"Target tensor count mismatch: expected {EXPECTED_TARGET_TENSORS}, got {len(target_state)}"
        )

    source_checkpoint = torch_load(source)
    if not isinstance(source_checkpoint, dict) or "net_state_dict" not in source_checkpoint:
        raise KeyError("Source checkpoint does not contain net_state_dict")
    source_state = source_checkpoint["net_state_dict"]
    if len(source_state) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(
            f"Source tensor count mismatch: expected {EXPECTED_SOURCE_TENSORS}, got {len(source_state)}"
        )

    converted, exact_keys, inflated_keys = convert_state_dict(source_state, target_state)
    missing_target_keys = sorted(set(target_state) - set(converted))
    if len(converted) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"Converted tensor count mismatch: {len(converted)}")
    if len(exact_keys) != EXPECTED_EXACT_TENSORS:
        raise ValueError(f"Exact tensor count mismatch: {len(exact_keys)}")
    if len(inflated_keys) != EXPECTED_INFLATED_TENSORS:
        raise ValueError(f"Inflated tensor count mismatch: {len(inflated_keys)}")
    if len(missing_target_keys) != EXPECTED_NEW_TARGET_TENSORS:
        raise ValueError(f"New target tensor count mismatch: {len(missing_target_keys)}")

    cascade_pattern = re.compile(r"^recon_model\.cascades\.(\d+)\.")
    cascade_counts = Counter()
    for key in converted:
        match = cascade_pattern.match(key)
        if not match:
            raise ValueError(f"Converted key is not cascade-owned: {key}")
        cascade_counts[int(match.group(1))] += 1
    expected_cascade_counts = {
        index: EXPECTED_TENSORS_PER_CASCADE for index in range(EXPECTED_CASCADES)
    }
    if dict(sorted(cascade_counts.items())) != expected_cascade_counts:
        raise ValueError(
            f"Per-cascade tensor count mismatch: expected {expected_cascade_counts}, "
            f"got {dict(sorted(cascade_counts.items()))}"
        )

    created_at = datetime.now(timezone.utc).isoformat()
    conversion_metadata = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "source_path": str(args.source),
        "source_sha256": source_sha256,
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "source_tensor_count": len(source_state),
        "target_tensor_count": len(target_state),
        "transferred_tensor_count": len(converted),
        "exact_tensor_count": len(exact_keys),
        "center_inflated_tensor_count": len(inflated_keys),
        "new_target_tensor_count": len(missing_target_keys),
        "new_target_tensors_are_included": False,
    }
    atomic_torch_save(
        {
            "net_state_dict": converted,
            "compatibility_conversion": conversion_metadata,
        },
        output,
    )

    reloaded = torch_load(output)
    reloaded_state = reloaded["net_state_dict"]
    if len(reloaded_state) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"Reloaded tensor count mismatch: {len(reloaded_state)}")
    for key, tensor in reloaded_state.items():
        require_finite(key, tensor)

    loaded, unchanged, skipped, unexpected, loader_inflated = load_shape_compatible_state_dict(
        model, reloaded_state
    )
    if len(loaded) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"Dry-run loaded tensor count mismatch: {len(loaded)}")
    if len(unchanged) != EXPECTED_NEW_TARGET_TENSORS:
        raise ValueError(f"Dry-run unchanged tensor count mismatch: {len(unchanged)}")
    if skipped or unexpected or loader_inflated:
        raise ValueError(
            "Dry-run loader mismatch: "
            f"skipped={len(skipped)}, unexpected={len(unexpected)}, "
            f"loader_inflated={len(loader_inflated)}"
        )

    output_sha256 = sha256_file(output)
    manifest_payload = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "source": {
            "path": str(args.source),
            "resolved_path": str(source),
            "is_symlink": args.source.is_symlink(),
            "symlink_target": os.readlink(args.source) if args.source.is_symlink() else None,
            "sha256": source_sha256,
            "bytes": source.stat().st_size,
            "checkpoint_keys": sorted(source_checkpoint.keys()),
            "net_tensor_count": len(source_state),
        },
        "config": {
            "path": str(args.config),
            "sha256": config_sha256,
            "model_variant": config.model_variant,
            "experiment": config.exp,
            "flow": config.flow,
            "num_cascades": config.num_cascades,
            "time_cond": config.time_cond,
            "label_cond": config.label_cond,
        },
        "artifact": {
            "path": str(output),
            "sha256": output_sha256,
            "bytes": output.stat().st_size,
            "net_tensor_count": len(reloaded_state),
            "weights_only": True,
        },
        "conversion": {
            "rename": "cascades.* -> recon_model.cascades.*",
            "conv2d_to_conv3d": "copy source kernel into center depth slice; zero all other depth slices",
            "transferred_tensor_count": len(converted),
            "exact_tensor_count": len(exact_keys),
            "center_inflated_tensor_count": len(inflated_keys),
            "new_target_tensor_count": len(missing_target_keys),
            "new_target_tensors_are_included": False,
            "per_cascade_transferred": {
                str(key): value for key, value in sorted(cascade_counts.items())
            },
            "missing_target_keys": missing_target_keys,
        },
        "validation": {
            "updated_keys": len(loaded),
            "unchanged_keys": len(unchanged),
            "skipped_shape_keys": len(skipped),
            "unexpected_keys": len(unexpected),
            "loader_inflated_keys": len(loader_inflated),
            "all_transferred_tensors_finite": True,
            "checkpoint_reload_passed": True,
        },
        "repository": {
            "root": str(REPO_ROOT),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short").splitlines(),
        },
    }
    atomic_json_save(manifest_payload, manifest)

    print(json.dumps({
        "output": str(output),
        "output_sha256": output_sha256,
        "manifest": str(manifest),
        "updated_keys": len(loaded),
        "unchanged_keys": len(unchanged),
        "skipped_shape_keys": len(skipped),
        "unexpected_keys": len(unexpected),
    }, indent=2))


if __name__ == "__main__":
    main()
