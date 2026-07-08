#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from models.vaa import VascularAttentionAdapter
from mra_utils import load_vessel_mask_prior
from utils import normalize_recon_mode, validate_phase3_config, windowed_input_x_slab


class LegacyReferenceAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, gamma_init: float = 0.0):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    def gamma(self) -> torch.Tensor:
        base = torch.sigmoid(torch.zeros((), device=self.gamma_raw.device, dtype=self.gamma_raw.dtype))
        return torch.clamp((torch.sigmoid(self.gamma_raw) - base) / (1.0 - base), 0.0, 1.0)

    def forward(self, feature: torch.Tensor, vessel_map: torch.Tensor) -> torch.Tensor:
        if vessel_map.shape[-2:] != feature.shape[-2:]:
            vessel_map = nn.functional.interpolate(
                vessel_map, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
        vessel_map = vessel_map.to(device=feature.device, dtype=feature.dtype).clamp(0.0, 1.0)
        delta = self.net(torch.cat([feature, vessel_map], dim=1))
        return feature + self.gamma().to(dtype=feature.dtype) * delta


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def check_configs() -> dict[str, object]:
    expected = {
        "nv_raw2insights_mri_base_4dflow_pg.json": {
            "attention": "legacy",
            "recon_mode": "slice",
            "enable_vaa": False,
        },
        "nv_raw2insights_mri_base_4dflow_vaa_pg.json": {
            "attention": "legacy",
            "recon_mode": "slice",
            "enable_vaa": True,
        },
        "nv_raw2insights_mri_base_4dflow_slab_no_vaa.json": {
            "attention": None,
            "recon_mode": "slab",
            "enable_vaa": False,
            "freeze_backbone": False,
            "freeze_vaa": True,
        },
        "nv_raw2insights_mri_base_4dflow_mask_vaa_slab.json": {
            "attention": "qkv",
            "recon_mode": "slab",
            "enable_vaa": True,
            "freeze_backbone": True,
            "freeze_vaa": False,
        },
    }
    observed = {}
    raw_configs = {}
    for name, expected_values in expected.items():
        data = json.loads((REPO_ROOT / "configs" / name).read_text())
        raw_configs[name] = data
        args = to_namespace(data)
        validate_phase3_config(args)
        vaa = getattr(args.phase3, "vaa", None)
        freeze = getattr(args.phase3, "freeze", None)
        values = {
            "attention": getattr(vaa, "attention", None),
            "recon_mode": args.phase3.recon_mode,
            "enable_vaa": args.phase3.enable_vaa,
        }
        if freeze is not None:
            values.update(
                freeze_backbone=freeze.backbone,
                freeze_vaa=freeze.vaa,
            )
        for key, expected_value in expected_values.items():
            if values.get(key) != expected_value:
                raise AssertionError(f"{name}: expected {key}={expected_value!r}, got {values.get(key)!r}")
        observed[name] = values

    baseline = raw_configs["nv_raw2insights_mri_base_4dflow_slab_no_vaa.json"]
    followup = raw_configs["nv_raw2insights_mri_base_4dflow_mask_vaa_slab.json"]
    expected_resume = str(Path(baseline["exp_dir"]) / baseline["exp"] / baseline["model_filename"])
    if followup["resume_ckpt"] != expected_resume:
        raise AssertionError(
            "Slab VAA checkpoint handoff mismatch: "
            f"expected {expected_resume!r}, got {followup['resume_ckpt']!r}"
        )
    return observed


def check_adapters(device: torch.device) -> dict[str, float]:
    torch.manual_seed(20260708)
    feature = torch.randn(2, 16, 8, 10, device=device)
    prior = torch.rand(2, 1, 8, 10, device=device)

    legacy = VascularAttentionAdapter(16, reduction=4, gamma_init=1.0, attention="legacy").to(device)
    reference = LegacyReferenceAdapter(16, reduction=4, gamma_init=1.0).to(device)
    reference.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        legacy_out = legacy(feature, prior)
        reference_out = reference(feature, prior)
    legacy_error = float((legacy_out - reference_out).abs().max().cpu())
    if legacy_error != 0.0:
        raise AssertionError(f"Legacy adapter mismatch: max_abs={legacy_error}")

    results = {"legacy_max_abs_error": legacy_error}
    for mode in ("legacy", "gate", "qkv"):
        adapter = VascularAttentionAdapter(
            16,
            reduction=4,
            gamma_init=0.0,
            attention=mode,
            num_heads=4,
            attention_stride=2,
        ).to(device)
        with torch.no_grad():
            output = adapter(feature, prior)
        identity_error = float((output - feature).abs().max().cpu())
        if identity_error != 0.0:
            raise AssertionError(f"{mode} gamma-zero identity failed: max_abs={identity_error}")
        results[f"{mode}_identity_max_abs_error"] = identity_error
    return results


def check_slab_window() -> dict[str, object]:
    total_frames = 5
    total_slices = 4
    source = torch.arange(total_frames * total_slices, dtype=torch.float32).reshape(-1, 1)
    final_shape = (1, 1, total_frames, total_slices, 1, 1, 2)
    gathered, indices = windowed_input_x_slab(
        source,
        micro_b=[0, total_frames * total_slices - 1],
        final_shape=final_shape,
        num_frames=5,
        num_slices=3,
    )
    if tuple(indices.shape) != (2, 3, 5):
        raise AssertionError(f"Unexpected slab index shape: {tuple(indices.shape)}")
    if not torch.equal(indices[0, 0], indices[0, 1]):
        raise AssertionError("Lower slab boundary did not use replicated indices")
    if not torch.equal(indices[1, 1], indices[1, 2]):
        raise AssertionError("Upper slab boundary did not use replicated indices")
    return {"gathered_shape": list(gathered.shape), "index_shape": list(indices.shape)}


def check_mask_loader() -> dict[str, object]:
    mask = np.zeros((2, 3, 4), dtype=np.float32)
    mask[1, 2, 3] = 1.0
    args = to_namespace(
        {
            "phase3": {
                "mask": {
                    "keys": ["segmask"],
                    "axis_order": "zyx",
                    "binary": True,
                    "threshold": 0.5,
                }
            }
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segmask.mat"
        scipy.io.savemat(path, {"segmask": mask})
        loaded = load_vessel_mask_prior(path, args)
    if loaded.shape != (4, 2, 3) or loaded[3, 1, 2] != 1.0:
        raise AssertionError(f"Unexpected loaded mask shape/content: {loaded.shape}")
    return {"shape": list(loaded.shape), "nonzero": int(np.count_nonzero(loaded))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    if normalize_recon_mode("2.5d") != "slice" or normalize_recon_mode("3d") != "slab":
        raise AssertionError("Reconstruction mode aliases are incorrect")

    result = {
        "device": str(device),
        "configs": check_configs(),
        "adapters": check_adapters(device),
        "slab_window": check_slab_window(),
        "mask_loader": check_mask_loader(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
