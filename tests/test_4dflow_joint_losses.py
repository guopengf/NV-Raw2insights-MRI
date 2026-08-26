from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from joint_encoding import JointEncodingLoss, joint_encoding_spec
from mra_utils import load_roi_loss_mask, roi_loss_mask_needed, vaa_prior_needed
from mri_data.flow_losses import (
    BestFlowMetrics,
    angular_cosine_loss,
    circular_phase_loss,
    complex2magflow_torch,
    complex_roi_l1_loss,
    flow_loss_ramp,
    relerr_loss,
    select_training_center_time,
    velocity_component_loss,
)
from train import resolve_training_loss_flags
from utils import load_config


OFFICIAL_DIR = Path("/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode")


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def _roi_args(axis_order="zyx"):
    return _namespace(
        phase3=_namespace(
            enable_vaa=False,
            loss=_namespace(
                roi=_namespace(
                    enabled=True,
                    source="segmask",
                    keys=["segmask"],
                    axis_order=axis_order,
                    binary=True,
                    threshold=0.5,
                )
            ),
        )
    )


def _official_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roi_mask_loading_is_independent_from_vaa_and_oriented_xzy():
    args = _roi_args()
    assert roi_loss_mask_needed(args)
    assert not vaa_prior_needed(args)
    source = np.ones((2, 3, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "segmask.mat"
        scipy.io.savemat(path, {"segmask": source})
        mask = load_roi_loss_mask(path, args)
    assert mask.shape == (4, 2, 3)


def test_complex2magflow_matches_official():
    official = _official_module("utils_flow.py", "official_utils_flow")
    rng = np.random.default_rng(7)
    values = (rng.normal(size=(4, 2, 3, 4, 5)) + 1j * rng.normal(size=(4, 2, 3, 4, 5))).astype(
        np.complex64
    )
    expected_mag, expected_flow = official.complex2magflow(values)
    actual_mag, actual_flow = complex2magflow_torch(torch.view_as_real(torch.from_numpy(values)), encoding_dim=0)
    np.testing.assert_allclose(actual_mag.numpy(), expected_mag, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_flow.numpy(), expected_flow, rtol=1e-6, atol=1e-6)


def test_relerr_matches_official_with_joint_singleton_coil_dimension():
    sys.path.insert(0, str(OFFICIAL_DIR))
    official = _official_module("utils_metrics.py", "official_utils_metrics")
    rng = np.random.default_rng(9)
    target = rng.normal(size=(3, 2, 4, 5, 6)).astype(np.float32)
    pred = rng.normal(size=target.shape).astype(np.float32)
    mask = (rng.random(size=(2, 4, 5, 6)) > 0.2).astype(np.float32)
    expected = official.RelErr(pred, target, mask[0])
    actual = relerr_loss(
        torch.from_numpy(pred[:, :1]).permute(1, 0, 2, 3, 4).unsqueeze(3),
        torch.from_numpy(target[:, :1]).permute(1, 0, 2, 3, 4).unsqueeze(3),
        torch.from_numpy(mask[:1]),
        component_dim=1,
    )
    expected_first = official.RelErr(pred[:, :1], target[:, :1], mask[0])
    np.testing.assert_allclose(actual.numpy(), expected_first, rtol=1e-6, atol=1e-6)
    assert np.isfinite(expected)


def test_roi_losses_fall_back_to_full_image_when_mask_is_missing():
    pred = torch.randn(2, 4, 3, 1, 5, 6, 2, requires_grad=True)
    target = torch.randn_like(pred)
    pred_flow = complex2magflow_torch(pred)[1]
    target_flow = complex2magflow_torch(target)[1]
    losses = (
        complex_roi_l1_loss(pred, target, None),
        circular_phase_loss(pred, target, None),
        velocity_component_loss(pred_flow, target_flow, None),
        relerr_loss(pred_flow, target_flow, None),
        angular_cosine_loss(pred_flow, target_flow, None),
    )
    for loss in losses:
        gradient = torch.autograd.grad(loss, pred, retain_graph=True)[0]
        assert torch.isfinite(loss)
        assert torch.isfinite(gradient).all()


def test_pengfei_profile_uses_exact_relerr_for_speed_component():
    pred = torch.randn(2, 4, 3, 1, 5, 6, 2, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.ones(2, 3, 5, 6)
    loss, components = JointEncodingLoss(profile="pengfei_joint")(pred, target, mask)
    pred_flow = complex2magflow_torch(pred)[1]
    target_flow = complex2magflow_torch(target)[1]
    expected = relerr_loss(pred_flow, target_flow, mask)
    torch.testing.assert_close(components["speed"], expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_official_profile_exposes_expected_components_and_ramp():
    pred = torch.randn(2, 4, 3, 1, 5, 6, 2, requires_grad=True)
    target = torch.randn_like(pred)
    loss_fn = JointEncodingLoss(profile="official_aligned")
    loss_zero_ramp, components = loss_fn(pred, target, None, ramp=0.0)
    expected_complex = loss_fn.weights["complex"] * components["complex"]
    torch.testing.assert_close(loss_zero_ramp, expected_complex)
    loss_full, components = loss_fn(pred, target, None, ramp=1.0)
    assert set(components) == {"complex", "circular", "velocity", "relerr", "angular"}
    assert loss_full >= loss_zero_ramp


def test_loss_profile_config_switches_defaults_without_changing_io_or_model():
    config = load_config(REPO_ROOT / "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_joint_venc_flow.json")
    spec = joint_encoding_spec(config)
    assert spec.enabled and spec.mode == "batch"
    official_flags = resolve_training_loss_flags(config, spec)
    assert official_flags["main_zy"] and not official_flags["phase"]

    delattr(config.phase3.loss, "use_ssim_zy") if hasattr(config.phase3.loss, "use_ssim_zy") else None
    delattr(config.phase3.loss, "use_phase") if hasattr(config.phase3.loss, "use_phase") else None
    config.phase3.loss.profile = "pengfei_joint"
    pengfei_flags = resolve_training_loss_flags(config, spec)
    assert not pengfei_flags["main_zy"] and pengfei_flags["phase"]


def test_relative_phase_is_invariant_to_common_global_phase():
    values = torch.randn(2, 4, 3, 4, 5, 2)
    phase = torch.tensor(1.234)
    rotation = torch.complex(torch.cos(phase), torch.sin(phase))
    rotated = torch.view_as_real(torch.view_as_complex(values.contiguous()) * rotation)
    torch.testing.assert_close(
        complex2magflow_torch(values)[1],
        complex2magflow_torch(rotated)[1],
        atol=2e-6,
        rtol=2e-6,
    )


def test_flow_ramp_and_three_slice_supervision_are_unchanged():
    assert flow_loss_ramp(5, enabled=True, start_epoch=5, end_epoch=20) == 0.0
    assert flow_loss_ramp(10, enabled=True, start_epoch=5, end_epoch=20) == 1.0 / 3.0
    slab = torch.randn(2, 3, 5, 4, 6, 7, 2)
    selected = select_training_center_time(slab, 5)
    assert selected.shape == (2, 3, 4, 6, 7, 2)
    torch.testing.assert_close(selected, slab[:, :, 2])


def test_checkpoint_metric_directions():
    best = BestFlowMetrics()
    assert set(best.update({"flow": 2.0, "relerr": 0.5, "angerr": 20.0, "ssim": 0.8}, 1)) == {
        "flow",
        "relerr",
        "angerr",
        "ssim",
    }
    assert set(best.update({"flow": 1.9, "relerr": 0.6, "angerr": 19.0, "ssim": 0.7}, 2)) == {
        "flow",
        "angerr",
    }


def run_directly():
    tests = sorted(
        (name, value) for name, value in globals().items() if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}", flush=True)
    print(f"COMPLETED {len(tests)} joint loss tests", flush=True)


if __name__ == "__main__":
    run_directly()
