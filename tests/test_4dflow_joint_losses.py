from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mra_utils import load_roi_loss_mask, roi_loss_mask_needed, vaa_prior_needed
from mri_data.flow_losses import (
    BestFlowMetrics,
    EMALossNormalizer,
    angular_cosine_loss,
    circular_phase_loss,
    complex2magflow_torch,
    complex_roi_l1_loss,
    flatten_joint_venc,
    flow_loss_ramp,
    regroup_joint_venc,
    relerr_loss,
    select_training_center_time,
    velocity_component_loss,
)
from mri_data.four_dflow_data import (
    JointVencDataset,
    Paired4DFlowRoll,
    gather_joint_window,
    group_joint_venc_manifests,
)
from utils import load_config, windowed_input_x_slab


OFFICIAL_FLOW = Path("/mnt/nas/nas3/openData/rawdata/4dFlow/ChallengeData_GT/EvaluationCode/utils_flow.py")
OFFICIAL_METRICS = OFFICIAL_FLOW.with_name("utils_metrics.py")


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
                    field="segmask",
                    keys=["segmask"],
                    axis_order=axis_order,
                    binary=True,
                    threshold=0.5,
                )
            ),
        )
    )


def _write_manifests(directory: Path, patient="P001", acceleration=10):
    paths = []
    for enc in range(4):
        path = directory / f"{patient}__ktGaussian{acceleration}__enc{enc}.json"
        path.write_text(
            json.dumps(
                {
                    "kspace": str(directory / patient / f"kdata_ktGaussian{acceleration}.mat"),
                    "target_kspace": str(directory / patient / "kdata_full.mat"),
                    "mask_type": f"ktGaussian{acceleration}",
                    "encoding_idx": enc,
                }
            )
        )
        paths.append(path)
    return paths


def test_roi_mask_loading_is_independent_from_vaa():
    args = _roi_args()
    assert roi_loss_mask_needed(args)
    assert not vaa_prior_needed(args)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "segmask.mat"
        scipy.io.savemat(path, {"segmask": np.ones((2, 3, 4), dtype=np.float32)})
        mask = load_roi_loss_mask(path, args)
    assert mask.shape == (4, 2, 3)
    assert np.all(mask == 1)


def test_segmentation_axis_conversion_is_xzy():
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    args = _roi_args("zyx")
    args.phase3.loss.roi.binary = False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "segmask.mat"
        scipy.io.savemat(path, {"segmask": source})
        actual = load_roi_loss_mask(path, args)
    expected = np.transpose(np.clip(source, 0, 1), (2, 0, 1))
    np.testing.assert_array_equal(actual, expected)


def test_old_per_encoding_mode_remains_disabled_by_default():
    config = load_config(REPO_ROOT / "configs/nv_raw2insights_mri_base_4dflow_3d_flowvn_multiplane.json")
    joint = getattr(config.phase3.loss, "joint_venc", None)
    assert not bool(getattr(joint, "enabled", False))


def test_joint_grouping_is_ordered_and_same_case_acceleration():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_manifests(Path(tmp))
        groups = group_joint_venc_manifests(reversed(paths), encodings=[0, 1, 2, 3])
        encodings = [json.loads(Path(item["kspace"]).read_text())["encoding_idx"] for item in groups[0]]
        assert len(groups) == 1
        assert encodings == [0, 1, 2, 3]


def test_joint_dataset_stacks_complete_encoding_group():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_manifests(Path(tmp))

        def fake_transform(item):
            manifest = json.loads(Path(item["kspace"]).read_text())
            enc = manifest["encoding_idx"]
            tensor = torch.full((6, 1, 2, 3, 2), float(enc))
            return {
                "kspace_ifft": tensor,
                "kspace_masked_ifft": tensor,
                "mask": torch.ones(6, 1, 2, 3, 1),
                "mean": torch.zeros(6, 1, 1, 1, 2),
                "std": torch.ones(6, 1, 1, 1, 1),
                "roi_mask": torch.ones(3, 2, 3),
                "mask_type": "ktGaussian10",
                "acc_factor": 10.0,
                "acquisition": "Flow4d",
                "temporal_shuffle": torch.arange(2),
                "kspace_meta_dict": {
                    "shape": np.array([2, 3, 1, 2, 3], dtype=np.int32),
                    "filename": str(item["kspace"]),
                },
            }

        sample = JointVencDataset(paths, fake_transform)[0]
        assert sample["kspace_ifft"].shape == (4, 6, 1, 2, 3, 2)
        assert sample["encoding_indices"].tolist() == [0, 1, 2, 3]


def test_all_encodings_share_identical_slab_time_indices():
    encodings, frames, slices = 4, 5, 7
    source = torch.arange(encodings * frames * slices).reshape(encodings, frames * slices, 1)
    _, indices = windowed_input_x_slab(source[0], [8, 17], [frames, slices, 1, 1, 1], 5, 3)
    gathered = gather_joint_window(source, indices)
    for enc in range(encodings):
        np.testing.assert_array_equal(
            (gathered[:, enc, ..., 0] - enc * frames * slices).numpy(),
            indices.numpy(),
        )


def test_joint_flatten_model_regroup_shapes():
    x = torch.randn(2, 4, 3, 5, 2, 6, 7, 2)
    flat, shape = flatten_joint_venc(x)
    model = torch.nn.Identity()
    restored = regroup_joint_venc(model(flat), shape)
    assert flat.shape == (8, 3, 5, 2, 6, 7, 2)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_relative_phase_is_invariant_to_common_global_phase():
    x = torch.randn(2, 4, 3, 4, 5, 2)
    phase = torch.tensor(1.234)
    rotation = torch.stack((torch.cos(phase), torch.sin(phase)))
    z = torch.view_as_complex(x.contiguous()) * torch.view_as_complex(rotation.reshape(1, 2))
    rotated = torch.view_as_real(z)
    flow = complex2magflow_torch(x)[1]
    rotated_flow = complex2magflow_torch(rotated)[1]
    torch.testing.assert_close(flow, rotated_flow, atol=2e-6, rtol=2e-6)


def test_circular_phase_loss_wraps_across_pi_boundary():
    eps = 1e-3
    reference = torch.ones(1, 1, 1, 1, dtype=torch.complex64)
    pred_encoded = torch.polar(torch.ones(1), torch.tensor([-torch.pi + eps])).reshape(1, 1, 1, 1)
    target_encoded = torch.polar(torch.ones(1), torch.tensor([torch.pi - eps])).reshape(1, 1, 1, 1)
    pred = torch.view_as_real(torch.stack((reference, pred_encoded), dim=1))
    target = torch.view_as_real(torch.stack((reference, target_encoded), dim=1))
    loss = circular_phase_loss(pred, target, torch.ones(1, 1, 1, 1))
    assert loss < 1e-5


def test_torch_complex2magflow_matches_official_numpy():
    spec = importlib.util.spec_from_file_location("official_utils_flow", OFFICIAL_FLOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(7)
    x = (rng.normal(size=(4, 2, 3, 4, 5)) + 1j * rng.normal(size=(4, 2, 3, 4, 5))).astype(np.complex64)
    official_mag, official_flow = module.complex2magflow(x)
    torch_mag, torch_flow = complex2magflow_torch(torch.view_as_real(torch.from_numpy(x)), encoding_dim=0)
    np.testing.assert_allclose(torch_mag.numpy(), official_mag, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(torch_flow.numpy(), official_flow, rtol=1e-6, atol=1e-6)


def test_differentiable_relerr_matches_official_formula():
    sys.path.insert(0, str(OFFICIAL_METRICS.parent))
    spec = importlib.util.spec_from_file_location("official_utils_metrics", OFFICIAL_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(9)
    target = rng.normal(size=(3, 2, 4, 5, 6)).astype(np.float32)
    pred = rng.normal(size=target.shape).astype(np.float32)
    mask = (rng.random(size=(4, 5, 6)) > 0.2).astype(np.float32)
    expected = module.RelErr(pred, target, mask)
    actual = relerr_loss(
        torch.from_numpy(pred).unsqueeze(0),
        torch.from_numpy(target).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0).expand(1, 2, -1, -1, -1),
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
    tiny_target = target * 1e-8
    tiny_pred = pred * 1e-8
    tiny_expected = module.RelErr(tiny_pred, tiny_target, mask)
    tiny_actual = relerr_loss(
        torch.from_numpy(tiny_pred).unsqueeze(0),
        torch.from_numpy(tiny_target).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0).expand(1, 2, -1, -1, -1),
    )
    np.testing.assert_allclose(tiny_actual.numpy(), tiny_expected, rtol=1e-6, atol=1e-8)


def test_all_flow_loss_gradients_are_finite():
    pred = torch.randn(2, 4, 3, 4, 5, 2, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.ones(2, 3, 4, 5)
    pred_flow = complex2magflow_torch(pred)[1]
    target_flow = complex2magflow_torch(target)[1]
    losses = (
        complex_roi_l1_loss(pred, target, mask),
        circular_phase_loss(pred, target, mask),
        velocity_component_loss(pred_flow, target_flow, mask),
        relerr_loss(pred_flow, target_flow, mask),
        angular_cosine_loss(pred_flow, target_flow, mask),
    )
    for loss in losses:
        grad = torch.autograd.grad(loss, pred, retain_graph=True)[0]
        assert torch.isfinite(loss)
        assert torch.isfinite(grad).all()


def test_empty_roi_does_not_produce_nan():
    pred = torch.randn(1, 4, 3, 2, 2, 2, requires_grad=True)
    target = torch.randn_like(pred)
    empty = torch.zeros(1, 3, 2, 2)
    pred_flow = complex2magflow_torch(pred)[1]
    target_flow = complex2magflow_torch(target)[1]
    losses = [
        complex_roi_l1_loss(pred, target, empty),
        circular_phase_loss(pred, target, empty),
        velocity_component_loss(pred_flow, target_flow, empty),
        relerr_loss(pred_flow, target_flow, empty),
        angular_cosine_loss(pred_flow, target_flow, empty),
    ]
    assert all(torch.isfinite(loss) and loss == 0 for loss in losses)


def test_zero_velocity_angular_loss_does_not_produce_nan():
    pred = torch.zeros(1, 3, 2, 2, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    loss = angular_cosine_loss(pred, target, torch.ones(1, 2, 2, 2))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()

    exact = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    exact_relerr = relerr_loss(exact, exact.detach(), torch.ones(1, 2, 2, 2))
    exact_relerr.backward()
    assert exact_relerr == 0
    assert torch.isfinite(exact.grad).all()


def test_validation_grouping_yields_one_result_per_complete_case():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _write_manifests(root, patient="P001") + _write_manifests(root, patient="P002")
        groups = group_joint_venc_manifests(paths)
    results = [{"relerr": index} for index, _group in enumerate(groups)]
    assert len(groups) == len(results) == 2


def test_checkpoint_metric_directions():
    best = BestFlowMetrics()
    assert set(best.update({"flow": 2.0, "relerr": 0.5, "angerr": 20.0, "ssim": 0.8}, 1)) == {
        "flow", "relerr", "angerr", "ssim"
    }
    improved = best.update({"flow": 1.9, "relerr": 0.6, "angerr": 19.0, "ssim": 0.7}, 2)
    assert set(improved) == {"flow", "angerr"}


def test_flow_ramp_and_optional_ema_normalization():
    assert flow_loss_ramp(5, enabled=True, start_epoch=5, end_epoch=20) == 0.0
    assert flow_loss_ramp(10, enabled=True, start_epoch=5, end_epoch=20) == 1.0 / 3.0
    assert flow_loss_ramp(20, enabled=True, start_epoch=5, end_epoch=20) == 1.0
    value = torch.tensor(2.0, requires_grad=True)
    normalizer = EMALossNormalizer(enabled=True)
    normalized = normalizer.normalize("velocity", value)
    normalized.backward()
    assert value.grad == 0.5
    assert not normalizer.ema["velocity"].requires_grad


def test_training_center_time_keeps_all_three_slices():
    slab = torch.randn(2, 3, 5, 4, 6, 7, 2)
    selected = select_training_center_time(slab, 5)
    assert selected.shape == (2, 3, 4, 6, 7, 2)
    torch.testing.assert_close(selected, slab[:, :, 2])


def test_paired_roll_uses_one_geometry_and_leaves_sampling_support_unchanged():
    sample = {
        "kspace_ifft": torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2),
        "kspace_masked_ifft": torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2),
        "sensitivity_maps": torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2),
        "mask": torch.ones(4, 1, 2, 3, 1),
        "roi_mask": torch.arange(5 * 2 * 3).reshape(5, 2, 3),
    }
    random.seed(3)
    transformed = Paired4DFlowRoll(enabled=True, probability=1.0, max_shift_zy=(1, 1))(sample)
    shift = tuple(int(value) for value in transformed["paired_roll_zy"])
    assert torch.equal(transformed["kspace_ifft"], torch.roll(sample["kspace_ifft"], shift, (-3, -2)))
    assert torch.equal(transformed["kspace_masked_ifft"], transformed["kspace_ifft"])
    assert torch.equal(transformed["mask"], sample["mask"])
