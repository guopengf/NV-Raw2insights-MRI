from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from joint_encoding import (
    JointEncodingLoss,
    flatten_joint_model_batch,
    joint_encoding_spec,
    joint_group_batch_size,
    joint_windowed_input_x_slab,
    restore_joint_model_batch,
)
from inference import prepare_joint_encoding_output_for_save
from models.restormer.restormer import restormer_mri
from train import build_4dflow_aorta_manifests
from transforms import raw_4dflow_to_hybrid, raw_4dflow_to_joint_hybrid
from utils import load_config, load_shape_compatible_state_dict


def test_joint_batch_size_and_flatten_round_trip():
    args = SimpleNamespace(
        batch_size=8,
        phase3=SimpleNamespace(
            joint_encoding=SimpleNamespace(enabled=True, mode="batch", count=4, order=[0, 1, 2, 3])
        ),
    )
    assert joint_encoding_spec(args).mode == "batch"
    assert joint_group_batch_size(args) == 2

    grouped = torch.randn(2, 4, 3, 5)
    flat = flatten_joint_model_batch(grouped)
    assert flat.shape == (8, 3, 5)
    assert torch.equal(restore_joint_model_batch(flat, 4), grouped)


def test_joint_slab_window_uses_identical_indices_for_every_encoding():
    total_frames, total_slices, encoding_count = 3, 4, 4
    values = torch.empty(total_frames * total_slices, encoding_count, 1)
    for flat_idx in range(values.shape[0]):
        for encoding_idx in range(encoding_count):
            values[flat_idx, encoding_idx, 0] = 100 * flat_idx + encoding_idx

    gathered, indices = joint_windowed_input_x_slab(
        values,
        micro_b=[5],
        final_shape=[total_frames, total_slices, 1, 1, 1],
        num_frames=3,
        num_slices=3,
    )
    assert indices.tolist() == [[[0, 4, 8], [1, 5, 9], [2, 6, 10]]]
    assert gathered.shape == (1, 4, 3, 3, 1)
    for encoding_idx in range(encoding_count):
        assert torch.equal(gathered[0, encoding_idx, ..., 0] - encoding_idx, indices[0] * 100)


def test_joint_hybrid_matches_stacked_single_encoding_conversion():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(4, 3, 2, 2, 3, 5)) + 1j * rng.normal(size=(4, 3, 2, 2, 3, 5))
    joint = raw_4dflow_to_joint_hybrid(raw)
    expected = np.stack([raw_4dflow_to_hybrid(raw[e : e + 1]).reshape(3, 5, 2, 2, 3, 2) for e in range(4)], axis=2)
    assert joint.shape == expected.shape
    np.testing.assert_allclose(joint, expected, rtol=0, atol=0)


def test_joint_loss_is_zero_for_identity_and_has_finite_gradients():
    torch.manual_seed(3)
    target = torch.randn(2, 4, 3, 1, 6, 8, 2)
    pred = target.clone().requires_grad_(True)
    mask = torch.ones(2, 3, 6, 8)
    loss_fn = JointEncodingLoss(encoding_count=4)

    identity_loss, components = loss_fn(pred, target, mask)
    assert identity_loss.abs().item() < 1e-6
    assert all(value.abs().item() < 1e-6 for value in components.values())

    perturbed = target.clone()
    perturbed[:, 2, ..., 0] += 0.2
    perturbed.requires_grad_(True)
    loss, components = loss_fn(perturbed, target, mask)
    assert loss.item() > 0
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert perturbed.grad is not None
    assert torch.isfinite(perturbed.grad).all()


def test_grouped_manifest_has_one_json_per_case_acceleration(tmp_path):
    case = tmp_path / "Center001" / "Scanner" / "P001"
    case.mkdir(parents=True)
    for name in ("kdata_full.mat", "kdata_ktGaussian10.mat", "usmask_ktGaussian10.mat"):
        (case / name).touch()

    manifests = build_4dflow_aorta_manifests(
        [tmp_path],
        tmp_path.parent / f"{tmp_path.name}_manifests",
        accelerations=[10],
        encodings=[0, 1, 2, 3],
        joint_encodings=True,
    )
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text())
    assert payload["joint_encodings"] is True
    assert payload["encoding_indices"] == [0, 1, 2, 3]
    assert "encoding_idx" not in payload


def _small_joint_config(mode: str):
    config = load_config(
        REPO_ROOT
        / "configs"
        / f"nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_{mode}_pg.json"
    )
    config.num_frames = 3
    config.channels = [8, 16]
    config.num_blocks = [1, 1]
    config.num_heads = [1, 2]
    config.num_refinement = 1
    config.mlp_ratio = 2
    config.drop_path = 0.0
    config.time_cond = False
    config.label_cond = False
    config.use_dc_weight_map = False
    config.pretrained_csm = None
    config.pretrained_recon = None
    config.phase3.flowvn_mixer.features = 2
    config.phase3.flowvn_mixer.num_knots = 5
    return config


def test_batch_and_channel_models_preserve_shape_and_channel_bootstrap_equivalence():
    torch.manual_seed(17)
    device = torch.device(os.environ.get("JOINT_TEST_DEVICE", "cpu"))
    batch_model = restormer_mri(_small_joint_config("batch")).to(device).eval()
    channel_model = restormer_mri(_small_joint_config("channel")).to(device).eval()
    loaded, unchanged, skipped, unexpected, inflated = load_shape_compatible_state_dict(
        channel_model, batch_model.state_dict()
    )
    assert loaded
    assert not skipped
    assert not unexpected
    assert len(inflated) == 2
    assert not [key for key in unchanged if key not in inflated]
    assert channel_model.recon_model.embed_conv.in_channels == 4 * batch_model.recon_model.embed_conv.in_channels
    assert channel_model.recon_model.output.out_channels == 4 * batch_model.recon_model.output.out_channels

    one_encoding = torch.randn(1, 3, 3, 2, 6, 8, 2, device=device)
    x = one_encoding.repeat(4, 1, 1, 1, 1, 1, 1)
    sensitivity = torch.randn_like(one_encoding).repeat(4, 1, 1, 1, 1, 1, 1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    kwargs = dict(
        mask_type="ktGaussian",
        acc_factor=10,
        acq_type="Flow4d",
        sensitivity_maps=sensitivity,
    )
    with torch.no_grad():
        batch_output = batch_model(x, x.clone(), mask, **kwargs)[0]
        channel_output = channel_model(x, x.clone(), mask, **kwargs)[0]
    assert batch_output.shape == x.shape
    assert channel_output.shape == x.shape
    difference = (channel_output - batch_output).abs()
    print(
        f"equivalence device={device} max_abs={difference.max().item():.8g} "
        f"mean_abs={difference.mean().item():.8g}",
        flush=True,
    )
    if device.type == "cuda":
        assert difference.max().item() < 5e-3
        assert difference.mean().item() < 5e-4
    else:
        assert torch.allclose(channel_output, batch_output, atol=1e-5, rtol=1e-5)

    channel_model.train()
    channel_model.zero_grad(set_to_none=True)
    train_output = channel_model(x, x.clone(), mask, **kwargs)[0]
    train_output.square().mean().backward()
    assert channel_model.recon_model.embed_conv.weight.grad is not None
    assert channel_model.recon_model.output.weight.grad is not None
    assert torch.isfinite(channel_model.recon_model.embed_conv.weight.grad).all()
    assert torch.isfinite(channel_model.recon_model.output.weight.grad).all()


def test_joint_configs_keep_fixed_training_cardinality():
    for mode in ("batch", "channel"):
        config_path = (
            REPO_ROOT
            / "configs"
            / f"nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_{mode}_pg.json"
        )
        payload = json.loads(config_path.read_text())
        assert payload["batch_size"] == 8
        assert payload["num_samples_per_case"] == 8
        assert payload["phase3"]["joint_encoding"]["mode"] == mode
        assert payload["exp"] != "small_ft_4dflow_encbatch_phase4_3d_flowvn_multiplane"


def test_joint_inference_export_preserves_complex_encoding_volume():
    output = np.arange(2 * 3 * 1 * 4 * 5 * 2, dtype=np.float32).reshape(2, 3, 1, 4, 5, 2)
    args = SimpleNamespace(save_coil_combined_output=True)
    prepared = prepare_joint_encoding_output_for_save(output, None, args)
    expected = np.transpose(output[:, :, 0], (3, 2, 1, 0, 4))
    assert prepared.shape == (5, 4, 3, 2, 2)
    np.testing.assert_array_equal(prepared, expected)


def run_directly():
    tests = [
        test_joint_batch_size_and_flatten_round_trip,
        test_joint_slab_window_uses_identical_indices_for_every_encoding,
        test_joint_hybrid_matches_stacked_single_encoding_conversion,
        test_joint_loss_is_zero_for_identity_and_has_finite_gradients,
        test_batch_and_channel_models_preserve_shape_and_channel_bootstrap_equivalence,
        test_joint_configs_keep_fixed_training_cardinality,
        test_joint_inference_export_preserves_complex_encoding_volume,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    with tempfile.TemporaryDirectory() as temporary:
        test_grouped_manifest_has_one_json_per_case_acceleration(Path(temporary))
    print("PASS test_grouped_manifest_has_one_json_per_case_acceleration")
    print(f"COMPLETED {len(tests) + 1} joint-encoding tests")


if __name__ == "__main__":
    run_directly()
