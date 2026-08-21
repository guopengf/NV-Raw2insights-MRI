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
from models.flowvn_mixer import FlowVNMultiPlaneMixer
from models.restormer.restormer import restormer_mri
from readers import CMRxReconReader
from train import (
    build_4dflow_aorta_manifests,
    group_4dflow_manifests_by_target,
    joint_encoding_loss_enabled,
    partition_4dflow_target_groups,
    resolve_training_loss_flags,
)
from transforms import raw_4dflow_to_hybrid, raw_4dflow_to_joint_hybrid
from train_utils import get_optimizer
from utils import TargetGroupedSampler, load_config, load_shape_compatible_state_dict


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
    identity_loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    perturbed = target.clone()
    perturbed[:, 2, ..., 0] += 0.2
    perturbed.requires_grad_(True)
    loss, components = loss_fn(perturbed, target, mask)
    assert loss.item() > 0
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert perturbed.grad is not None
    assert torch.isfinite(perturbed.grad).all()


def test_joint_loss_empty_mask_is_zero_with_finite_gradients():
    torch.manual_seed(11)
    target = torch.randn(2, 4, 3, 1, 6, 8, 2)
    pred = torch.randn_like(target, requires_grad=True)
    mask = torch.zeros(2, 3, 6, 8)

    loss, components = JointEncodingLoss(encoding_count=4)(pred, target, mask)

    assert loss.abs().item() < 1e-6
    assert all(value.abs().item() < 1e-6 for value in components.values())
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == 0


def test_joint_loss_zero_signal_has_finite_gradients():
    target = torch.zeros(2, 4, 3, 1, 6, 8, 2)
    pred = torch.zeros_like(target, requires_grad=True)
    mask = torch.ones(2, 3, 6, 8)

    loss, components = JointEncodingLoss(encoding_count=4)(pred, target, mask)

    assert loss.abs().item() < 1e-6
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_joint_configs_keep_global_flowvn_complex_loss_enabled():
    config_names = (
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json",
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
    )
    for config_name in config_names:
        config = load_config(REPO_ROOT / "configs" / config_name)
        spec = joint_encoding_spec(config)
        flags = resolve_training_loss_flags(config, spec)

        assert spec.enabled is True
        assert config.phase3.loss.phase.method == "flowvn_complex_l1"
        assert config.phase3.loss.phase.weight > 0
        assert flags == {"main_zy": False, "phase": True, "vascular": False}


def test_joint_loss_gate_is_backward_compatible_and_independent():
    spec = SimpleNamespace(enabled=True)
    legacy_args = SimpleNamespace(phase3=SimpleNamespace(loss=SimpleNamespace(joint=SimpleNamespace())))
    disabled_args = SimpleNamespace(
        phase3=SimpleNamespace(loss=SimpleNamespace(joint=SimpleNamespace(enabled=False)))
    )

    assert joint_encoding_loss_enabled(legacy_args, spec) is True
    assert joint_encoding_loss_enabled(disabled_args, spec) is False
    assert joint_encoding_loss_enabled(legacy_args, SimpleNamespace(enabled=False)) is False


def test_joint_configs_enable_online_augmentation():
    config_names = (
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json",
        "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
    )
    for config_name in config_names:
        config = load_config(REPO_ROOT / "configs" / config_name)
        spec = joint_encoding_spec(config)
        augmentation = config.four_dflow_augmentation

        assert spec.enabled is True
        assert config.data_aug is True
        assert augmentation.enabled is True
        assert augmentation.flip.prob == 0.25
        assert list(augmentation.flip.axes) == ["z", "y"]
        assert augmentation.shift.prob == 0.25
        assert list(augmentation.shift.max_pixels) == [3, 8]
        assert augmentation.contrast.prob == 0.2
        assert list(augmentation.contrast.gamma) == [0.8, 1.2]
        assert resolve_training_loss_flags(config, spec) == {
            "main_zy": False,
            "phase": True,
            "vascular": False,
        }
        assert config.phase3.loss.phase.method == "flowvn_complex_l1"
        assert config.phase3.loss.phase.weight > 0


def test_non_joint_loss_flags_remain_config_driven():
    args = SimpleNamespace(
        phase3=SimpleNamespace(
            loss=SimpleNamespace(use_ssim_zy=True, use_phase=True, use_vascular=True),
        ),
    )
    spec = SimpleNamespace(enabled=False)

    assert resolve_training_loss_flags(args, spec) == {
        "main_zy": True,
        "phase": True,
        "vascular": True,
    }


def test_flowvn_conv3d_kernels_use_adam_with_muon_optimizer():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flowvn_mixer = FlowVNMultiPlaneMixer(features=2, num_knots=5)
            self.hidden = torch.nn.Linear(4, 4, bias=False)

    model = Model()
    args = SimpleNamespace(muon=True, lookahead=False, lr=1e-3, weight_decay=0.0)
    optimizer = get_optimizer(args, model)
    muon_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["use_muon"]
        for parameter in group["params"]
    }
    adam_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if not group["use_muon"]
        for parameter in group["params"]
    }
    flowvn_kernel_ids = {
        id(regularizer.weight)
        for regularizer in model.flowvn_mixer.regularizers.values()
    }

    assert flowvn_kernel_ids <= adam_ids
    assert flowvn_kernel_ids.isdisjoint(muon_ids)
    assert id(model.hidden.weight) in muon_ids


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


def test_target_group_partition_keeps_accelerations_adjacent_and_balanced(tmp_path):
    manifests = []
    for patient_index in range(3):
        target = tmp_path / f"patient_{patient_index}" / "kdata_full.mat"
        target.parent.mkdir()
        for acceleration in (10, 20, 30, 40, 50):
            manifest = tmp_path / f"patient_{patient_index}_acc{acceleration}.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kspace": str(target.parent / f"kdata_ktGaussian{acceleration}.mat"),
                        "target_kspace": str(target),
                    }
                )
            )
            manifests.append(manifest)

    groups = group_4dflow_manifests_by_target(manifests)
    assert [len(group) for group in groups] == [5, 5, 5]
    rank0 = partition_4dflow_target_groups(groups, num_partitions=2, rank=0, seed=7)
    rank1 = partition_4dflow_target_groups(groups, num_partitions=2, rank=1, seed=7)
    assert sum(map(len, rank0)) == sum(map(len, rank1)) == 8
    assert all(
        len({json.loads(path.read_text())["target_kspace"] for path in group}) == 1
        for group in rank0 + rank1
    )
    assert sum(len(group) == 5 for group in rank0 + rank1) >= 2


def test_target_grouped_sampler_shuffles_groups_without_splitting_them():
    sampler = TargetGroupedSampler([5, 5, 5], seed=3, shuffle=True)
    sampler.set_epoch(0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))
    sampler.set_epoch(0)
    assert list(iter(sampler)) == epoch0
    assert epoch1 != epoch0
    for indices in (epoch0, epoch1):
        assert sorted(indices) == list(range(15))
        chunks = [indices[offset : offset + 5] for offset in range(0, 15, 5)]
        assert all(chunk == list(range(chunk[0], chunk[0] + 5)) for chunk in chunks)


def test_reader_target_lru_reuses_one_full_array_per_worker():
    args = SimpleNamespace(reader_target_cache_entries=1, reader_coilmap_cache_entries=1)
    reader = CMRxReconReader(fixed_mask_types=["fixed"], args=args)
    calls = []

    def fake_read(path, preferred_keys=(), selection=None, return_shape=False):
        calls.append(str(path))
        value = np.full((4, 2), len(calls), dtype=np.float32)
        return (value, value.shape) if return_shape else value

    reader.read_first_mat_array = fake_read
    first, first_shape, first_hit = reader.read_cached_mat_array("target", "target_a.mat")
    second, second_shape, second_hit = reader.read_cached_mat_array("target", "target_a.mat")
    assert not first_hit and second_hit
    assert first_shape == second_shape == (4, 2)
    assert first is second
    assert calls == ["target_a.mat"]

    reader.read_cached_mat_array("target", "target_b.mat")
    _, _, reloaded_hit = reader.read_cached_mat_array("target", "target_a.mat")
    assert not reloaded_hit
    assert calls == ["target_a.mat", "target_b.mat", "target_a.mat"]


def test_non_joint_reader_slices_one_encoding_without_target_cache(tmp_path):
    manifest = tmp_path / "encoding_2.json"
    manifest.write_text(
        json.dumps(
            {
                "kspace": "undersampled.mat",
                "target_kspace": "target.mat",
                "mask": [],
                "is_4dflow": True,
                "joint_encodings": False,
                "encoding_idx": 2,
            }
        )
    )
    args = SimpleNamespace(
        reader_target_cache_entries=1,
        reader_coilmap_cache_entries=1,
        performance_timing=SimpleNamespace(worker_timing_enabled=True),
    )
    reader = CMRxReconReader(fixed_mask_types=["fixed"], args=args)
    calls = []

    def fake_read(path, preferred_keys=(), selection=None, return_shape=False):
        calls.append((str(path), selection))
        value = np.zeros((1, 2), dtype=np.complex64)
        return (value, (4, 2)) if return_shape else value

    def fail_if_cached(*args, **kwargs):
        raise AssertionError("The non-joint reader must not load/cache the full four-encoding target")

    reader.read_first_mat_array = fake_read
    reader.read_cached_mat_array = fail_if_cached
    sample = reader.read(manifest)
    assert [path for path, _ in calls] == ["undersampled.mat", "target.mat"]
    assert all(selection[0] == slice(2, 3) and selection[1] is Ellipsis for _, selection in calls)
    assert sample["num_encodings"] == 4
    assert sample["worker_timing"]["target_cache_hit"] == 0.0


def _small_joint_config(mode: str):
    config_names = {
        "batch": "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json",
        "channel": "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
    }
    config = load_config(
        REPO_ROOT
        / "configs"
        / config_names[mode]
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
    config_names = {
        "batch": "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_pg.json",
        "channel": "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
    }
    expected_workers = {"batch": (4, 0), "channel": (8, 4)}
    for mode, config_name in config_names.items():
        config_path = (
            REPO_ROOT
            / "configs"
            / config_name
        )
        payload = json.loads(config_path.read_text())
        assert payload["batch_size"] == 8
        assert payload["num_samples_per_case"] == 8
        train_workers, val_workers = expected_workers[mode]
        assert payload["train_num_workers"] == train_workers
        assert payload["val_num_workers"] == val_workers
        assert payload["group_accelerations_by_target"] is True
        assert payload["reader_target_cache_entries"] == 1
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
        test_joint_loss_empty_mask_is_zero_with_finite_gradients,
        test_joint_loss_zero_signal_has_finite_gradients,
        test_joint_configs_keep_global_flowvn_complex_loss_enabled,
        test_joint_loss_gate_is_backward_compatible_and_independent,
        test_augmented_joint_configs_disable_only_joint_loss,
        test_non_joint_loss_flags_remain_config_driven,
        test_flowvn_conv3d_kernels_use_adam_with_muon_optimizer,
        test_target_grouped_sampler_shuffles_groups_without_splitting_them,
        test_reader_target_lru_reuses_one_full_array_per_worker,
        test_batch_and_channel_models_preserve_shape_and_channel_bootstrap_equivalence,
        test_joint_configs_keep_fixed_training_cardinality,
        test_joint_inference_export_preserves_complex_encoding_volume,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    with tempfile.TemporaryDirectory() as temporary:
        test_grouped_manifest_has_one_json_per_case_acceleration(Path(temporary))
    with tempfile.TemporaryDirectory() as temporary:
        test_target_group_partition_keeps_accelerations_adjacent_and_balanced(Path(temporary))
    with tempfile.TemporaryDirectory() as temporary:
        test_non_joint_reader_slices_one_encoding_without_target_cache(Path(temporary))
    print("PASS test_grouped_manifest_has_one_json_per_case_acceleration")
    print("PASS test_target_group_partition_keeps_accelerations_adjacent_and_balanced")
    print("PASS test_non_joint_reader_slices_one_encoding_without_target_cache")
    print(f"COMPLETED {len(tests) + 3} joint-encoding tests")


if __name__ == "__main__":
    run_directly()
