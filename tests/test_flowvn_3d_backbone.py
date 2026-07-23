from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from models.flowvn_mixer import FLOWVN_BRANCHES, FlowVNMultiPlaneMixer
from models.restormer.restormer import Restormer, restormer_mri
from models.vaa import VascularAttentionAdapter
from utils import load_config, load_shape_compatible_state_dict


def test_flowvn_mixer_zero_scale_is_exact_bypass():
    mixer = FlowVNMultiPlaneMixer(
        in_channels=1,
        features=2,
        kernel_size=3,
        num_knots=5,
        branches=FLOWVN_BRANCHES,
        scale_init=0.0,
    )
    x = torch.randn(1, 3, 5, 1, 6, 8, 2)
    update = mixer(x, acceleration=20)

    assert update.shape == x.shape
    assert torch.count_nonzero(update).item() == 0

    probe = torch.randn_like(update)
    (update * probe).sum().backward()
    assert mixer.scale.grad is not None
    assert torch.isfinite(mixer.scale.grad)
    assert mixer.scale.grad.abs().item() > 0


def test_each_flowvn_plane_restores_the_input_axis_order():
    x = torch.randn(2, 3, 5, 1, 4, 6, 2)
    for branch in FLOWVN_BRANCHES:
        mixer = FlowVNMultiPlaneMixer(
            in_channels=1,
            features=2,
            kernel_size=3,
            num_knots=5,
            branches=[branch],
            scale_init=1.0,
            acceleration_modulation=False,
        )
        update = mixer(x)
        assert update.shape == x.shape
        assert torch.isfinite(update).all()


def test_spatial_3d_restormer_preserves_raw_x_and_odd_zy_shape():
    phase3 = SimpleNamespace(enable_vaa=False, recon_mode="slab", num_slices=3)
    model = Restormer(
        in_channel=10,
        out_channel=10,
        num_blocks=[1, 1],
        num_heads=[1, 2],
        channels=[8, 16],
        num_refinement=1,
        expansion_factor=2,
        norm_type="ln",
        phase3=phase3,
        spatial_dims=3,
    )
    x = torch.randn(1, 10, 3, 7, 9)
    output, cascade_skips = model(x)

    assert output.shape == x.shape
    assert len(cascade_skips) == 1
    assert cascade_skips[0].shape[2] == x.shape[2]
    output.square().mean().backward()
    assert model.embed_conv.weight.grad is not None
    assert torch.isfinite(model.embed_conv.weight.grad).all()


def test_existing_2d_restormer_path_remains_available():
    phase3 = SimpleNamespace(enable_vaa=False, recon_mode="slab", num_slices=3)
    model = Restormer(
        in_channel=30,
        out_channel=30,
        num_blocks=[1, 1],
        num_heads=[1, 2],
        channels=[8, 16],
        num_refinement=1,
        expansion_factor=2,
        norm_type="ln",
        phase3=phase3,
        spatial_dims=2,
    )
    x = torch.randn(1, 30, 7, 9)
    output, _ = model(x)
    assert output.shape == x.shape


def test_3d_vaa_accepts_raw_x_mask_as_depth():
    adapter = VascularAttentionAdapter(
        channels=8,
        reduction=2,
        gamma_init=0.0,
        attention="qkv",
        num_heads=2,
        attention_stride=2,
        spatial_dims=3,
    )
    feature = torch.randn(1, 8, 3, 4, 6)
    mask = torch.randint(0, 2, (1, 3, 4, 6), dtype=torch.float32)
    output = adapter(feature, mask)

    assert output.shape == feature.shape
    assert torch.equal(output, feature)


def test_conv2d_checkpoint_weight_is_center_inflated_into_conv3d():
    module = nn.Conv3d(2, 4, kernel_size=3, padding=1, bias=False)
    old_weight = torch.randn(4, 2, 3, 3)
    loaded, _unchanged, skipped, unexpected, inflated = load_shape_compatible_state_dict(
        module,
        {"weight": old_weight},
    )

    assert loaded == ["weight"]
    assert inflated == ["weight"]
    assert not skipped
    assert not unexpected
    assert torch.equal(module.weight[:, :, 1], old_weight)
    assert torch.count_nonzero(module.weight[:, :, 0]).item() == 0
    assert torch.count_nonzero(module.weight[:, :, 2]).item() == 0

    x = torch.randn(1, 2, 3, 6, 8)
    expected = torch.stack(
        [torch.nn.functional.conv2d(x[:, :, depth], old_weight, padding=1) for depth in range(x.shape[2])],
        dim=2,
    )
    actual = module(x)
    assert torch.allclose(actual, expected)


def test_3d_flowvn_config_validates():
    config = load_config(REPO_ROOT / "configs" / "nv_raw2insights_mri_base_4dflow_3d_flowvn_multiplane.json")
    assert config.phase3.backbone.spatial_dims == 3
    assert config.phase3.backbone.downsample_axes == "zy"
    assert config.phase3.flowvn_mixer.branches == list(FLOWVN_BRANCHES)


def test_restormer_mri_3d_flowvn_forward_keeps_slab_contract():
    config = load_config(REPO_ROOT / "configs" / "nv_raw2insights_mri_base_4dflow_3d_flowvn_multiplane.json")
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
    config.phase3.flowvn_mixer.features = 2
    config.phase3.flowvn_mixer.num_knots = 5

    model = restormer_mri(config)
    x = torch.randn(1, 3, 3, 2, 6, 8, 2)
    sensitivity = torch.randn_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)

    output, cascade_skips, returned_sensitivity = model(
        x,
        x.clone(),
        mask,
        mask_type="ktGaussian",
        acc_factor=10,
        acq_type="Flow4d",
        sensitivity_maps=sensitivity,
    )

    assert output.shape == x.shape
    assert len(cascade_skips) == 1
    assert cascade_skips[0].ndim == 5
    assert returned_sensitivity.shape[0] == 1 * 3 * 3


def test_restormer_mri_3d_mask_qkv_vaa_accepts_slab_prior():
    config = load_config(REPO_ROOT / "configs" / "nv_raw2insights_mri_base_4dflow_mask_vaa_slab.json")
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
    config.phase3.backbone = SimpleNamespace(spatial_dims=3, downsample_axes="zy")
    config.phase3.vaa.num_heads = 2
    config.phase3.vaa.attention_stride = 2

    model = restormer_mri(config)
    x = torch.randn(1, 3, 3, 2, 6, 8, 2)
    sensitivity = torch.randn_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    vessel_prior = torch.randint(0, 2, (1, 3, 6, 8), dtype=torch.float32)

    output, _cascade_skips, _returned_sensitivity = model(
        x,
        x.clone(),
        mask,
        mask_type="ktGaussian",
        acc_factor=10,
        acq_type="Flow4d",
        sensitivity_maps=sensitivity,
        mra_prior=vessel_prior,
    )
    assert output.shape == x.shape
