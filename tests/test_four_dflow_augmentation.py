from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from monai.data.fft_utils import fftn_centered


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from four_dflow_augmentation import (  # noqa: E402
    FourDFlowAugmentationParams,
    FourDFlowOnlineAugmenter,
    apply_four_dflow_augmentation,
)


def _complex_random(shape, seed=11):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(*shape, 2, generator=generator)
    return value.contiguous()


def _geometry(tensor, params, axes):
    result = tensor
    dims = []
    if params.flip_z:
        dims.append(axes[0])
    if params.flip_y:
        dims.append(axes[1])
    if dims:
        result = torch.flip(result, dims=dims)
    return torch.roll(result, shifts=(params.shift_z, params.shift_y), dims=axes)


def test_joint_augmentation_preserves_alignment_phase_and_data_consistency():
    target = _complex_random((2, 3, 2, 4, 2, 7, 9))
    sensitivity = _complex_random((2, 3, 2, 4, 2, 7, 9), seed=12)
    segmask = torch.zeros(2, 3, 7, 9)
    segmask[..., 2:5, 3:7] = 1
    mask = torch.zeros(2, 3, 2, 7, 9)
    mask[..., ::2, 1::3] = 1
    params = FourDFlowAugmentationParams(
        flip_z=True,
        flip_y=False,
        shift_z=2,
        shift_y=-3,
        gamma=0.85,
    )

    result = apply_four_dflow_augmentation(
        target,
        mask,
        params=params,
        joint_encodings=True,
        sensitivity_maps=sensitivity,
        segmask=segmask,
    )

    expected_sensitivity = _geometry(sensitivity, params, (-3, -2))
    expected_segmask = _geometry(segmask, params, (-2, -1))
    torch.testing.assert_close(result["sensitivity_maps"], expected_sensitivity)
    torch.testing.assert_close(result["segmask"], expected_segmask)

    geometric_target = _geometry(target, params, (-3, -2))
    geometric_complex = torch.view_as_complex(geometric_target)
    augmented_complex = torch.view_as_complex(result["target_image"])
    phase_residual = augmented_complex * geometric_complex.conj()
    assert phase_residual.imag.abs().max().item() < 2e-5
    assert phase_residual.real.min().item() >= -2e-5

    positive = geometric_complex.abs() > 1e-5
    ratios = torch.zeros_like(geometric_complex.abs())
    ratios[positive] = augmented_complex.abs()[positive] / geometric_complex.abs()[positive]
    shared_ratio_error = (ratios - ratios.mean(dim=(-4, -3), keepdim=True))[positive]
    assert shared_ratio_error.abs().max().item() < 2e-5

    target_kspace = fftn_centered(result["target_image"], spatial_dims=2, is_complex=True)
    input_kspace = fftn_centered(result["input_image"], spatial_dims=2, is_complex=True)
    target_kspace = torch.view_as_complex(target_kspace)
    input_kspace = torch.view_as_complex(input_kspace)
    expanded_mask = mask[:, :, :, None, None]
    torch.testing.assert_close(input_kspace, target_kspace * expanded_mask, atol=2e-5, rtol=2e-5)


def test_non_joint_augmentation_uses_the_same_forward_model():
    target = _complex_random((2, 3, 2, 2, 7, 9), seed=21)
    mask = torch.ones(2, 3, 2, 7, 9)
    mask[..., 1::2] = 0
    params = FourDFlowAugmentationParams(flip_y=True, shift_y=2, gamma=1.15)
    result = apply_four_dflow_augmentation(
        target,
        mask,
        params=params,
        joint_encodings=False,
    )
    target_kspace = torch.view_as_complex(
        fftn_centered(result["target_image"], spatial_dims=2, is_complex=True)
    )
    input_kspace = torch.view_as_complex(
        fftn_centered(result["input_image"], spatial_dims=2, is_complex=True)
    )
    torch.testing.assert_close(input_kspace, target_kspace * mask.unsqueeze(-3), atol=2e-5, rtol=2e-5)


def test_parameter_sampling_is_reproducible_and_bounded():
    args = SimpleNamespace(
        data_aug=True,
        four_dflow_augmentation=SimpleNamespace(
            enabled=True,
            flip=SimpleNamespace(prob=1.0, axes=["z", "y"]),
            shift=SimpleNamespace(prob=1.0, max_pixels=[3, 8]),
            contrast=SimpleNamespace(prob=1.0, gamma=[0.8, 1.2]),
        ),
    )
    augmenter = FourDFlowOnlineAugmenter(args)
    first = augmenter.sample_params(torch.Generator().manual_seed(123))
    second = augmenter.sample_params(torch.Generator().manual_seed(123))
    assert first == second
    assert first.flip_z and first.flip_y
    assert abs(first.shift_z) <= 3
    assert abs(first.shift_y) <= 8
    assert 0.8 <= first.gamma <= 1.2


def run_directly():
    tests = (
        test_joint_augmentation_preserves_alignment_phase_and_data_consistency,
        test_non_joint_augmentation_uses_the_same_forward_model,
        test_parameter_sampling_is_reproducible_and_bounded,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"COMPLETED {len(tests)} 4D-flow augmentation tests", flush=True)


if __name__ == "__main__":
    run_directly()
