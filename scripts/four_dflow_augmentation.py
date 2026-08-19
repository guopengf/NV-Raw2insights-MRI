"""Forward-model-consistent online augmentation for 4D-flow MRI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from monai.data.fft_utils import fftn_centered, ifftn_centered


def cfg_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def four_dflow_augmentation_enabled(args: Any) -> bool:
    return bool(getattr(args, "data_aug", False)) and bool(
        cfg_get(args, "four_dflow_augmentation.enabled", True)
    )


@dataclass(frozen=True)
class FourDFlowAugmentationParams:
    flip_z: bool = False
    flip_y: bool = False
    shift_z: int = 0
    shift_y: int = 0
    gamma: float = 1.0


def _transform_spatial(tensor: torch.Tensor | None, params: FourDFlowAugmentationParams, axes: tuple[int, int]):
    if tensor is None:
        return None
    result = torch.as_tensor(tensor)
    flip_axes = []
    if params.flip_z:
        flip_axes.append(axes[0])
    if params.flip_y:
        flip_axes.append(axes[1])
    if flip_axes:
        result = torch.flip(result, dims=flip_axes)
    if params.shift_z or params.shift_y:
        result = torch.roll(result, shifts=(params.shift_z, params.shift_y), dims=axes)
    return result.contiguous()


def _shared_contrast_ratio(
    target_complex: torch.Tensor,
    *,
    gamma: float,
    joint_encodings: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    magnitude = target_complex.abs()
    reduction_dims = (-4, -3) if joint_encodings else (-3,)
    reference = magnitude.square().mean(dim=reduction_dims, keepdim=True).sqrt()
    maximum = reference.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    normalized = (reference / maximum).clamp(0.0, 1.0)
    adjusted = normalized.pow(float(gamma)) * maximum
    floor = eps * maximum
    ratio = torch.where(reference > floor, adjusted / reference.clamp_min(floor), torch.ones_like(reference))
    return ratio


def _broadcast_mask(mask: torch.Tensor, target_kspace_complex: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(mask, device=target_kspace_complex.device)
    if result.ndim == target_kspace_complex.ndim + 1 and result.shape[-1] == 1:
        result = result[..., 0]
    while result.ndim < target_kspace_complex.ndim:
        result = result.unsqueeze(-3)
    if result.ndim != target_kspace_complex.ndim:
        raise ValueError(
            f"Cannot broadcast 4D-flow mask shape {tuple(mask.shape)} to target shape "
            f"{tuple(target_kspace_complex.shape)}"
        )
    return result.to(dtype=target_kspace_complex.real.dtype)


def apply_four_dflow_augmentation(
    target_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    params: FourDFlowAugmentationParams,
    joint_encodings: bool,
    sensitivity_maps: torch.Tensor | None = None,
    segmask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Augment aligned image-domain fields and regenerate the masked input exactly."""
    target = torch.as_tensor(target_image)
    if target.shape[-1] != 2:
        raise ValueError(f"Expected real/imag target with trailing size 2, got {tuple(target.shape)}")

    target = _transform_spatial(target, params, (-3, -2))
    sensitivity = _transform_spatial(sensitivity_maps, params, (-3, -2))
    segmentation = _transform_spatial(segmask, params, (-2, -1))

    target_complex = torch.view_as_complex(target.contiguous())
    if params.gamma != 1.0:
        ratio = _shared_contrast_ratio(
            target_complex,
            gamma=params.gamma,
            joint_encodings=joint_encodings,
        )
        target_complex = target_complex * ratio
        target = torch.view_as_real(target_complex).contiguous()

    target_kspace = fftn_centered(target, spatial_dims=2, is_complex=True)
    target_kspace_complex = torch.view_as_complex(target_kspace.contiguous())
    expanded_mask = _broadcast_mask(mask, target_kspace_complex)
    input_kspace = torch.view_as_real(target_kspace_complex * expanded_mask).contiguous()
    input_image = ifftn_centered(input_kspace, spatial_dims=2, is_complex=True).contiguous()

    return {
        "input_image": input_image,
        "target_image": target,
        "sensitivity_maps": sensitivity,
        "segmask": segmentation,
        "params": asdict(params),
    }


class FourDFlowOnlineAugmenter:
    """Sample one transform per case and share it across encodings, frames, and coils."""

    def __init__(self, args: Any) -> None:
        self.enabled = four_dflow_augmentation_enabled(args)
        self.flip_prob = float(cfg_get(args, "four_dflow_augmentation.flip.prob", 0.25))
        self.flip_axes = tuple(str(value).lower() for value in cfg_get(
            args, "four_dflow_augmentation.flip.axes", ("z", "y")
        ))
        self.shift_prob = float(cfg_get(args, "four_dflow_augmentation.shift.prob", 0.25))
        self.max_shift = tuple(int(value) for value in cfg_get(
            args, "four_dflow_augmentation.shift.max_pixels", (3, 8)
        ))
        self.contrast_prob = float(cfg_get(args, "four_dflow_augmentation.contrast.prob", 0.20))
        self.gamma_range = tuple(float(value) for value in cfg_get(
            args, "four_dflow_augmentation.contrast.gamma", (0.8, 1.2)
        ))
        if len(self.max_shift) != 2 or min(self.max_shift) < 0:
            raise ValueError(f"shift.max_pixels must contain two nonnegative values, got {self.max_shift}")
        if len(self.gamma_range) != 2 or min(self.gamma_range) <= 0:
            raise ValueError(f"contrast.gamma must contain two positive values, got {self.gamma_range}")
        unsupported_axes = set(self.flip_axes) - {"z", "y"}
        if unsupported_axes:
            raise ValueError(f"Only in-window z/y flips are supported, got {sorted(unsupported_axes)}")

    @staticmethod
    def _draw(probability: float, generator: torch.Generator | None) -> bool:
        return probability > 0 and bool(torch.rand((), generator=generator).item() < probability)

    def sample_params(self, generator: torch.Generator | None = None) -> FourDFlowAugmentationParams:
        flip_z = "z" in self.flip_axes and self._draw(self.flip_prob, generator)
        flip_y = "y" in self.flip_axes and self._draw(self.flip_prob, generator)
        shift_z = 0
        shift_y = 0
        if self._draw(self.shift_prob, generator):
            shift_z = int(torch.randint(-self.max_shift[0], self.max_shift[0] + 1, (), generator=generator).item())
            shift_y = int(torch.randint(-self.max_shift[1], self.max_shift[1] + 1, (), generator=generator).item())
        gamma = 1.0
        if self._draw(self.contrast_prob, generator):
            low, high = sorted(self.gamma_range)
            gamma = low + (high - low) * float(torch.rand((), generator=generator).item())
        return FourDFlowAugmentationParams(
            flip_z=flip_z,
            flip_y=flip_y,
            shift_z=shift_z,
            shift_y=shift_y,
            gamma=gamma,
        )

    def __call__(
        self,
        target_image: torch.Tensor,
        mask: torch.Tensor,
        *,
        joint_encodings: bool,
        sensitivity_maps: torch.Tensor | None = None,
        segmask: torch.Tensor | None = None,
        params: FourDFlowAugmentationParams | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        if not self.enabled and params is None:
            raise RuntimeError("FourDFlowOnlineAugmenter was called while data augmentation is disabled")
        return apply_four_dflow_augmentation(
            target_image,
            mask,
            params=params or self.sample_params(generator),
            joint_encodings=joint_encodings,
            sensitivity_maps=sensitivity_maps,
            segmask=segmask,
        )


class FourDFlowOnlineAugmentd:
    """Dictionary adapter for the raw-MAT MONAI transform pipeline."""

    def __init__(self, args: Any) -> None:
        self.augmenter = FourDFlowOnlineAugmenter(args)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        if not self.augmenter.enabled:
            return d
        meta = dict(d["kspace_meta_dict"])
        result = self.augmenter(
            d["kspace_ifft"],
            d["mask"],
            joint_encodings=bool(meta.get("joint_encodings", False)),
            sensitivity_maps=d.get("sensitivity_maps"),
            segmask=meta.get("joint_segmask"),
        )
        d["kspace_masked_ifft"] = result["input_image"]
        d["kspace_ifft"] = result["target_image"]
        if result["sensitivity_maps"] is not None:
            d["sensitivity_maps"] = result["sensitivity_maps"]
        if result["segmask"] is not None:
            meta["joint_segmask"] = result["segmask"]
        meta["four_dflow_augmentation"] = result["params"]
        d["kspace_meta_dict"] = meta
        return d
