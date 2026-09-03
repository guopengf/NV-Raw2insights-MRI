"""Joint four-encoding helpers for 4D Flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from mri_data.flow_losses import (
    EMALossNormalizer,
    angular_cosine_loss,
    circular_phase_loss,
    complex2magflow_torch,
    complex_roi_l1_loss,
    relerr_loss,
    velocity_component_loss,
)


@dataclass(frozen=True)
class JointEncodingSpec:
    enabled: bool = False
    mode: str = "batch"
    count: int = 4
    order: tuple[int, ...] = (0, 1, 2, 3)


def _cfg_get(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, default)
    return cur


def joint_encoding_spec(args) -> JointEncodingSpec:
    enabled = bool(_cfg_get(args, "phase3.joint_encoding.enabled", False))
    mode = str(_cfg_get(args, "phase3.joint_encoding.mode", "batch")).lower()
    count = int(_cfg_get(args, "phase3.joint_encoding.count", 4))
    order = tuple(int(value) for value in _cfg_get(args, "phase3.joint_encoding.order", list(range(count))))
    if mode not in {"batch", "channel"}:
        raise ValueError(f"phase3.joint_encoding.mode must be batch or channel, got {mode!r}")
    if count < 2:
        raise ValueError(f"phase3.joint_encoding.count must be >=2, got {count}")
    if len(order) != count or len(set(order)) != count:
        raise ValueError(f"Joint encoding order must contain {count} unique entries, got {order}")
    return JointEncodingSpec(enabled=enabled, mode=mode, count=count, order=order)


def joint_group_batch_size(args) -> int:
    spec = joint_encoding_spec(args)
    if not spec.enabled:
        return int(args.batch_size)
    batch_size = int(args.batch_size)
    if batch_size % spec.count != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by joint encoding count={spec.count}; "
            "batch_size counts total encoding images"
        )
    return batch_size // spec.count


def joint_windowed_input_x_slab(input_tensor, micro_b, final_shape, num_frames: int, num_slices: int):
    """Gather aligned [G,E,S,T,...] windows from [T*X,E,...] input."""
    total_frames = int(final_shape[-5])
    total_slices = int(final_shape[-4])
    frame_half = num_frames // 2
    slice_half = num_slices // 2
    rows = []
    for idx in micro_b:
        idx = int(idx)
        frame_i = idx // total_slices
        slice_i = idx % total_slices
        slab = []
        for slice_offset in range(-slice_half, num_slices - slice_half):
            selected_slice = max(0, min(total_slices - 1, slice_i + slice_offset))
            temporal = []
            for frame_offset in range(-frame_half, num_frames - frame_half):
                selected_frame = (frame_i + frame_offset) % total_frames
                temporal.append(selected_frame * total_slices + selected_slice)
            slab.append(temporal)
        rows.append(slab)
    window_idx = torch.as_tensor(rows, dtype=torch.long, device=input_tensor.device)
    gathered = torch.as_tensor(input_tensor[window_idx])
    return rearrange(gathered, "g s t e ... -> g e s t ..."), window_idx


def gather_joint_window(tensor, window_idx):
    gathered = torch.as_tensor(tensor[window_idx])
    return rearrange(gathered, "g s t e ... -> g e s t ...")


def flatten_joint_model_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return None
    return rearrange(tensor, "g e ... -> (g e) ...")


def restore_joint_model_batch(tensor: torch.Tensor, encoding_count: int) -> torch.Tensor:
    return rearrange(tensor, "(g e) ... -> g e ...", e=encoding_count)


def select_joint_mask_slab(case_mask, micro_b, final_shape, num_slices: int):
    if case_mask is None:
        return None
    mask = torch.as_tensor(case_mask, dtype=torch.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected joint segmask [X,Z,Y], got {tuple(mask.shape)}")
    total_frames = int(final_shape[-5])
    total_slices = int(final_shape[-4])
    half = num_slices // 2
    rows = []
    for idx in micro_b:
        slice_i = int(idx) % total_slices
        rows.append([max(0, min(total_slices - 1, slice_i + off)) for off in range(-half, num_slices - half)])
    return mask[torch.as_tensor(rows, dtype=torch.long)]


class JointEncodingLoss(nn.Module):
    """Configurable joint loss over [G,E,...,2] coil-combined images."""

    def __init__(
        self,
        *,
        complex_weight: float = 1.0,
        magnitude_weight: float = 0.1,
        circular_weight: float = 0.5,
        speed_weight: float = 0.25,
        direction_weight: float = 0.05,
        velocity_weight: float = 0.1,
        relerr_weight: float = 0.1,
        angular_weight: float = 0.1,
        encoding_count: int = 4,
        reference_index: int = 0,
        profile: str = "pengfei_joint",
        velocity_type: str = "smooth_l1",
        angular_min_speed: float = 0.0,
        loss_normalization_enabled: bool = False,
        loss_normalization_momentum: float = 0.99,
        loss_normalization_eps: float = 1e-8,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.profile = str(profile).lower()
        if self.profile not in {"pengfei_joint", "official_aligned"}:
            raise ValueError(
                "phase3.loss.profile must be 'pengfei_joint' or 'official_aligned', "
                f"got {profile!r}"
            )
        if self.profile == "pengfei_joint":
            self.weights = {
                "complex": float(complex_weight),
                "magnitude": float(magnitude_weight),
                "circular": float(circular_weight),
                "speed": float(speed_weight),
                "direction": float(direction_weight),
            }
        else:
            self.weights = {
                "complex": float(complex_weight),
                "circular": float(circular_weight),
                "velocity": float(velocity_weight),
                "relerr": float(relerr_weight),
                "angular": float(angular_weight),
            }
        self.encoding_count = int(encoding_count)
        self.reference_index = int(reference_index)
        if not 0 <= self.reference_index < self.encoding_count:
            raise ValueError(
                f"reference_index={self.reference_index} outside encoding_count={self.encoding_count}"
            )
        self.velocity_type = str(velocity_type)
        self.angular_min_speed = float(angular_min_speed)
        self.eps = float(eps)
        self.normalizer = EMALossNormalizer(
            enabled=loss_normalization_enabled,
            momentum=loss_normalization_momentum,
            eps=loss_normalization_eps,
        )

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return value.mean()
        mask = mask.to(device=value.device, dtype=value.dtype)
        if value.ndim == mask.ndim + 2:
            mask = mask.unsqueeze(1).unsqueeze(-3)
        elif value.ndim == mask.ndim + 1:
            mask = mask.unsqueeze(-3)
        elif value.ndim != mask.ndim:
            raise ValueError(f"Cannot align mask {tuple(mask.shape)} with value {tuple(value.shape)}")
        mask = torch.broadcast_to(mask, value.shape)
        return (value * mask).sum() / mask.sum().clamp_min(self.eps)

    def _pengfei_components(self, pred: torch.Tensor, target: torch.Tensor, mask):
        pred_complex = torch.view_as_complex(pred.contiguous())
        target_complex = torch.view_as_complex(target.contiguous())

        complex_error = torch.abs(pred_complex - target_complex)
        magnitude_error = torch.abs(torch.abs(pred_complex) - torch.abs(target_complex))
        target_amplitude = torch.abs(target_complex)
        amplitude_scale = self._masked_mean(target_amplitude, mask).clamp_min(self.eps)

        reference = pred_complex[:, self.reference_index : self.reference_index + 1]
        target_reference = target_complex[:, self.reference_index : self.reference_index + 1]
        flow_indices = [index for index in range(self.encoding_count) if index != self.reference_index]
        pred_relative = pred_complex[:, flow_indices] * torch.conj(reference)
        target_relative = target_complex[:, flow_indices] * torch.conj(target_reference)
        pred_unit = pred_relative / torch.abs(pred_relative).clamp_min(self.eps)
        target_unit = target_relative / torch.abs(target_relative).clamp_min(self.eps)
        circular_error = 1.0 - torch.real(pred_unit * torch.conj(target_unit))
        circular_error = torch.where(torch.abs(target_relative) > self.eps, circular_error, 0.0)

        pred_flow = torch.angle(pred_relative)
        target_flow = torch.angle(target_relative)
        pred_speed = torch.linalg.vector_norm(pred_flow, dim=1)
        target_speed = torch.linalg.vector_norm(target_flow, dim=1)
        dot = torch.sum(pred_flow * target_flow, dim=1)
        norms = pred_speed * target_speed
        direction_error = 1.0 - torch.clamp(dot / norms.clamp_min(self.eps), -1.0, 1.0)
        direction_error = torch.where(target_speed > self.eps, direction_error, 0.0)

        return {
            "complex": self._masked_mean(complex_error, mask) / amplitude_scale,
            "magnitude": self._masked_mean(magnitude_error, mask) / amplitude_scale,
            "circular": self._masked_mean(circular_error, mask),
            # Keep the historical component name while using the exact official RelErr formula.
            "speed": relerr_loss(pred_flow, target_flow, mask, component_dim=1),
            "direction": self._masked_mean(direction_error, mask),
        }

    def _official_components(self, pred: torch.Tensor, target: torch.Tensor, mask):
        _, pred_flow = complex2magflow_torch(
            pred,
            encoding_dim=1,
            reference_index=self.reference_index,
        )
        _, target_flow = complex2magflow_torch(
            target,
            encoding_dim=1,
            reference_index=self.reference_index,
        )
        return {
            "complex": complex_roi_l1_loss(pred, target, mask),
            "circular": circular_phase_loss(
                pred,
                target,
                mask,
                encoding_dim=1,
                reference_index=self.reference_index,
            ),
            "velocity": velocity_component_loss(
                pred_flow,
                target_flow,
                mask,
                loss_type=self.velocity_type,
            ),
            "relerr": relerr_loss(pred_flow, target_flow, mask, component_dim=1),
            "angular": angular_cosine_loss(
                pred_flow,
                target_flow,
                mask,
                component_dim=1,
                min_speed=self.angular_min_speed,
            ),
        }

    def normalization_state_dict(self) -> dict[str, float]:
        return self.normalizer.state_dict()

    def load_normalization_state_dict(self, state: dict[str, float] | None) -> None:
        self.normalizer.load_state_dict(state)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        ramp: float = 1.0,
    ):
        if (
            pred.shape != target.shape
            or pred.ndim < 4
            or pred.shape[1] != self.encoding_count
            or pred.shape[-1] != 2
        ):
            raise ValueError(
                f"Expected matching [G,{self.encoding_count},...,2] tensors, "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}"
            )

        pred = pred.float()
        target = target.float()
        components = (
            self._pengfei_components(pred, target, mask)
            if self.profile == "pengfei_joint"
            else self._official_components(pred, target, mask)
        )
        total = pred.sum() * 0.0
        for name, value in components.items():
            normalized = self.normalizer.normalize(name, value)
            ramp_multiplier = 1.0
            if self.profile == "official_aligned" and name != "complex":
                ramp_multiplier = float(ramp)
            total = total + self.weights[name] * ramp_multiplier * normalized
        return total, components
