"""Joint four-encoding helpers for 4D Flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange


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
    """Stable metric-aligned loss over [G,E,...,2] coil-combined images."""

    def __init__(
        self,
        *,
        complex_weight: float = 1.0,
        magnitude_weight: float = 0.1,
        circular_weight: float = 0.5,
        speed_weight: float = 0.25,
        direction_weight: float = 0.05,
        encoding_count: int = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.weights = {
            "complex": float(complex_weight),
            "magnitude": float(magnitude_weight),
            "circular": float(circular_weight),
            "speed": float(speed_weight),
            "direction": float(direction_weight),
        }
        self.encoding_count = int(encoding_count)
        self.eps = float(eps)

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

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
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
        pred_complex = torch.view_as_complex(pred.contiguous())
        target_complex = torch.view_as_complex(target.contiguous())

        complex_error = torch.abs(pred_complex - target_complex)
        magnitude_error = torch.abs(torch.abs(pred_complex) - torch.abs(target_complex))
        target_amplitude = torch.abs(target_complex)
        amplitude_scale = self._masked_mean(target_amplitude, mask).clamp_min(self.eps)

        pred_relative = pred_complex[:, 1:] * torch.conj(pred_complex[:, 0:1])
        target_relative = target_complex[:, 1:] * torch.conj(target_complex[:, 0:1])
        pred_unit = pred_relative / torch.abs(pred_relative).clamp_min(self.eps)
        target_unit = target_relative / torch.abs(target_relative).clamp_min(self.eps)
        circular_error = 1.0 - torch.real(pred_unit * torch.conj(target_unit))
        circular_error = torch.where(torch.abs(target_relative) > self.eps, circular_error, 0.0)

        pred_flow = torch.angle(pred_relative)
        target_flow = torch.angle(target_relative)
        pred_speed = torch.linalg.vector_norm(pred_flow, dim=1)
        target_speed = torch.linalg.vector_norm(target_flow, dim=1)
        speed_error = (pred_speed - target_speed).square()
        speed_denominator = self._masked_mean(target_speed.square(), mask).clamp_min(self.eps)
        speed_loss = torch.sqrt(self._masked_mean(speed_error, mask) / speed_denominator)

        dot = torch.sum(pred_flow * target_flow, dim=1)
        norms = pred_speed * target_speed
        direction_error = 1.0 - torch.clamp(dot / norms.clamp_min(self.eps), -1.0, 1.0)
        direction_error = torch.where(target_speed > self.eps, direction_error, 0.0)

        components = {
            "complex": self._masked_mean(complex_error, mask) / amplitude_scale,
            "magnitude": self._masked_mean(magnitude_error, mask) / amplitude_scale,
            "circular": self._masked_mean(circular_error, mask),
            "speed": speed_loss,
            "direction": self._masked_mean(direction_error, mask),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components
