"""Differentiable 4D Flow conversion, ROI losses, and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def as_complex(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 2:
        raise ValueError(f"Expected real/imaginary last dimension of size 2, got {tuple(x.shape)}")
    return torch.view_as_complex(x.contiguous())


def complex2magflow_torch(
    x: torch.Tensor,
    *,
    encoding_dim: int = 1,
    reference_index: int = 0,
    venc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match official ``complex2magflow`` while preserving autograd.

    The official challenge conversion is ``angle(x[d] * conj(x[0]))`` for
    every non-reference encoding. Without ``venc`` the output is in radians,
    which is how the official 2026 scorer calls the function.
    """

    z = as_complex(x)
    encoding_dim = encoding_dim % z.ndim
    num_encodings = z.shape[encoding_dim]
    if not 0 <= reference_index < num_encodings:
        raise ValueError(f"reference_index={reference_index} outside E={num_encodings}")

    magnitude = torch.abs(z)
    reference = z.select(encoding_dim, reference_index).unsqueeze(encoding_dim)
    flow_indices = [idx for idx in range(num_encodings) if idx != reference_index]
    encoded = torch.index_select(
        z,
        encoding_dim,
        torch.as_tensor(flow_indices, device=z.device, dtype=torch.long),
    )
    flow = torch.angle(encoded * torch.conj(reference))
    if venc is not None:
        scale = torch.as_tensor(venc, dtype=flow.dtype, device=flow.device) / torch.pi
        shape = [1] * flow.ndim
        shape[encoding_dim] = len(flow_indices)
        flow = flow * scale.reshape(shape)
    return magnitude, flow


def relative_phase(x: torch.Tensor, *, encoding_dim: int = 1, reference_index: int = 0) -> torch.Tensor:
    return complex2magflow_torch(
        x,
        encoding_dim=encoding_dim,
        reference_index=reference_index,
    )[1]


def _broadcast_mask(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
    """Align [B,(S),Z,Y] ROI tensors with joint component/coil dimensions."""

    if mask is None:
        return None
    mask = torch.as_tensor(mask, dtype=values.dtype, device=values.device)
    if mask.ndim > values.ndim:
        raise ValueError(f"Cannot align ROI mask {tuple(mask.shape)} with values {tuple(values.shape)}")
    if mask.shape == values.shape:
        return mask

    candidates = []
    for positions in combinations(range(values.ndim), mask.ndim):
        if mask.ndim > 0 and mask.shape[0] == values.shape[0] and positions[0] != 0:
            continue
        if all(mask.shape[index] in (1, values.shape[position]) for index, position in enumerate(positions)):
            candidates.append(positions)
    if not candidates:
        raise ValueError(f"Cannot align ROI mask {tuple(mask.shape)} with values {tuple(values.shape)}")

    # Prefer later non-batch dimensions. This maps [B,S,Z,Y] to
    # [B,1,S,1,Z,Y], rather than accidentally treating S as VENC.
    positions = max(candidates)
    shape = [1] * values.ndim
    for source_dim, target_dim in enumerate(positions):
        shape[target_dim] = mask.shape[source_dim]
    return torch.broadcast_to(mask.reshape(shape), values.shape)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, *, eps: float = 1e-8) -> torch.Tensor:
    mask = _broadcast_mask(mask, values)
    if mask is None:
        return values.mean()
    denominator = mask.sum()
    return torch.where(
        denominator > 0,
        (values * mask).sum() / denominator.clamp_min(eps),
        values.sum() * 0.0,
    )


def complex_roi_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor | None,
    *,
    normalize_by_mask: bool = True,
    outside_weight: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Coordinate-wise complex L1: mean of |dRe| and |dIm| inside ROI."""

    values = torch.abs(pred - target).mean(dim=-1)
    if normalize_by_mask:
        inside = _masked_mean(values, roi_mask, eps=eps)
    elif roi_mask is None:
        inside = values.mean()
    else:
        mask = _broadcast_mask(roi_mask, values)
        inside = (values * mask).mean()
    if outside_weight <= 0 or roi_mask is None:
        return inside
    outside = _masked_mean(values, 1.0 - torch.as_tensor(roi_mask), eps=eps)
    return inside + float(outside_weight) * outside


def circular_phase_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor | None,
    *,
    encoding_dim: int = 1,
    reference_index: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_phase = relative_phase(pred, encoding_dim=encoding_dim, reference_index=reference_index)
    target_phase = relative_phase(target, encoding_dim=encoding_dim, reference_index=reference_index)
    return _masked_mean(1.0 - torch.cos(pred_phase - target_phase), roi_mask, eps=eps)


def velocity_component_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    roi_mask: torch.Tensor | None,
    *,
    loss_type: str = "smooth_l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "smooth_l1":
        values = F.smooth_l1_loss(pred_flow, target_flow, reduction="none")
    elif loss_type == "l1":
        values = torch.abs(pred_flow - target_flow)
    elif loss_type in ("l2", "mse"):
        values = (pred_flow - target_flow).square()
    else:
        raise ValueError(f"Unsupported velocity loss type: {loss_type!r}")
    return _masked_mean(values, roi_mask, eps=eps)


class _SafeSqrt(torch.autograd.Function):
    """Exact sqrt forward with a defined zero gradient at x=0."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denominator = 2.0 * torch.sqrt(x.clamp_min(torch.finfo(x.dtype).tiny))
        return torch.where(x > 0, grad_output / denominator, torch.zeros_like(grad_output))


def relerr_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    roi_mask: torch.Tensor | None,
    *,
    component_dim: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable equivalent of official ``utils_metrics.RelErr``."""

    pred_speed = torch.linalg.vector_norm(pred_flow, dim=component_dim)
    target_speed = torch.linalg.vector_norm(target_flow, dim=component_dim)
    mask = _broadcast_mask(roi_mask, pred_speed)
    if mask is None:
        numerator = (target_speed - pred_speed).square().sum()
        denominator = target_speed.square().sum() + eps
    else:
        numerator = (mask * (target_speed - pred_speed).square()).sum()
        denominator = (mask * target_speed.square()).sum() + eps
    return _SafeSqrt.apply(numerator / denominator)


def angular_cosine_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    roi_mask: torch.Tensor | None,
    *,
    component_dim: int = 1,
    min_speed: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Stable cosine-direction surrogate for official degree-valued AngErr."""

    dot = (pred_flow * target_flow).sum(dim=component_dim)
    pred_speed = torch.linalg.vector_norm(pred_flow, dim=component_dim)
    target_speed = torch.linalg.vector_norm(target_flow, dim=component_dim)
    cosine = (dot / (pred_speed * target_speed + eps)).clamp(-1.0, 1.0)
    angular_mask = None if roi_mask is None else torch.as_tensor(roi_mask, dtype=cosine.dtype, device=cosine.device)
    if min_speed > 0:
        speed_mask = (target_speed >= float(min_speed)).to(cosine.dtype)
        angular_mask = speed_mask if angular_mask is None else _broadcast_mask(angular_mask, cosine) * speed_mask
    return _masked_mean(1.0 - cosine, angular_mask, eps=eps)


def flow_loss_ramp(epoch: int, *, enabled: bool, start_epoch: int, end_epoch: int) -> float:
    if not enabled:
        return 1.0
    if end_epoch < start_epoch:
        raise ValueError("flow_loss_ramp.end_epoch must be >= start_epoch")
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    if end_epoch == start_epoch:
        return 1.0
    return float(epoch - start_epoch) / float(end_epoch - start_epoch)


class EMALossNormalizer:
    """Optional detached scalar EMA normalization for auxiliary losses."""

    def __init__(self, *, enabled: bool = False, momentum: float = 0.99, eps: float = 1e-8):
        self.enabled = bool(enabled)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.ema: dict[str, torch.Tensor] = {}

    def normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return value
        with torch.no_grad():
            observed = value.detach().float()
            if dist.is_available() and dist.is_initialized():
                observed = observed.clone()
                dist.all_reduce(observed, op=dist.ReduceOp.SUM)
                observed /= dist.get_world_size()
            if name not in self.ema:
                self.ema[name] = observed
            else:
                self.ema[name].mul_(self.momentum).add_(observed, alpha=1.0 - self.momentum)
            scale = self.ema[name].to(device=value.device, dtype=value.dtype)
        return value / (scale + self.eps)

    def state_dict(self) -> dict[str, float]:
        return {name: float(value.cpu()) for name, value in self.ema.items()}

    def load_state_dict(self, state: dict[str, float] | None) -> None:
        self.ema = {} if not state else {name: torch.tensor(value, dtype=torch.float32) for name, value in state.items()}


def select_training_center_time(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Select center time while deliberately retaining every slab slice."""

    if x.ndim < 4 or x.shape[2] != num_frames:
        raise ValueError(f"Expected [B,S,T,...] with T={num_frames}, got {tuple(x.shape)}")
    return x[:, :, num_frames // 2]


def flatten_joint_venc(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Flatten [B,E,...] to [B*E,...] for shared per-encoding reconstruction."""

    if x.ndim < 3:
        raise ValueError(f"Expected [B,E,...], got {tuple(x.shape)}")
    batch, encodings = x.shape[:2]
    return x.reshape(batch * encodings, *x.shape[2:]), (batch, encodings)


def regroup_joint_venc(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    batch, encodings = shape
    if x.shape[0] != batch * encodings:
        raise ValueError(f"Cannot regroup leading dimension {x.shape[0]} as B={batch}, E={encodings}")
    return x.reshape(batch, encodings, *x.shape[1:])


def flow_score(relerr: float, angerr: float, config) -> float:
    rel_weight = float(getattr(config, "relerr_weight", 1.0))
    ang_weight = float(getattr(config, "angerr_weight", 1.0))
    ang_scale = float(getattr(config, "angerr_scale", 100.0))
    if ang_scale <= 0:
        raise ValueError("checkpoint_selection.flow_score.angerr_scale must be positive")
    return rel_weight * float(relerr) + ang_weight * float(angerr) / ang_scale


@dataclass
class BestFlowMetrics:
    flow: float = float("inf")
    relerr: float = float("inf")
    angerr: float = float("inf")
    ssim: float = -float("inf")
    epochs: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, state: dict | None, legacy_ssim: float | None = None):
        if not state:
            return cls(ssim=-float("inf") if legacy_ssim is None else float(legacy_ssim))
        return cls(
            flow=float(state.get("flow", float("inf"))),
            relerr=float(state.get("relerr", float("inf"))),
            angerr=float(state.get("angerr", float("inf"))),
            ssim=float(state.get("ssim", -float("inf"))),
            epochs=dict(state.get("epochs", {})),
        )

    def update(self, metrics: dict[str, float], epoch: int) -> list[str]:
        improved = []
        for name, direction in (("flow", "min"), ("relerr", "min"), ("angerr", "min"), ("ssim", "max")):
            value = float(metrics[name])
            is_better = value < getattr(self, name) if direction == "min" else value > getattr(self, name)
            if is_better:
                setattr(self, name, value)
                self.epochs[name] = int(epoch)
                improved.append(name)
        return improved

    def state_dict(self) -> dict:
        return {
            "flow": self.flow,
            "relerr": self.relerr,
            "angerr": self.angerr,
            "ssim": self.ssim,
            "epochs": dict(self.epochs),
        }
