"""FlowVN-inspired multi-plane regularization for 4D Flow MRI.

The four-plane construction follows the FlowVN demo released with
CMRx4DFlow2026. Only the learned regularizer is implemented here; data
consistency and unrolled-stage momentum remain owned by this repository's
cascaded reconstruction model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


FLOWVN_BRANCHES = ("xyz", "xyt", "yzt", "xzt")


class LearnedPiecewiseLinear(nn.Module):
    """Channel-wise piecewise-linear potential derivative."""

    def __init__(self, channels: int, num_knots: int = 71, value_range: float = 3.5, init_scale: float = 0.01):
        super().__init__()
        if num_knots < 3 or num_knots % 2 == 0:
            raise ValueError(f"num_knots must be an odd integer >= 3, got {num_knots}")
        if value_range <= 0:
            raise ValueError(f"value_range must be positive, got {value_range}")

        self.channels = int(channels)
        self.num_knots = int(num_knots)
        self.value_range = float(value_range)
        grid = torch.linspace(-self.value_range, self.value_range, self.num_knots)
        self.knots = nn.Parameter(grid.repeat(self.channels, 1) * float(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},D,H,W], got {tuple(x.shape)}")

        flat = x.flatten(2)
        position = (flat.clamp(-self.value_range, self.value_range) + self.value_range)
        position = position * ((self.num_knots - 1) / (2.0 * self.value_range))
        lower = position.floor().long().clamp(0, self.num_knots - 2)
        fraction = position - lower.to(position.dtype)

        table = self.knots.unsqueeze(0).expand(flat.shape[0], -1, -1)
        lower_value = torch.gather(table, 2, lower)
        upper_value = torch.gather(table, 2, lower + 1)
        output = lower_value + fraction * (upper_value - lower_value)
        return output.reshape_as(x)


class AccelerationModulation(nn.Module):
    """Positive spline modulation over the ktGaussian acceleration factor."""

    def __init__(self, min_acceleration: float = 9.0, max_acceleration: float = 51.0, num_knots: int = 11):
        super().__init__()
        if num_knots < 2:
            raise ValueError("Acceleration modulation requires at least two knots.")
        self.min_acceleration = float(min_acceleration)
        self.max_acceleration = float(max_acceleration)
        self.num_knots = int(num_knots)
        inverse_softplus_one = math.log(math.exp(1.0) - 1.0)
        self.knots = nn.Parameter(torch.full((self.num_knots,), inverse_softplus_one))

    def forward(self, acceleration, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(acceleration, device=device, dtype=dtype)
        value = value.reshape(-1)
        position = (value.clamp(self.min_acceleration, self.max_acceleration) - self.min_acceleration)
        position = position * ((self.num_knots - 1) / (self.max_acceleration - self.min_acceleration))
        lower = position.floor().long().clamp(0, self.num_knots - 2)
        fraction = position - lower.to(position.dtype)
        raw = self.knots[lower] + fraction * (self.knots[lower + 1] - self.knots[lower])
        return F.softplus(raw)


class FlowVNPlaneRegularizer(nn.Module):
    """Tied K^T phi(Kx) regularizer for one 3D plane family."""

    def __init__(
        self,
        in_channels: int,
        features: int = 8,
        kernel_size: int = 3,
        num_knots: int = 71,
        activation_range: float = 3.5,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.in_channels = int(in_channels)
        self.features = int(features)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2

        weight = torch.randn(self.features, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        weight = weight - weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        weight = weight / weight.square().sum(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-8)
        self.weight = nn.Parameter(weight)
        self.activation = LearnedPiecewiseLinear(
            self.features,
            num_knots=num_knots,
            value_range=activation_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = F.conv3d(x, self.weight, padding=self.padding)
        response = self.activation(response)
        return F.conv_transpose3d(response, self.weight, padding=self.padding)


class FlowVNMultiPlaneMixer(nn.Module):
    """Four 3D plane families over the explicit [X,Z,Y,T] domain.

    Input and output use [B,S,T,C,Z,Y,2], where S is raw-x after the kx
    inverse FFT and the final dimension stores real/imaginary components.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: int = 8,
        kernel_size: int = 3,
        branches: Sequence[str] = FLOWVN_BRANCHES,
        num_knots: int = 71,
        activation_range: float = 3.5,
        scale_init: float = 0.0,
        acceleration_modulation: bool = True,
    ):
        super().__init__()
        normalized_branches = tuple(str(branch).lower() for branch in branches)
        unknown = sorted(set(normalized_branches) - set(FLOWVN_BRANCHES))
        if unknown:
            raise ValueError(f"Unsupported FlowVN mixer branches: {unknown}")
        if not normalized_branches:
            raise ValueError("At least one FlowVN mixer branch is required.")
        if len(set(normalized_branches)) != len(normalized_branches):
            raise ValueError(f"FlowVN mixer branches must be unique, got {normalized_branches}")

        self.in_channels = int(in_channels)
        self.features = int(features)
        self.branches = normalized_branches
        self.regularizers = nn.ModuleDict(
            {
                branch: FlowVNPlaneRegularizer(
                    self.in_channels,
                    features=self.features,
                    kernel_size=kernel_size,
                    num_knots=num_knots,
                    activation_range=activation_range,
                )
                for branch in self.branches
            }
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.acceleration_modulation = (
            AccelerationModulation() if bool(acceleration_modulation) else None
        )

    def _apply_branch(self, branch: str, x: torch.Tensor) -> torch.Tensor:
        b, s, t, c, z, y, two = x.shape
        regularizer = self.regularizers[branch]
        if branch == "xyz":
            plane = rearrange(x, "b s t c z y r -> (b t r) c s z y")
            update = regularizer(plane)
            return rearrange(update, "(b t r) c s z y -> b s t c z y r", b=b, t=t, r=two)
        if branch == "xyt":
            plane = rearrange(x, "b s t c z y r -> (b z r) c s y t")
            update = regularizer(plane)
            return rearrange(update, "(b z r) c s y t -> b s t c z y r", b=b, z=z, r=two)
        if branch == "yzt":
            plane = rearrange(x, "b s t c z y r -> (b s r) c z y t")
            update = regularizer(plane)
            return rearrange(update, "(b s r) c z y t -> b s t c z y r", b=b, s=s, r=two)
        if branch == "xzt":
            plane = rearrange(x, "b s t c z y r -> (b y r) c s z t")
            update = regularizer(plane)
            return rearrange(update, "(b y r) c s z t -> b s t c z y r", b=b, y=y, r=two)
        raise AssertionError(f"Unhandled FlowVN mixer branch: {branch}")

    def forward(self, x: torch.Tensor, acceleration=None) -> torch.Tensor:
        if x.ndim != 7 or x.shape[-1] != 2:
            raise ValueError(f"Expected [B,S,T,C,Z,Y,2], got {tuple(x.shape)}")
        if x.shape[3] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[3]}")

        update = torch.zeros_like(x)
        for branch in self.branches:
            update = update + self._apply_branch(branch, x)
        update = update / float(self.features)

        scale = self.scale.to(dtype=x.dtype)
        if self.acceleration_modulation is not None:
            if acceleration is None:
                raise ValueError("FlowVN acceleration modulation is enabled but acceleration was not provided.")
            modulation = self.acceleration_modulation(acceleration, device=x.device, dtype=x.dtype)
            if modulation.numel() == 1:
                scale = scale * modulation[0]
            elif modulation.numel() == x.shape[0]:
                scale = scale * modulation.view(-1, 1, 1, 1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Acceleration has {modulation.numel()} values for batch size {x.shape[0]}."
                )
        return scale * update
