from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VascularAttentionAdapter(nn.Module):
    """Lightweight residual adapter guided by a single-channel vessel map."""

    def __init__(self, channels: int, reduction: int = 4, gamma_init: float = 0.0, gamma_mode: str = "shifted_sigmoid"):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.gamma_mode = gamma_mode
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

        # Strict identity is guaranteed by gamma=0. Keep the adapter branch
        # non-zero so gamma receives gradient at the first optimization step.

    def gamma(self) -> torch.Tensor:
        if self.gamma_mode == "direct_clamp":
            return torch.clamp(self.gamma_raw, 0.0, 1.0)
        if self.gamma_mode == "shifted_sigmoid":
            base = torch.sigmoid(torch.zeros((), device=self.gamma_raw.device, dtype=self.gamma_raw.dtype))
            return torch.clamp((torch.sigmoid(self.gamma_raw) - base) / (1.0 - base), 0.0, 1.0)
        raise ValueError(f"Unsupported VAA gamma mode: {self.gamma_mode}")

    def forward(self, feature: torch.Tensor, vessel_map: torch.Tensor) -> torch.Tensor:
        if vessel_map is None:
            raise ValueError("VAA is enabled but vessel_map is None.")
        if vessel_map.dim() == 3:
            vessel_map = vessel_map.unsqueeze(1)
        if vessel_map.shape[-2:] != feature.shape[-2:]:
            vessel_map = F.interpolate(vessel_map, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        vessel_map = vessel_map.to(device=feature.device, dtype=feature.dtype).clamp(0.0, 1.0)
        delta = self.net(torch.cat([feature, vessel_map], dim=1))
        return feature + self.gamma().to(dtype=feature.dtype) * delta
