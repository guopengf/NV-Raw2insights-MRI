from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VascularAttentionAdapter(nn.Module):
    """Mask-guided vascular adapter.

    ``attention="legacy"`` reproduces the original residual adapter so existing
    VAA checkpoints remain compatible. ``attention="gate"`` adds a learned
    mask gate to that residual branch. ``attention="qkv"`` uses true spatial
    QKV attention: Q/K/V are generated from feature maps and the vessel prior is
    injected as attention bias over spatial keys.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        gamma_init: float = 0.0,
        gamma_mode: str = "shifted_sigmoid",
        prior_channels: int = 1,
        attention: str = "legacy",
        num_heads: int = 4,
        attention_stride: int = 1,
        use_mask_bias: bool = True,
        spatial_dims: int = 2,
    ):
        super().__init__()
        if spatial_dims not in {2, 3}:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        hidden = max(channels // reduction, 1)
        self.hidden = hidden
        self.prior_channels = int(prior_channels)
        self.spatial_dims = int(spatial_dims)
        self.attention = str(attention).lower()
        if self.attention not in {"legacy", "gate", "qkv"}:
            raise ValueError(f"Unsupported VAA attention type: {attention}. Use 'legacy', 'gate', or 'qkv'.")

        if self.attention in {"legacy", "gate"}:
            self.net = nn.Sequential(
                conv(channels + self.prior_channels, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                conv(hidden, hidden, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                conv(hidden, channels, kernel_size=1, bias=True),
            )
            if self.attention == "gate":
                self.gate = nn.Sequential(
                    conv(self.prior_channels, hidden, kernel_size=1, bias=True),
                    nn.GELU(),
                    conv(hidden, 1, kernel_size=3, padding=1, bias=True),
                    nn.Sigmoid(),
                )
        else:
            num_heads = max(int(num_heads), 1)
            if hidden % num_heads != 0:
                num_heads = math.gcd(hidden, num_heads) or 1
            self.num_heads = num_heads
            self.head_dim = hidden // num_heads
            self.attention_stride = max(int(attention_stride), 1)
            self.use_mask_bias = bool(use_mask_bias)

            self.q_proj = conv(channels, hidden, kernel_size=1, bias=False)
            self.k_proj = conv(channels, hidden, kernel_size=1, bias=False)
            self.v_proj = conv(channels, hidden, kernel_size=1, bias=False)
            self.out_proj = conv(hidden, channels, kernel_size=1, bias=True)
            if self.use_mask_bias:
                self.mask_bias = conv(self.prior_channels, self.num_heads, kernel_size=1, bias=True)
                self.mask_bias_scale = nn.Parameter(torch.ones(1))

        self.gamma_mode = gamma_mode
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    def gamma(self) -> torch.Tensor:
        if self.gamma_mode == "direct_clamp":
            return torch.clamp(self.gamma_raw, 0.0, 1.0)
        if self.gamma_mode == "shifted_sigmoid":
            base = torch.sigmoid(torch.zeros((), device=self.gamma_raw.device, dtype=self.gamma_raw.dtype))
            return torch.clamp((torch.sigmoid(self.gamma_raw) - base) / (1.0 - base), 0.0, 1.0)
        raise ValueError(f"Unsupported VAA gamma mode: {self.gamma_mode}")

    def _prepare_prior(self, vessel_map: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if vessel_map is None:
            raise ValueError("VAA is enabled but vessel_map is None.")
        if feature.dim() == 4 and vessel_map.dim() == 3:
            vessel_map = vessel_map.unsqueeze(1)
        elif feature.dim() == 5 and vessel_map.dim() == 4:
            vessel_map = vessel_map.unsqueeze(1)
        if vessel_map.dim() != feature.dim():
            expected = "[B,C,H,W]" if feature.dim() == 4 else "[B,C,D,H,W]"
            raise ValueError(f"Expected vessel_map with shape {expected}, got {tuple(vessel_map.shape)}")
        vessel_map = vessel_map.to(device=feature.device, dtype=feature.dtype).clamp(0.0, 1.0)
        if vessel_map.shape[2:] != feature.shape[2:]:
            mode = "trilinear" if feature.dim() == 5 else "bilinear"
            vessel_map = F.interpolate(vessel_map, size=feature.shape[2:], mode=mode, align_corners=False)
        if vessel_map.shape[1] == self.prior_channels:
            return vessel_map
        if vessel_map.shape[1] == 1 and self.prior_channels > 1:
            return vessel_map.expand(-1, self.prior_channels, *([-1] * self.spatial_dims))
        if self.prior_channels == 1:
            return vessel_map.mean(dim=1, keepdim=True)
        if vessel_map.shape[1] > self.prior_channels:
            return vessel_map[:, : self.prior_channels]
        pad = self.prior_channels - vessel_map.shape[1]
        tail = vessel_map[:, -1:].expand(-1, pad, *([-1] * self.spatial_dims))
        return torch.cat([vessel_map, tail], dim=1)

    def forward(self, feature: torch.Tensor, vessel_map: torch.Tensor) -> torch.Tensor:
        prior = self._prepare_prior(vessel_map, feature)
        gamma = self.gamma().to(dtype=feature.dtype)
        if self.attention in {"legacy", "gate"}:
            delta = self.net(torch.cat([feature, prior], dim=1))
            if self.attention == "legacy":
                return feature + gamma * delta
            gate = self.gate(prior)
            return feature + gamma * gate * delta

        b = feature.shape[0]
        spatial_shape = feature.shape[2:]
        query_points = math.prod(spatial_shape)

        q = self.q_proj(feature)
        key_feature = feature
        value_feature = feature
        key_prior = prior
        if self.attention_stride > 1:
            stride = self.attention_stride
            if self.spatial_dims == 3:
                pool_kernel = (1, stride, stride)
                key_feature = F.avg_pool3d(key_feature, kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
                value_feature = F.avg_pool3d(
                    value_feature, kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True
                )
                key_prior = F.avg_pool3d(key_prior, kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
            else:
                key_feature = F.avg_pool2d(key_feature, kernel_size=stride, stride=stride, ceil_mode=True)
                value_feature = F.avg_pool2d(value_feature, kernel_size=stride, stride=stride, ceil_mode=True)
                key_prior = F.avg_pool2d(key_prior, kernel_size=stride, stride=stride, ceil_mode=True)

        k = self.k_proj(key_feature)
        v = self.v_proj(value_feature)
        key_points = math.prod(k.shape[2:])

        q = q.view(b, self.num_heads, self.head_dim, query_points).transpose(-1, -2)
        k = k.view(b, self.num_heads, self.head_dim, key_points)
        v = v.view(b, self.num_heads, self.head_dim, key_points).transpose(-1, -2)

        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        if self.use_mask_bias:
            bias = self.mask_bias(key_prior).view(b, self.num_heads, 1, key_points)
            attn = attn + self.mask_bias_scale.to(dtype=attn.dtype) * bias
        attn = torch.softmax(attn, dim=-1)
        delta = torch.matmul(attn, v)
        delta = delta.transpose(-1, -2).contiguous().view(b, self.hidden, *spatial_shape)
        delta = self.out_proj(delta)
        return feature + gamma * delta
