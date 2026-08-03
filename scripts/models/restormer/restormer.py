# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.flowvn_mixer import FlowVNMultiPlaneMixer
from models.vaa import VascularAttentionAdapter
from models.varnet import CoilSensitivityModel_DCAE
from monai.apps.reconstruction.networks.nets.utils import divisible_pad_t, inverse_divisible_pad_t
from monai.data.fft_utils import fftn_centered, ifftn_centered
from monai.networks.utils import copy_model_state
from timm.layers import DropPath
from torch import Tensor, einsum
from torch.utils.checkpoint import checkpoint
from utils import *


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob, labels_embed_magnitude_scale=1.0):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.labels_embed_magnitude_scale = labels_embed_magnitude_scale

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings * self.labels_embed_magnitude_scale


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def to(x):
    return {"device": x.device, "dtype": x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def conv_nd(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


def divisible_pad_zy(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Symmetrically pad only Z/Y for a [B,C,X,Z,Y] tensor."""

    z, y = x.shape[-2:]
    pad_z = (factor - z % factor) % factor
    pad_y = (factor - y % factor) % factor
    pad_top = pad_z // 2
    pad_bottom = pad_z - pad_top
    pad_left = pad_y // 2
    pad_right = pad_y - pad_left
    if pad_z or pad_y:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, 0, 0))
    return x, (pad_left, pad_right, pad_top, pad_bottom)


def inverse_divisible_pad_zy(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = padding
    z_end = x.shape[-2] - pad_bottom if pad_bottom else x.shape[-2]
    y_end = x.shape[-1] - pad_right if pad_right else x.shape[-1]
    return x[..., pad_top:z_end, pad_left:y_end]


class PixelUnshuffleZY(nn.Module):
    """Pixel-unshuffle Z/Y while preserving the raw-x depth."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = self.factor
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f"Z/Y shape {tuple(x.shape[-2:])} is not divisible by {factor}")
        return rearrange(
            x,
            "b c d (z rz) (y ry) -> b (c rz ry) d z y",
            rz=factor,
            ry=factor,
        )


class PixelShuffleZY(nn.Module):
    """Pixel-shuffle Z/Y while preserving the raw-x depth."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = self.factor
        divisor = factor * factor
        if x.shape[1] % divisor:
            raise ValueError(f"Channel count {x.shape[1]} is not divisible by {divisor}")
        return rearrange(
            x,
            "b (c rz ry) d z y -> b c d (z rz) (y ry)",
            rz=factor,
            ry=factor,
        )


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class OCAB(nn.Module):
    def __init__(
        self,
        dim,
        window_size=8,
        overlap_ratio=0.5,
        num_heads=2,
        dim_head=16,
        bias=False,
    ):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head=self.dim_head,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(
            qs,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        # split heads
        qs, ks, vs = map(
            lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads),
            (qs, ks, vs),
        )

        # attention
        qs = qs * self.scale
        spatial_attn = qs @ ks.transpose(-2, -1)
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = spatial_attn @ vs

        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )

        # merge spatial and channel
        out = self.project_out(out)

        return out


class MDTA(nn.Module):
    def __init__(self, channels, num_heads, spatial_dims=2):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        conv = conv_nd(spatial_dims)

        self.qkv = conv(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = conv(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = conv(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        spatial_points = math.prod(spatial_shape)
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, spatial_points)
        k = k.reshape(b, self.num_heads, -1, spatial_points)
        v = v.reshape(b, self.num_heads, -1, spatial_points)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, *spatial_shape))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, emb_channels=None, spatial_dims=2):
        super(GDFN, self).__init__()

        conv = conv_nd(spatial_dims)
        hidden_channels = int(channels * expansion_factor)
        self.project_in = conv(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = conv(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = conv(hidden_channels, channels, kernel_size=1, bias=False)

        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    hidden_channels,
                ),
            )
        else:
            self.emb_layers = None

    def forward(self, x, emb=None):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if emb is not None and self.emb_layers is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out

        x = self.project_out(x)
        return x


class MoE(nn.Module):
    """
    A sparse Mixture of Experts (MoE) layer with top-k gating.

    This layer routes each input to a specific number of experts (`top_k`)
    and computes a weighted sum of their outputs.
    """

    def __init__(self, channels, expansion_factor, num_experts=4, top_k=1, spatial_dims=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.spatial_dims = int(spatial_dims)
        self.experts = nn.ModuleList(
            [GDFN(channels, expansion_factor, spatial_dims=self.spatial_dims) for _ in range(self.num_experts)]
        )

        # Gating network
        self.gate = nn.Linear(channels, self.num_experts)

    def forward(self, x):
        b, c = x.shape[:2]

        # 1. Use global average pooling to get a feature vector for routing
        if self.spatial_dims == 3:
            gating_input = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)
        else:
            gating_input = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1)

        # 2. Get router logits from the gate
        router_logits = self.gate(gating_input)  # Shape: (b, num_experts)

        # 3. Select top-k experts
        # Find the top_k scores and their indices
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # Shape: (b, top_k)

        # 4. Re-normalize the weights of the top-k experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 5. Create a sparse tensor of the final weights
        # Initialize a tensor of zeros with the same shape as the router logits
        sparse_weights = torch.zeros_like(router_logits)
        # Place the calculated top-k weights at the correct indices
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)  # In-place scatter

        # 6. Compute and combine expert outputs
        # Note: This implementation computes all experts but only uses the top-k ones.
        # For true computational savings, a more complex dispatching mechanism is needed.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        routing_weights = sparse_weights.view(b, self.num_experts, *([1] * (x.ndim - 1)))

        # Compute the weighted sum of expert outputs (only top-k will have non-zero weights)
        weighted_output = torch.sum(routing_weights * expert_outputs, dim=1)

        return weighted_output


class TransformerBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        num_heads,
        expansion_factor,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        emb_channels=None,
        spatial_dims=2,
    ):
        super(TransformerBlock, self).__init__()

        self.norm_type = norm_type
        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        elif self.norm_type == "bn":
            norm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
            self.norm1 = norm(channels)
            self.norm2 = norm(channels)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise NotImplementedError("Norm {} is not implemented. (bn or ln or null)".format(norm_type))
        self.attn = MDTA(channels, num_heads, spatial_dims=spatial_dims)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Conditionally use MoE or a single GDFN
        if num_experts and num_experts > 0:
            self.ffn = MoE(channels, expansion_factor, num_experts, top_k, spatial_dims=spatial_dims)
        else:
            self.ffn = GDFN(
                channels,
                expansion_factor,
                emb_channels=emb_channels,
                spatial_dims=spatial_dims,
            )

        self.post_norm = post_norm

    def forward(self, x, emb=None):
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        if self.norm_type == "ln":
            if self.post_norm:
                x = x + self.norm1(
                    self.drop_path(self.attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, *spatial_shape)
                x = x + self.norm2(
                    self.drop_path(self.ffn(x, emb)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, *spatial_shape)
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, *spatial_shape)
                    )
                )
                x = x + self.drop_path(
                    self.ffn(
                        self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, *spatial_shape),
                        emb,
                    )
                )
        else:
            if self.post_norm:
                x = x + self.norm1(self.drop_path(self.attn(x)))
                x = x + self.norm2(self.drop_path(self.ffn(x, emb)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x), emb))
        return x


class HybridTransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_channel_heads,
        num_spatial_heads,
        expansion_factor,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
    ):
        super(HybridTransformerBlock, self).__init__()

        self.norm_type = norm_type
        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.norm3 = nn.LayerNorm(channels)
            self.norm4 = nn.LayerNorm(channels)
        elif self.norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            self.norm3 = nn.BatchNorm2d(channels)
            self.norm4 = nn.BatchNorm2d(channels)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            self.norm4 = nn.Identity()
        else:
            raise NotImplementedError("Norm {} is not implemented. (bn or ln or null)".format(norm_type))
        self.channel_attn = MDTA(channels, num_channel_heads)
        self.spatial_attn = OCAB(channels, num_heads=num_spatial_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Conditionally use MoE or a single GDFN
        if num_experts and num_experts > 0:
            self.channel_ffn = MoE(channels, expansion_factor, num_experts, top_k)
            self.spatial_ffn = MoE(channels, expansion_factor, num_experts, top_k)
        else:
            self.channel_ffn = GDFN(channels, expansion_factor)
            self.spatial_ffn = GDFN(channels, expansion_factor)

        self.post_norm = post_norm

    def forward(self, x):
        b, c, h, w = x.shape
        if self.norm_type == "ln":
            if self.post_norm:
                x = x + self.norm1(
                    self.drop_path(self.channel_attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm2(
                    self.drop_path(self.channel_ffn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm3(
                    self.drop_path(self.spatial_attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm4(
                    self.drop_path(self.spatial_ffn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
            else:
                x = x + self.drop_path(
                    self.channel_attn(
                        self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.channel_ffn(
                        self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.spatial_attn(
                        self.norm3(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.spatial_ffn(
                        self.norm4(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
        else:
            if self.post_norm:
                x = x + self.norm1(self.drop_path(self.channel_attn(x)))
                x = x + self.norm2(self.drop_path(self.channel_ffn(x)))
                x = x + self.norm3(self.drop_path(self.spatial_attn(x)))
                x = x + self.norm4(self.drop_path(self.spatial_ffn(x)))
            else:
                x = x + self.drop_path(self.channel_attn(self.norm1(x)))
                x = x + self.drop_path(self.channel_ffn(self.norm2(x)))
                x = x + self.drop_path(self.spatial_attn(self.norm3(x)))
                x = x + self.drop_path(self.spatial_ffn(self.norm4(x)))
        return x


class DownSample(nn.Module):
    def __init__(self, channels, spatial_dims=2):
        super(DownSample, self).__init__()
        conv = conv_nd(spatial_dims)
        shuffle = PixelUnshuffleZY(2) if spatial_dims == 3 else nn.PixelUnshuffle(2)
        self.body = nn.Sequential(
            conv(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            shuffle,
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, spatial_dims=2):
        super(UpSample, self).__init__()
        conv = conv_nd(spatial_dims)
        shuffle = PixelShuffleZY(2) if spatial_dims == 3 else nn.PixelShuffle(2)
        self.body = nn.Sequential(
            conv(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            shuffle,
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """
    A Restormer-style U-Net backbone that supports an arbitrary number of levels.

    Args:
        in_channel (int): input channels.
        out_channel (int): output channels.
        num_blocks (List[int]): number of TransformerBlocks per level (length L).
        num_heads  (List[int]): attention heads per level (length L).
        channels   (List[int]): channel width per level from top to bottom (length L, L >= 1).
        num_refinement (int): number of TransformerBlocks in final refinement at top level.
        expansion_factor, num_experts, top_k, drop_path, norm_type, post_norm: passed to TransformerBlock.

    cas_skips contract (same as original):
        - In forward(cas_skips=...), cas_skips is a list/tuple of length L-1.
        - cas_skips[i-1] is added to the encoder *input* at level i (i = 1..L-1).

    Returns:
        out: final image-like tensor.
        cas_outs (tuple of length L-1):
            [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        For L=4, this matches original behavior: (out_dec2, out_dec3, out_enc4).
        For L=1, this is (out_dec1).
    """

    def __init__(
        self,
        in_channel=2,
        out_channel=2,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        ms_refinement=False,
        time_cond=False,
        label_cond=False,
        num_classes=9,
        timestep_scale=50.0,
        label_class_scale=10.0,
        time_embed_scale=1,
        class_dropout_prob=0.1,
        labels_embed_magnitude_scale=1.0,
        phase3=None,
        spatial_dims=2,
    ):
        super().__init__()

        # ---- Validation ----
        assert (
            len(num_blocks) == len(num_heads) == len(channels)
        ), "num_blocks, num_heads, and channels must have the same length"
        # (Relaxed) allow L == 1 now
        self.L = len(channels)
        self.channels = channels
        self.time_cond = time_cond
        self.label_cond = label_cond
        self.num_classes = num_classes
        self.timestep_scale = timestep_scale
        self.label_class_scale = label_class_scale
        self.ms_refinement = ms_refinement
        self.spatial_dims = int(spatial_dims)
        conv = conv_nd(self.spatial_dims)
        self.enable_vaa = bool(getattr(phase3, "enable_vaa", False)) if phase3 is not None else False
        vaa_cfg = getattr(phase3, "vaa", None) if phase3 is not None else None
        gamma_cfg = getattr(phase3, "gamma", None) if phase3 is not None else None
        recon_mode = normalize_recon_mode(getattr(phase3, "recon_mode", "slice") if phase3 is not None else "slice")
        if self.spatial_dims == 3:
            prior_channels = 1
        else:
            prior_channels = int(getattr(phase3, "num_slices", 1)) if recon_mode == "slab" else 1
        self.vaa_locations = list(getattr(vaa_cfg, "locations", [])) if vaa_cfg is not None else []
        gamma_init = float(getattr(gamma_cfg, "init", 0.0)) if gamma_cfg is not None else 0.0
        gamma_mode = getattr(gamma_cfg, "mode", "shifted_sigmoid") if gamma_cfg is not None else "shifted_sigmoid"
        gamma_trainable = bool(getattr(gamma_cfg, "trainable", True)) if gamma_cfg is not None else True
        reduction = int(getattr(vaa_cfg, "reduction", 4)) if vaa_cfg is not None else 4
        attention = getattr(vaa_cfg, "attention", "legacy") if vaa_cfg is not None else "legacy"
        vaa_heads = int(getattr(vaa_cfg, "num_heads", 4)) if vaa_cfg is not None else 4
        attention_stride = int(getattr(vaa_cfg, "attention_stride", 1)) if vaa_cfg is not None else 1
        use_mask_bias = bool(getattr(vaa_cfg, "use_mask_bias", True)) if vaa_cfg is not None else True
        L = self.L

        self.emb_channels = channels[0] * time_embed_scale if time_cond else None

        # Shallow embedding at level 0
        self.embed_conv = conv(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        # Encoders: level 0..L-1
        self.encoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                            spatial_dims=self.spatial_dims,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L)
            ]
        )

        # Down path: L-1 transitions
        self.downs = nn.ModuleList([DownSample(channels[i], spatial_dims=self.spatial_dims) for i in range(L - 1)])

        # Up path: L-1 transitions (index i corresponds to going from level i+1 -> i)
        self.ups = nn.ModuleList([UpSample(channels[i + 1], spatial_dims=self.spatial_dims) for i in range(L - 1)])

        # Reduce 1x1 after concatenation at each decoder stage i (levels i = L-2 .. 0)
        self.reduces = nn.ModuleList(
            [conv(2 * channels[i], channels[i], kernel_size=1, bias=False) for i in range(L - 1)]
        )

        # Decoders: one per stage for levels i = 0..L-2
        self.decoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            drop_path=drop_path,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                            spatial_dims=self.spatial_dims,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L - 1)
            ]
        )
        self.refinement = TimestepEmbedSequential(
            *[
                TransformerBlock(
                    channels[0],
                    num_heads[0],
                    expansion_factor,
                    num_experts,
                    top_k,
                    drop_path=drop_path,
                    norm_type=norm_type,
                    post_norm=post_norm,
                    emb_channels=self.emb_channels,
                    spatial_dims=self.spatial_dims,
                )
                for _ in range(num_refinement)
            ]
        )
        # Final conv
        self.output = conv(channels[0], out_channel, kernel_size=3, padding=1, bias=False)
        self.vaa_adapters = nn.ModuleDict()
        if self.enable_vaa:
            if "bottleneck" in self.vaa_locations:
                self.vaa_adapters["bottleneck"] = VascularAttentionAdapter(
                    channels[-1],
                    reduction=reduction,
                    gamma_init=gamma_init,
                    gamma_mode=gamma_mode,
                    prior_channels=prior_channels,
                    attention=attention,
                    num_heads=vaa_heads,
                    attention_stride=attention_stride,
                    use_mask_bias=use_mask_bias,
                    spatial_dims=self.spatial_dims,
                )
            if "intermediate" in self.vaa_locations:
                self.vaa_adapters["intermediate"] = VascularAttentionAdapter(
                    channels[0],
                    reduction=reduction,
                    gamma_init=gamma_init,
                    gamma_mode=gamma_mode,
                    prior_channels=prior_channels,
                    attention=attention,
                    num_heads=vaa_heads,
                    attention_stride=attention_stride,
                    use_mask_bias=use_mask_bias,
                    spatial_dims=self.spatial_dims,
                )
            for adapter in self.vaa_adapters.values():
                adapter.gamma_raw.requires_grad = gamma_trainable

        # For padding to be divisible by 2^(L-1)
        self.pad_factor = 2 ** (L - 1)

        if self.time_cond:
            self.time_embed = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

        if self.label_cond:
            # self.label_emb = LabelEmbedder(self.num_classes, self.emb_channels, class_dropout_prob, labels_embed_magnitude_scale)
            self.label_emb = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

    def apply_vaa(self, location: str, feature: torch.Tensor, mra_prior: torch.Tensor | None) -> torch.Tensor:
        if not self.enable_vaa or location not in self.vaa_adapters:
            return feature
        return self.vaa_adapters[location](feature, mra_prior)

    def forward(self, x, cas_skips=None, timestep=None, label=None, mra_prior=None):
        L = self.L

        # Enforce cas_skips contract if provided
        if cas_skips is not None:
            if L >= 2:
                assert len(cas_skips) == (L - 1), f"cas_skips must have length {L-1} if L >= 2, got {len(cas_skips)}"
            else:
                assert len(cas_skips) == 1, f"cas_skips must have length 1 if L = 1, got {len(cas_skips)}"

        if self.spatial_dims == 3:
            x, padding_sizes = divisible_pad_zy(x, factor=self.pad_factor)
        else:
            x, padding_sizes = divisible_pad_t(x, k=self.pad_factor)

        if timestep is not None and self.time_cond:
            timesteps = torch.tensor(float(timestep) * self.timestep_scale, device=x.device, dtype=x.dtype).expand(
                x.shape[0]
            )
            t_emb = timestep_embedding(timesteps, self.channels[0], repeat_only=False)
            emb = self.time_embed(t_emb)
        else:
            emb = None

        if label is not None and self.label_cond:
            y = torch.tensor(float(label) * self.label_class_scale, device=x.device, dtype=x.dtype).expand(x.shape[0])
            y_emb = timestep_embedding(y, self.channels[0], repeat_only=False)
            emb = emb + self.label_emb(y_emb) if emb is not None else self.label_emb(y_emb)

        # ---- Encoder ----
        feats = [None] * L  # store encoder outputs at each level
        x0 = self.embed_conv(x)
        feats[0] = self.encoders[0](x0, emb=emb)

        for i in range(1, L):
            down_in = self.downs[i - 1](feats[i - 1])
            if cas_skips is not None:
                # cas_skips length is L-1, where cas_skips[i-1] matches level i
                down_in = down_in + cas_skips[i - 1]
            feats[i] = self.encoders[i](down_in, emb=emb)
        feats[-1] = self.apply_vaa("bottleneck", feats[-1], mra_prior)

        # ---- Decoder ----
        if L >= 2:
            x_dec = feats[-1]  # start from bottom level (L-1)
            dec_outs = [None] * (L - 1)  # per-level decoder outputs for i=0..L-2

            # iterate levels from L-2 down to 0
            for i in reversed(range(L - 1)):
                up = self.ups[i](x_dec)  # upsample from level i+1 -> i
                cat = torch.cat([up, feats[i]], dim=1)
                red = self.reduces[i](cat)  # project 2*C_i -> C_i
                x_dec = self.decoders[i](red, emb=emb)  # decode at level i
                dec_outs[i] = x_dec  # store decoder output aligned with level i

            ref_in = self.apply_vaa("intermediate", dec_outs[0], mra_prior)
        else:
            # L == 1: no decoder stages; refine the single encoder output
            dec_outs = []
            ref_in = feats[0] + cas_skips[0] if cas_skips is not None else feats[0]
            ref_in = self.apply_vaa("intermediate", ref_in, mra_prior)

        # ---- Refinement + Output ----
        fr = self.refinement(ref_in, emb=emb)
        out = self.output(fr)

        if self.spatial_dims == 3:
            out = inverse_divisible_pad_zy(out, padding_sizes)
        else:
            out = inverse_divisible_pad_t(out, padding_sizes)

        # ---- Cascade outputs (generalized) ----
        # Return: [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        cas_outs = []
        if L >= 2:
            for i in range(1, L - 1):
                cas_outs.append(dec_outs[i])  # decoder outputs for levels 1..L-2
            cas_outs.append(feats[-1])  # deepest encoder output at level L-1
        else:
            cas_outs.append(fr)  # L == 1: empty cascade outputs

        return out, tuple(cas_outs)


class HybridRestormer(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=2,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        num_spatial_heads=[1, 2, 3, 4],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
    ):
        super(HybridRestormer, self).__init__()
        self.embed_conv = nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        HybridTransformerBlock(
                            num_ch,
                            num_ah,
                            num_sh,
                            expansion_factor,
                            num_experts,
                            top_k,
                            norm_type=norm_type,
                            post_norm=post_norm,
                        )
                        for _ in range(num_tb)
                    ]
                )
                for num_tb, num_ah, num_sh, num_ch in zip(num_blocks, num_heads, num_spatial_heads, channels)
            ]
        )

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(2, len(channels)))
            ]
        )

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        HybridTransformerBlock(
                            channels[2],
                            num_heads[2],
                            num_spatial_heads[2],
                            expansion_factor,
                            num_experts,
                            top_k,
                            drop_path,
                            norm_type=norm_type,
                            post_norm=post_norm,
                        )
                        for _ in range(num_blocks[2])
                    ]
                )
            ]
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    HybridTransformerBlock(
                        channels[1],
                        num_heads[1],
                        num_spatial_heads[2],
                        expansion_factor,
                        num_experts,
                        top_k,
                        drop_path,
                        norm_type=norm_type,
                        post_norm=post_norm,
                    )
                    for _ in range(num_blocks[1])
                ]
            )
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    HybridTransformerBlock(
                        channels[1],
                        num_heads[0],
                        num_spatial_heads[2],
                        expansion_factor,
                        num_experts,
                        top_k,
                        drop_path,
                        norm_type=norm_type,
                        post_norm=post_norm,
                    )
                    for _ in range(num_blocks[0])
                ]
            )
        )

        self.refinement = nn.Sequential(
            *[
                HybridTransformerBlock(
                    channels[1],
                    num_heads[0],
                    num_spatial_heads[2],
                    expansion_factor,
                    num_experts,
                    top_k,
                    norm_type=norm_type,
                    post_norm=post_norm,
                )
                for _ in range(num_refinement)
            ]
        )

        self.output = nn.Conv2d(channels[1], out_channel, kernel_size=3, padding=1, bias=False)
        self.pad_factor = 2 ** (len(channels) - 1) * 8

    def forward(self, x, cas_skips=None):
        x, padding_sizes = divisible_pad_t(x, k=self.pad_factor)
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = (
            self.encoders[1](self.downs[0](out_enc1) + cas_skips[0])
            if cas_skips
            else self.encoders[1](self.downs[0](out_enc1))
        )
        out_enc3 = (
            self.encoders[2](self.downs[1](out_enc2) + cas_skips[1])
            if cas_skips
            else self.encoders[2](self.downs[1](out_enc2))
        )
        out_enc4 = (
            self.encoders[3](self.downs[2](out_enc3) + cas_skips[2])
            if cas_skips
            else self.encoders[3](self.downs[2](out_enc3))
        )

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)  # + x
        out = inverse_divisible_pad_t(out, padding_sizes)
        return out, (out_dec2, out_dec3, out_enc4)


class restormer_mri(nn.Module):
    def __init__(self, args, use_acs_region=None, last_cascade=False, first_cascade=False):
        super().__init__()

        self.args = args
        self.use_csm = args.use_csm
        self.use_external_csm = getattr(args, "use_external_csm", False)
        self.use_acs_region = use_acs_region if use_acs_region is not None else args.use_acs_region
        self.num_frames = int(args.num_frames)
        self.recon_slab = is_slab_recon(args)
        self.num_slices = slab_num_slices(args)
        phase3 = getattr(args, "phase3", None)
        backbone_cfg = getattr(phase3, "backbone", None) if phase3 is not None else None
        self.backbone_spatial_dims = int(getattr(backbone_cfg, "spatial_dims", 2))
        if self.backbone_spatial_dims not in {2, 3}:
            raise ValueError(f"phase3.backbone.spatial_dims must be 2 or 3, got {self.backbone_spatial_dims}")
        if self.backbone_spatial_dims == 3 and not self.recon_slab:
            raise ValueError("The spatial 3D Restormer requires phase3.recon_mode='slab'.")
        self.num_coils = 10
        self.num_reduced_coils = int(args.num_reduced_coils) if hasattr(args, "num_reduced_coils") else 1

        self.mask_types = (
            [m.split("_")[-1] for m in args.train_mask_types_for_def_model]
            if hasattr(args, "train_mask_types_for_def_model")
            else [m.split("_")[-1] for m in args.train_mask_types]
        )
        self.acq_types = (
            args.acq_types
            if hasattr(args, "acq_types")
            else [
                "BlackBlood",
                "Cine",
                "Flow4d",
                "LGE",
                "Mapping",
                "Perfusion",
                "T1rho",
                "T1w",
                "T2w",
                "Aorta",
                "Tagging",
            ]
        )
        self.acc_factors = (
            [int(m) for m in args.accelerations_for_def_model]
            if hasattr(args, "accelerations_for_def_model")
            else [int(m) for m in args.accelerations]
        )

        self.drop_path = args.drop_path if hasattr(args, "drop_path") else 0.0
        self.last_cascade = last_cascade
        self.first_cascade = first_cascade
        self.adaptive_temporal_dc = args.adaptive_temporal_dc if hasattr(args, "adaptive_temporal_dc") else False
        self.gd = args.gd if hasattr(args, "gd") else False
        self.num_experts = args.num_experts if hasattr(args, "num_experts") else None
        self.top_k = args.top_k if hasattr(args, "top_k") else 1
        self.num_blocks = args.num_blocks if hasattr(args, "num_blocks") else [4, 6, 6, 8]
        self.num_heads = args.num_heads if hasattr(args, "num_heads") else [1, 2, 4, 8]
        self.num_refinement = args.num_refinement if hasattr(args, "num_refinement") else 4
        self.channels = args.channels if hasattr(args, "channels") else [48, 96, 192, 384]
        self.mlp_ratio = args.mlp_ratio if hasattr(args, "mlp_ratio") else 4
        self.norm_type = args.norm_type if hasattr(args, "norm_type") else None
        self.hybrid_attn = args.hybrid_attn if hasattr(args, "hybrid_attn") else False
        if self.backbone_spatial_dims == 3 and self.hybrid_attn:
            raise ValueError("HybridRestormer is not implemented for the spatial 3D backbone.")
        self.post_norm = args.post_norm if hasattr(args, "post_norm") else False
        self.mask_specific_dc_weight_map = (
            args.mask_specific_dc_weight_map if hasattr(args, "mask_specific_dc_weight_map") else False
        )
        self.use_single_csm = args.use_single_csm if hasattr(args, "use_single_csm") else False
        self.ms_refinement = args.ms_refinement if hasattr(args, "ms_refinement") else False
        self.time_cond = args.time_cond if hasattr(args, "time_cond") else False
        self.label_cond = args.label_cond if hasattr(args, "label_cond") else False
        self.num_classes = (
            args.num_classes
            if hasattr(args, "num_classes")
            else len(self.mask_types) * len(self.acq_types) * len(self.acc_factors)
        )
        labels_embed_magnitude_scale = (
            args.labels_embed_magnitude_scale if hasattr(args, "labels_embed_magnitude_scale") else 1.0
        )

        if self.label_cond and not hasattr(args, "acq_types"):
            print(
                f"label_cond enabled but without acq types specified in config file, using default acq types: {self.acq_types}"
            )

        restormer = HybridRestormer if self.hybrid_attn else Restormer
        restormer_phase3_kwargs = {} if self.hybrid_attn else {"phase3": getattr(args, "phase3", None)}
        restormer_spatial_kwargs = {} if self.hybrid_attn else {"spatial_dims": self.backbone_spatial_dims}

        mixer_cfg = getattr(phase3, "flowvn_mixer", None) if phase3 is not None else None
        self.enable_flowvn_mixer = bool(getattr(mixer_cfg, "enabled", False)) if mixer_cfg is not None else False
        if self.enable_flowvn_mixer and not self.recon_slab:
            raise ValueError("FlowVN multi-plane mixing requires phase3.recon_mode='slab'.")
        mixer_channels = self.num_reduced_coils if self.use_csm else self.num_coils
        self.flowvn_mixer = (
            FlowVNMultiPlaneMixer(
                in_channels=mixer_channels,
                features=int(getattr(mixer_cfg, "features", 8)),
                kernel_size=int(getattr(mixer_cfg, "kernel_size", 3)),
                branches=list(getattr(mixer_cfg, "branches", ["xyz", "xyt", "yzt", "xzt"])),
                num_knots=int(getattr(mixer_cfg, "num_knots", 71)),
                activation_range=float(getattr(mixer_cfg, "activation_range", 3.5)),
                scale_init=float(getattr(mixer_cfg, "scale_init", 0.0)),
                acceleration_modulation=bool(getattr(mixer_cfg, "acceleration_modulation", True)),
            )
            if self.enable_flowvn_mixer
            else None
        )

        self.pad_factor = 2**2
        if self.use_csm:
            if (self.first_cascade and self.use_single_csm) or not self.use_single_csm:
                self.coil_sensitivity_model = CoilSensitivityModel_DCAE(
                    spatial_dims=2,
                    features=(12, 24, 48, 96, 192),
                    pad_factor=self.pad_factor,
                )
            if self.args.pretrained_csm is not None:
                self.load_csm_model()
            if self.use_external_csm and hasattr(self, "coil_sensitivity_model"):
                for param in self.coil_sensitivity_model.parameters():
                    param.requires_grad = False
            self.recon_model = restormer(
                in_channel=(
                    self.num_frames * self.num_reduced_coils * 2
                    if self.backbone_spatial_dims == 3
                    else self.num_slices * self.num_frames * self.num_reduced_coils * 2
                ),
                out_channel=(
                    self.num_frames * self.num_reduced_coils * 2
                    if self.backbone_spatial_dims == 3
                    else self.num_slices * self.num_frames * self.num_reduced_coils * 2
                ),
                num_experts=self.num_experts,
                top_k=self.top_k,
                expansion_factor=self.mlp_ratio,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                num_refinement=self.num_refinement,
                channels=self.channels,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
                post_norm=self.post_norm,
                ms_refinement=self.ms_refinement,
                time_cond=self.time_cond,
                label_cond=self.label_cond,
                num_classes=self.num_classes,
                labels_embed_magnitude_scale=labels_embed_magnitude_scale,
                **restormer_phase3_kwargs,
                **restormer_spatial_kwargs,
            )
        else:
            self.recon_model = restormer(
                in_channel=(
                    self.num_frames * self.num_coils * 2
                    if self.backbone_spatial_dims == 3
                    else self.num_slices * self.num_frames * self.num_coils * 2
                ),
                out_channel=(
                    self.num_frames * self.num_coils * 2
                    if self.backbone_spatial_dims == 3
                    else self.num_slices * self.num_frames * self.num_coils * 2
                ),
                num_experts=self.num_experts,
                top_k=self.top_k,
                expansion_factor=self.mlp_ratio,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                num_refinement=self.num_refinement,
                channels=self.channels,
                drop_path=self.drop_path,
                norm_type=self.norm_type,
                post_norm=self.post_norm,
                ms_refinement=self.ms_refinement,
                time_cond=self.time_cond,
                label_cond=self.label_cond,
                num_classes=self.num_classes,
                labels_embed_magnitude_scale=labels_embed_magnitude_scale,
                **restormer_phase3_kwargs,
                **restormer_spatial_kwargs,
            )
        self.use_dc_weight_map = args.use_dc_weight_map
        if self.use_dc_weight_map:
            dcwm_h, dcwm_w = (328, 806) if args.dataset.lower() == "cmrxrecon" else (768, 768)
            if self.mask_specific_dc_weight_map:
                self.dc_weight_map = nn.Parameter(
                    torch.ones(len(self.mask_types), self.num_frames, 1, dcwm_h, dcwm_w, 1)
                )  # 640 x 640 should be large enough
            else:
                self.dc_weight_map = nn.Parameter(
                    torch.ones(self.num_frames, 1, dcwm_h, dcwm_w, 1)
                )  # 640 x 640 should be large enough
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))

        if self.args.pretrained_recon is not None:
            self.recon_model = self.load_recon_model(self.recon_model)

    def load_recon_model(self, recon_model):
        try:
            loaded_state_dict = torch.load(self.args.pretrained_recon, map_location="cpu", weights_only=True)
        except BaseException:
            loaded_state_dict = torch.load(self.args.pretrained_recon, map_location="cpu", weights_only=False)[
                "net_state_dict"
            ]
        loaded_keys, _unchanged, skipped_shape_keys, _unexpected, inflated_keys = load_shape_compatible_state_dict(
            recon_model,
            loaded_state_dict,
        )
        print(f"Loaded pretrained Recon model from {self.args.pretrained_recon}")
        print(
            f"Loaded {len(loaded_keys)} keys from pretrained Recon model with {len(loaded_state_dict)} keys \
            from {self.args.pretrained_recon}."
        )
        if inflated_keys:
            print(f"Center-inflated {len(inflated_keys)} Conv2d weights for the spatial 3D Recon model.")
        if skipped_shape_keys:
            preview = ", ".join(key for key, _old_shape, _new_shape in skipped_shape_keys[:8])
            suffix = "..." if len(skipped_shape_keys) > 8 else ""
            print(f"Skipped {len(skipped_shape_keys)} pretrained Recon keys due to shape mismatch: {preview}{suffix}")
        return recon_model

    def load_csm_model(self):
        try:
            csm_state_dict = torch.load(self.args.pretrained_csm, map_location="cpu", weights_only=True)[
                "net_state_dict"
            ]
        except BaseException:
            csm_state_dict = torch.load(self.args.pretrained_csm, map_location="cpu", weights_only=False)[
                "net_state_dict"
            ]
        _, updated_keys, unchanged_keys = copy_model_state(self, csm_state_dict)
        print(f"Loaded pretrained CSM model from {self.args.pretrained_csm}")

    def hard_dc(self, x: torch.Tensor, ref_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kspace_pred = fftn_centered(x, spatial_dims=2)
        ref_kspace = fftn_centered(ref_image, spatial_dims=2)

        kspace_pred = ~mask * kspace_pred + mask * ref_kspace
        return ifftn_centered(kspace_pred, spatial_dims=2)

    def soft_dc(
        self,
        x: torch.Tensor,
        ref_image: torch.Tensor,
        mask: torch.Tensor,
        temporal_weight: torch.Tensor = None,
        mask_type: str = None,
    ) -> torch.Tensor:
        kspace_pred = fftn_centered(x, spatial_dims=2)
        ref_kspace = fftn_centered(ref_image, spatial_dims=2)

        if self.use_dc_weight_map:
            if self.mask_specific_dc_weight_map and mask_type is not None:
                dc_weight_map = self.dc_weight_map[
                    next(
                        (i for i, sub in enumerate(self.mask_types) if sub in mask_type.lower()),
                        -1,
                    )
                ]
            else:
                dc_weight_map = self.dc_weight_map

            _, _, H, W, _ = kspace_pred.shape
            _, _, H_map, W_map, _ = dc_weight_map.shape

            if H_map < H or W_map < W:
                pad_h = max(H - H_map, 0)
                pad_w = max(W - W_map, 0)

                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                weight_map_perm = dc_weight_map.permute(0, 1, 4, 2, 3)

                # Pad (W_left, W_right, H_top, H_bottom)
                weight_map_perm = F.pad(
                    weight_map_perm,
                    (pad_left, pad_right, pad_top, pad_bottom, 0, 0),
                    mode="replicate",
                )

                dc_weight_map = weight_map_perm.permute(0, 1, 3, 4, 2)
                _, _, H_map, W_map, _ = dc_weight_map.shape

            top = (H_map - H) // 2
            left = (W_map - W) // 2
            if temporal_weight:
                weight = (
                    dc_weight_map[None, :, :, top : top + H, left : left + W, :] * temporal_weight[:, : self.num_frames]
                )
                weight = weight.view(weight.shape[0] * weight.shape[1], *weight.shape[2:])
            else:
                weight = dc_weight_map[:, :, top : top + H, left : left + W, :].repeat(
                    int(mask.shape[0] / self.num_frames), 1, 1, 1, 1
                )
        else:
            weight = self.dc_weight

        kspace_pred = ~mask * kspace_pred + mask * ((1 - weight) * kspace_pred + weight * ref_kspace)
        return ifftn_centered(kspace_pred, spatial_dims=2)

    def forward(
        self,
        x: torch.Tensor,
        ref_image: torch.Tensor,
        mask: torch.Tensor = None,
        cas_skips: torch.Tensor = None,
        mask_type: str = None,
        acc_factor: int = None,
        acq_type: str = None,
        sensitivity_maps: torch.Tensor = None,
        timestep: int = None,
        mra_prior: torch.Tensor = None,
    ) -> tuple[Tensor | Any, Any]:
        # slice mode: x shape (B,T,C,H,W,2)
        # slab mode:  x shape (B,S,T,C,H,W,2)
        if self.recon_slab:
            B, S, T, C, H, W, two = x.shape
            if S != self.num_slices:
                raise ValueError(f"Expected slab with S={self.num_slices}, got {S}")
            x = rearrange(x, "b s t c h w two-> (b s t) c h w two")
            ref_image = rearrange(ref_image, "b s t c h w two-> (b s t) c h w two")
            mask = rearrange(mask, "b s t c h w two-> (b s t) c h w two")
        else:
            B, T, C, H, W, two = x.shape
            S = 1
            x = rearrange(x, "b t c h w two-> (b t) c h w two")
            ref_image = rearrange(ref_image, "b t c h w two-> (b t) c h w two")
            mask = rearrange(mask, "b t c h w two-> (b t) c h w two")

        skip = x.clone()

        if self.use_csm:
            if sensitivity_maps is not None:
                if self.recon_slab and sensitivity_maps.dim() == 7:
                    sensitivity_maps = rearrange(sensitivity_maps, "b s t c h w two -> (b s t) c h w two")
                elif sensitivity_maps.dim() == 6:
                    sensitivity_maps = rearrange(sensitivity_maps, "b t c h w two -> (b t) c h w two")
                sensitivity_maps = sensitivity_maps.to(device=x.device, dtype=x.dtype)
            elif self.use_external_csm:
                raise ValueError("use_external_csm=True but sensitivity_maps were not provided by the data loader.")
            elif sensitivity_maps is None or not self.use_single_csm:
                if self.use_acs_region and mask is not None:
                    x_acs = get_acs_image(x, mask)
                    if self.use_single_csm:
                        sensitivity_maps = self.coil_sensitivity_model(x_acs)  # shape (B,C,H,W,2)
                    else:
                        sensitivity_maps = checkpoint(self.coil_sensitivity_model, x_acs, use_reentrant=False)
                else:
                    if self.use_single_csm:
                        sensitivity_maps = self.coil_sensitivity_model(x)  # shape (B,C,H,W,2)
                    else:
                        sensitivity_maps = checkpoint(self.coil_sensitivity_model, x, use_reentrant=False)
            x = sensitivity_map_reduce(x, sensitivity_maps, k=self.num_reduced_coils)  # x will be of shape (B,1,H,W,2)

        flowvn_update = None
        if self.recon_slab:
            x_volume = rearrange(x, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T)
            if self.flowvn_mixer is not None:
                flowvn_update = self.flowvn_mixer(x_volume, acceleration=acc_factor)
            if self.backbone_spatial_dims == 3:
                x = rearrange(x_volume, "b s t c h w two -> b (t c two) s h w", two=2)
            else:
                x = rearrange(x_volume, "b s t c h w two -> b (s t c two) h w", two=2)
        else:
            x = rearrange(x, "(b t) c h w two -> b (t c two) h w", t=T, two=2)
        mask_idx = next((i for i, sub in enumerate(self.mask_types) if sub in mask_type.lower()), -1)
        acc_idx = next((i for i, sub in enumerate(self.acc_factors) if int(sub) == acc_factor), -1)
        if acc_idx == -1:
            acc_idx = min(range(len(self.acc_factors)), key=lambda i: abs(int(self.acc_factors[i]) - int(acc_factor)))
        acq_idx = next(
            (i for i, sub in enumerate(self.acq_types) if sub.lower() == acq_type.lower()),
            -1,
        )
        assert acq_idx != -1, f"acq type {acq_type} not found in {self.acq_types}"
        label = mask_idx * len(self.acc_factors) * len(self.acq_types) + acc_idx * len(self.acq_types) + acq_idx
        if self.hybrid_attn:
            x, cas_skips = self.recon_model(x, cas_skips)
        else:
            x, cas_skips = self.recon_model(x, cas_skips, timestep, label, mra_prior)
        if self.recon_slab:
            if self.backbone_spatial_dims == 3:
                x_volume = rearrange(x, "b (t c two) s h w -> b s t c h w two", t=T, two=2)
            else:
                x_volume = rearrange(x, "b (s t c two) h w -> b s t c h w two", s=S, t=T, two=2)
            if flowvn_update is not None:
                x_volume = x_volume - flowvn_update
            x = rearrange(x_volume, "b s t c h w two -> (b s t) c h w two")
        else:
            x = rearrange(x, "b (t c two) h w -> (b t) c h w two", t=T, two=2)

        if self.last_cascade:
            if self.recon_slab:
                skip = rearrange(skip, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T)[
                    :, :, T // 2 : T // 2 + 1, :, :, :, :
                ].expand(-1, -1, T, -1, -1, -1, -1)
                skip = rearrange(skip, "b s t c h w two-> (b s t) c h w two")
                ref_image = rearrange(ref_image, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T)[
                    :, :, T // 2 : T // 2 + 1, :, :, :, :
                ].expand(-1, -1, T, -1, -1, -1, -1)
                ref_image = rearrange(ref_image, "b s t c h w two-> (b s t) c h w two")
                mask = rearrange(mask, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T)[
                    :, :, T // 2 : T // 2 + 1, :, :, :, :
                ].expand(-1, -1, T, -1, -1, -1, -1)
                mask = rearrange(mask, "b s t c h w two-> (b s t) c h w two")
                if self.use_csm:
                    sensitivity_maps = rearrange(
                        sensitivity_maps, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T
                    )[:, :, T // 2 : T // 2 + 1, :, :, :, :].expand(-1, -1, T, -1, -1, -1, -1)
                    sensitivity_maps = rearrange(sensitivity_maps, "b s t c h w two-> (b s t) c h w two")
            else:
                skip = rearrange(skip, "(b t) c h w two -> b t c h w two", t=T)[:, T // 2 : T // 2 + 1, :, :, :, :].expand(
                    -1, T, -1, -1, -1, -1
                )
                skip = rearrange(skip, "b t c h w two-> (b t) c h w two")
                ref_image = rearrange(ref_image, "(b t) c h w two -> b t c h w two", t=T)[
                    :, T // 2 : T // 2 + 1, :, :, :, :
                ].expand(-1, T, -1, -1, -1, -1)
                ref_image = rearrange(ref_image, "b t c h w two-> (b t) c h w two")
                mask = rearrange(mask, "(b t) c h w two -> b t c h w two", t=T)[:, T // 2 : T // 2 + 1, :, :, :, :].expand(
                    -1, T, -1, -1, -1, -1
                )
                mask = rearrange(mask, "b t c h w two-> (b t) c h w two")
                if self.use_csm:
                    sensitivity_maps = rearrange(sensitivity_maps, "(b t) c h w two -> b t c h w two", t=T)[
                        :, T // 2 : T // 2 + 1, :, :, :, :
                    ].expand(-1, T, -1, -1, -1, -1)
                    sensitivity_maps = rearrange(sensitivity_maps, "b t c h w two-> (b t) c h w two")

        if self.use_csm:
            x = sensitivity_map_expand(x, sensitivity_maps)  # x will be of shape (B,C,H,W,2)
        # x shape: (B,C,H,W,2)

        if not self.gd:
            x = skip + x
            if mask is not None:
                with torch.autocast("cuda", torch.bfloat16, enabled=False):
                    x = self.soft_dc(x.float(), ref_image, mask, mask_type=mask_type)  # (B,C,H,W,2)
        else:
            if mask is not None:
                with torch.autocast("cuda", torch.bfloat16, enabled=False):
                    skip = self.soft_dc(skip.float(), ref_image, mask, mask_type=mask_type)  # (B,C,H,W,2)
            x = skip + x

        if self.recon_slab:
            x = rearrange(x, "(b s t) c h w two -> b s t c h w two", b=B, s=S, t=T)
        else:
            x = rearrange(x, "(b t) c h w two -> b t c h w two", t=T)

        if self.last_cascade:
            if self.recon_slab:
                x = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, T, -1, -1, -1, -1)
            else:
                x = torch.mean(x, dim=1, keepdim=True).expand(-1, T, -1, -1, -1, -1)

        return x, cas_skips, sensitivity_maps
