# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""Core neural network layers for DA3 in MLX."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# ModuleList helper
# ---------------------------------------------------------------------------
class ModuleList(nn.Module):
    """Simple wrapper to register a list of modules in MLX."""

    def __init__(self, modules: list[nn.Module]):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)
        self._len = len(modules)

    def __getitem__(self, i: int) -> nn.Module:
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError("ModuleList index out of range")
        return getattr(self, str(i))

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for i in range(self._len):
            yield self[i]


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------
class LayerScale(nn.Module):
    """Learnable per-channel scaling."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


# ---------------------------------------------------------------------------
# Mlp (GELU-based 2-layer MLP)
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: str = "gelu",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self._act = act_layer

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = _apply_act(x, self._act)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x12 = self.w12(x)
        half = x12.shape[-1] // 2
        x1 = x12[..., :half]
        x2 = x12[..., half:]
        hidden = nn.silu(x1) * x2
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# 2-D Rotary Position Embedding
# ---------------------------------------------------------------------------
class PositionGetter:
    """Generates and caches 2-D spatial positions for patches."""

    def __init__(self):
        self._cache: dict[tuple[int, int], mx.array] = {}

    def __call__(self, batch_size: int, height: int, width: int) -> mx.array:
        key = (height, width)
        if key not in self._cache:
            y = mx.arange(height)
            x = mx.arange(width)
            # cartesian product: (H*W, 2) with (y, x)
            yy = mx.repeat(y, width)
            xx = mx.tile(x, (height,))
            self._cache[key] = mx.stack([yy, xx], axis=-1)
        cached = self._cache[key]
        return mx.broadcast_to(mx.expand_dims(cached, 0), (batch_size, height * width, 2))


class RotaryPositionEmbedding2D(nn.Module):
    """2-D RoPE that applies separate 1-D rotary embeddings for vertical/horizontal."""

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self._freq_cache: dict = {}

    def _get_freq(self, dim: int, seq_len: int, dtype):
        key = (dim, seq_len, dtype)
        if key not in self._freq_cache:
            exponents = mx.arange(0, dim, 2).astype(mx.float32) / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = mx.arange(seq_len).astype(mx.float32)
            angles = mx.expand_dims(positions, 1) * mx.expand_dims(inv_freq, 0)
            angles = mx.concatenate([angles, angles], axis=-1).astype(dtype)
            self._freq_cache[key] = (mx.cos(angles), mx.sin(angles))
        return self._freq_cache[key]

    @staticmethod
    def _rotate(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_1d(self, tokens, positions, cos_comp, sin_comp):
        # positions: (B, N) int indices
        cos = cos_comp[positions][:, None, :, :]  # (B, 1, N, D)
        sin = sin_comp[positions][:, None, :, :]
        return tokens * cos + self._rotate(tokens) * sin

    def __call__(self, tokens: mx.array, positions: mx.array) -> mx.array:
        """Apply 2D RoPE.

        Args:
            tokens: (B, heads, N, dim)
            positions: (B, N, 2) with [y, x] coords
        """
        feat_dim = tokens.shape[-1] // 2
        max_pos = int(mx.max(positions).item()) + 1
        cos_comp, sin_comp = self._get_freq(feat_dim, max_pos, tokens.dtype)

        vert = tokens[..., :feat_dim]
        horiz = tokens[..., feat_dim:]

        pos_y = positions[..., 0].astype(mx.int32)
        pos_x = positions[..., 1].astype(mx.int32)

        vert = self._apply_1d(vert, pos_y, cos_comp, sin_comp)
        horiz = self._apply_1d(horiz, pos_x, cos_comp, sin_comp)
        return mx.concatenate([vert, horiz], axis=-1)


# ---------------------------------------------------------------------------
# Multi-head Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = False,
        rope: RotaryPositionEmbedding2D | None = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def __call__(self, x: mx.array, pos: mx.array | None = None,
                 attn_mask: mx.array | None = None) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = mx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ mx.transpose(k, axes=(0, 1, 3, 2))  # (B, heads, N, N)
        if attn_mask is not None:
            # attn_mask: (B, N, N) boolean. True = attend, False = block.
            mask = mx.expand_dims(attn_mask, 1)
            mask = mx.broadcast_to(mask, attn.shape)
            attn = attn + mx.where(mask, mx.zeros_like(attn), mx.full(attn.shape, -1e9))
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # (B, heads, N, head_dim)
        out = mx.transpose(out, axes=(0, 2, 1, 3))  # (B, N, heads, head_dim)
        out = mx.reshape(out, (B, N, C))
        out = self.proj(out)
        return out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float | None = None,
        ffn_layer: str = "mlp",
        qk_norm: bool = False,
        rope: RotaryPositionEmbedding2D | None = None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_bias=proj_bias, qk_norm=qk_norm, rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else None
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        if ffn_layer == "swiglu":
            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden,
                                 out_features=dim, bias=ffn_bias)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden,
                           out_features=dim, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else None

    def __call__(self, x: mx.array, pos: mx.array | None = None,
                 attn_mask: mx.array | None = None) -> mx.array:
        attn_out = self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask)
        if self.ls1 is not None:
            attn_out = self.ls1(attn_out)
        x = x + attn_out
        ffn_out = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            ffn_out = self.ls2(ffn_out)
        x = x + ffn_out
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """2-D image to patch embedding: (B, H, W, C) -> (B, N, D).

    Note: MLX Conv2d uses channels-last (NHWC) format.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        grid_h = img_size // self.patch_size[0]
        grid_w = img_size // self.patch_size[1]
        self.num_patches = grid_h * grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size[0], stride=self.patch_size[0])

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) in NHWC
        x = self.proj(x)  # (B, H/p, W/p, D)
        B = x.shape[0]
        x = mx.reshape(x, (B, -1, x.shape[-1]))  # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# Camera Encoder Attention + Block (simpler variant from model/utils/)
# ---------------------------------------------------------------------------
class CamAttention(nn.Module):
    """Attention for the camera encoder (simpler: always uses scaled_dot_product)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 proj_bias: bool = True, qk_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        q = q * self.scale
        attn = q @ mx.transpose(k, axes=(0, 1, 3, 2))
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = mx.reshape(out, (B, N, C))
        return self.proj(out)


class CamBlock(nn.Module):
    """Transformer block for the camera encoder (no drop-path)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 init_values: float | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CamAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.ls1 = LayerScale(dim, init_values) if init_values else None
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden, out_features=dim, bias=True)
        self.ls2 = LayerScale(dim, init_values) if init_values else None

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(self.norm1(x))
        if self.ls1 is not None:
            attn_out = self.ls1(attn_out)
        x = x + attn_out
        ffn_out = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            ffn_out = self.ls2(ffn_out)
        x = x + ffn_out
        return x


# ---------------------------------------------------------------------------
# Positional embedding helpers (used in DPT head)
# ---------------------------------------------------------------------------
def create_uv_grid(width: int, height: int, aspect_ratio: float | None = None,
                   dtype=mx.float32) -> mx.array:
    """Create normalized UV grid (H, W, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    sx = aspect_ratio / diag
    sy = 1.0 / diag
    lx = -sx * (width - 1) / width
    rx = sx * (width - 1) / width
    ty = -sy * (height - 1) / height
    by = sy * (height - 1) / height
    xs = mx.linspace(lx, rx, width).astype(dtype)
    ys = mx.linspace(ty, by, height).astype(dtype)
    # meshgrid: (W,) x (H,) -> (H, W)
    xx = mx.broadcast_to(mx.expand_dims(xs, 0), (height, width))
    yy = mx.broadcast_to(mx.expand_dims(ys, 1), (height, width))
    return mx.stack([xx, yy], axis=-1)


def make_sincos_pos_embed(embed_dim: int, pos: mx.array,
                          omega_0: float = 100.0) -> mx.array:
    """1-D sinusoidal positional embedding."""
    omega = mx.arange(embed_dim // 2).astype(mx.float32) / (embed_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)
    pos_flat = mx.reshape(pos, (-1,)).astype(mx.float32)
    out = mx.expand_dims(pos_flat, 1) * mx.expand_dims(omega, 0)
    return mx.concatenate([mx.sin(out), mx.cos(out)], axis=1).astype(mx.float32)


def position_grid_to_embed(pos_grid: mx.array, embed_dim: int,
                           omega_0: float = 100.0) -> mx.array:
    """Convert 2D position grid (H, W, 2) to sinusoidal embeddings (H, W, C)."""
    H, W, _ = pos_grid.shape
    pos_flat = mx.reshape(pos_grid, (-1, 2))
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)
    emb = mx.concatenate([emb_x, emb_y], axis=-1)
    return mx.reshape(emb, (H, W, embed_dim))


# ---------------------------------------------------------------------------
# Bilinear interpolation helper
# ---------------------------------------------------------------------------
def bilinear_interpolate(x: mx.array, target_h: int, target_w: int,
                         align_corners: bool = True) -> mx.array:
    """Bilinear interpolation for NHWC tensors.

    Args:
        x: (N, H, W, C) input.
        target_h, target_w: target spatial dimensions.

    Returns:
        (N, target_h, target_w, C) interpolated tensor.
    """
    N, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x

    if align_corners and target_h > 1 and target_w > 1:
        y_coords = mx.linspace(0, H - 1, target_h).astype(mx.float32)
        x_coords = mx.linspace(0, W - 1, target_w).astype(mx.float32)
    else:
        y_coords = (mx.arange(target_h).astype(mx.float32) + 0.5) * H / target_h - 0.5
        x_coords = (mx.arange(target_w).astype(mx.float32) + 0.5) * W / target_w - 0.5

    y_coords = mx.clip(y_coords, 0, H - 1)
    x_coords = mx.clip(x_coords, 0, W - 1)

    y0 = mx.floor(y_coords).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H - 1)
    x0 = mx.floor(x_coords).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W - 1)

    wy = y_coords - y0.astype(mx.float32)
    wx = x_coords - x0.astype(mx.float32)

    # (target_h, target_w) grid indices
    # Gather corners: x[:, y0[i], x0[j], :]
    top_left = x[:, y0][:, :, x0]       # (N, target_h, target_w, C)
    top_right = x[:, y0][:, :, x1]
    bot_left = x[:, y1][:, :, x0]
    bot_right = x[:, y1][:, :, x1]

    wy = mx.reshape(wy, (1, target_h, 1, 1))
    wx = mx.reshape(wx, (1, 1, target_w, 1))

    out = (top_left * (1 - wy) * (1 - wx) +
           top_right * (1 - wy) * wx +
           bot_left * wy * (1 - wx) +
           bot_right * wy * wx)
    return out


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------
def _apply_act(x: mx.array, act: str = "gelu") -> mx.array:
    if act == "gelu":
        return nn.gelu(x)
    elif act == "relu":
        return nn.relu(x)
    elif act == "silu":
        return nn.silu(x)
    return x
