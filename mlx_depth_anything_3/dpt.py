# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""DualDPT depth prediction head for MLX.

Only the main (depth) head is implemented; the auxiliary ray head
is omitted since we only export predicted depth.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_depth_anything_3.layers import (
    ModuleList,
    bilinear_interpolate,
    create_uv_grid,
    position_grid_to_embed,
)


# ---------------------------------------------------------------------------
# ResidualConvUnit
# ---------------------------------------------------------------------------
class ResidualConvUnit(nn.Module):
    """Lightweight residual conv block used within fusion."""

    def __init__(self, features: int):
        super().__init__()
        # MLX Conv2d: (N, H, W, C) in/out
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(x)
        out = self.conv1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        return out + x


# ---------------------------------------------------------------------------
# FeatureFusionBlock
# ---------------------------------------------------------------------------
class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: optional residual merge + upsample + 1x1 conv."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, *xs, size=None) -> mx.array:
        """Forward.

        Args:
            xs[0]: top-path input (N, H, W, C).
            xs[1]: optional lateral input.
            size: (target_h, target_w) for upsampling.
        """
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)

        # Upsample
        if size is not None:
            target_h, target_w = size
        else:
            target_h = y.shape[1] * 2
            target_w = y.shape[2] * 2
        y = bilinear_interpolate(y, target_h, target_w, align_corners=True)
        y = self.out_conv(y)
        return y


# ---------------------------------------------------------------------------
# Permute helper (for Sequential with LayerNorm on channel dim)
# ---------------------------------------------------------------------------
class PermuteToChannelsLast(nn.Module):
    """(N, C, H, W) -> no-op since MLX already uses NHWC."""
    def __call__(self, x: mx.array) -> mx.array:
        return x


# ---------------------------------------------------------------------------
# DualDPT – depth-only variant
# ---------------------------------------------------------------------------
class DualDPT(nn.Module):
    """Dual-head DPT for depth prediction.

    Only the main depth head is computed (aux ray head is skipped).

    All convolutions work in NHWC (channels-last) layout.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: list[int] | tuple[int, ...] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.output_dim = output_dim

        # Token pre-norm + per-stage 1x1 projection
        self.norm = nn.LayerNorm(dim_in, eps=1e-6)
        self.projects = ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        # Spatial resize layers (NHWC)
        # Stage 0: 4x upsample via ConvTranspose
        # Stage 1: 2x upsample via ConvTranspose
        # Stage 2: identity
        # Stage 3: 2x downsample via Conv stride=2
        self.resize_0 = nn.ConvTranspose2d(out_channels[0], out_channels[0],
                                            kernel_size=4, stride=4, padding=0)
        self.resize_1 = nn.ConvTranspose2d(out_channels[1], out_channels[1],
                                            kernel_size=2, stride=2, padding=0)
        # resize_2 = identity
        self.resize_3 = nn.Conv2d(out_channels[3], out_channels[3],
                                   kernel_size=3, stride=2, padding=1)

        # Scratch: layer adapters (3x3 conv, no bias)
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3,
                                    stride=1, padding=1, bias=False)

        # Main fusion chain
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.refinenet3 = FeatureFusionBlock(features, has_residual=True)
        self.refinenet2 = FeatureFusionBlock(features, has_residual=True)
        self.refinenet1 = FeatureFusionBlock(features, has_residual=True)

        # Main output head: conv1 (3x3) -> conv2 (sequence: 3x3 + relu + 1x1)
        head_features_1 = features
        head_features_2 = 32
        self.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2,
                                       kernel_size=3, stride=1, padding=1)
        # output_conv2 sequence
        self.output_conv2_a = nn.Conv2d(head_features_1 // 2, head_features_2,
                                         kernel_size=3, stride=1, padding=1)
        self.output_conv2_b = nn.Conv2d(head_features_2, output_dim,
                                         kernel_size=1, stride=1, padding=0)

        # Auxiliary fusion chain (loaded for weight compatibility only; not used in forward)
        self.refinenet4_aux = FeatureFusionBlock(features, has_residual=False)
        self.refinenet3_aux = FeatureFusionBlock(features, has_residual=True)
        self.refinenet2_aux = FeatureFusionBlock(features, has_residual=True)
        self.refinenet1_aux = FeatureFusionBlock(features, has_residual=True)

        # Aux output layers (loaded for weight compatibility only; not used in forward)
        self.output_conv1_aux = ModuleList([
            self._make_aux_out1_module(head_features_1)
            for _ in range(4)
        ])

        # Aux output conv2 per level (with LN)
        self.output_conv2_aux = ModuleList([
            self._make_aux_out2_module(head_features_1 // 2, head_features_2)
            for _ in range(4)
        ])

    def _make_aux_out1_module(self, in_ch: int):
        """5-conv auxiliary pre-head stack."""
        class AuxOut1(nn.Module):
            def __init__(self, in_ch):
                super().__init__()
                self.layers = [
                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
                ]
            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        return AuxOut1(in_ch)

    def _make_aux_out2_module(self, in_ch: int, mid_ch: int):
        """Aux final projection: conv3x3 -> LN -> ReLU -> conv1x1."""
        class AuxOut2(nn.Module):
            def __init__(self, in_ch, mid_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)
                self.ln = nn.LayerNorm(mid_ch)
                self.conv2 = nn.Conv2d(mid_ch, 7, kernel_size=1, stride=1, padding=0)
            def __call__(self, x):
                return self.conv2(nn.relu(self.ln(self.conv1(x))))
        return AuxOut2(in_ch, mid_ch)

    def __call__(
        self,
        feats: list,
        H: int,
        W: int,
        patch_start_idx: int = 0,
    ) -> dict[str, mx.array]:
        """Forward pass.

        Args:
            feats: list of 4 (features, camera_tokens) tuples.
                   features: (B, S, N, C).
            H, W: original image dimensions.
            patch_start_idx: index where patch tokens start in the sequence.

        Returns:
            {"depth": (B, S, H, W), "depth_conf": (B, S, H, W)}
        """
        B, S, N, C = feats[0][0].shape
        feat_list = [feat[0] for feat in feats]
        # Flatten B*S
        feat_list = [mx.reshape(f, (B * S, N, C)) for f in feat_list]

        out_dict = self._forward_impl(feat_list, H, W, patch_start_idx)
        # Reshape back to (B, S, ...)
        out_dict = {k: mx.reshape(v, (B, S, *v.shape[1:])) for k, v in out_dict.items()}
        return out_dict

    def _forward_impl(
        self,
        feats: list[mx.array],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> dict[str, mx.array]:
        BS = feats[0].shape[0]
        C = feats[0].shape[2]
        ph = H // self.patch_size
        pw = W // self.patch_size

        resized_feats = []
        for stage_idx in range(4):
            x = feats[stage_idx][:, patch_start_idx:]  # (BS, N_patch, C)
            x = self.norm(x)
            # Reshape to spatial NHWC: (BS, ph, pw, C)
            x = mx.reshape(x, (BS, ph, pw, C))
            # 1x1 projection
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            # Resize
            if stage_idx == 0:
                x = self.resize_0(x)
            elif stage_idx == 1:
                x = self.resize_1(x)
            elif stage_idx == 2:
                pass  # identity
            else:
                x = self.resize_3(x)
            resized_feats.append(x)

        # Fuse pyramid (main path only)
        fused_main = self._fuse_main(resized_feats)

        # Upsample to target resolution
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = bilinear_interpolate(fused_main, h_out, w_out, align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        # Main head: conv2 sequence
        main_logits = self.output_conv2_a(fused_main)  # (BS, H, W, 32)
        main_logits = nn.relu(main_logits)
        main_logits = self.output_conv2_b(main_logits)  # (BS, H, W, output_dim)

        # Activation
        depth = self._apply_activation(main_logits[..., :-1], self.activation)
        conf = self._apply_activation(main_logits[..., -1], self.conf_activation)
        # depth: (BS, H, W) or (BS, H, W, 1) -> squeeze
        depth = mx.squeeze(depth, axis=-1) if depth.ndim == 4 else depth

        return {"depth": depth, "depth_conf": conf}

    def _fuse_main(self, feats: list[mx.array]) -> mx.array:
        """Main fusion pyramid."""
        l1, l2, l3, l4 = feats

        l1_rn = self.layer1_rn(l1)
        l2_rn = self.layer2_rn(l2)
        l3_rn = self.layer3_rn(l3)
        l4_rn = self.layer4_rn(l4)

        # 4 -> 3 -> 2 -> 1
        out = self.refinenet4(l4_rn, size=(l3_rn.shape[1], l3_rn.shape[2]))
        out = self.refinenet3(out, l3_rn, size=(l2_rn.shape[1], l2_rn.shape[2]))
        out = self.refinenet2(out, l2_rn, size=(l1_rn.shape[1], l1_rn.shape[2]))
        out = self.refinenet1(out, l1_rn)

        out = self.output_conv1(out)
        return out

    def _add_pos_embed(self, x: mx.array, W: int, H: int,
                       ratio: float = 0.1) -> mx.array:
        """Add UV positional embedding to feature map (NHWC)."""
        ph, pw = x.shape[1], x.shape[2]
        C = x.shape[3]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype)
        pe = position_grid_to_embed(pe, C) * ratio  # (ph, pw, C)
        pe = mx.expand_dims(pe, 0)  # (1, ph, pw, C)
        return x + pe

    @staticmethod
    def _apply_activation(x: mx.array, activation: str) -> mx.array:
        act = activation.lower()
        if act == "exp":
            return mx.exp(x)
        if act == "expp1":
            return mx.exp(x) + 1.0
        if act == "expm1":
            return mx.exp(x) - 1.0
        if act == "relu":
            return nn.relu(x)
        if act == "sigmoid":
            return mx.sigmoid(x)
        if act == "softplus":
            return mx.log(1.0 + mx.exp(x))
        if act == "tanh":
            return mx.tanh(x)
        return x  # linear
