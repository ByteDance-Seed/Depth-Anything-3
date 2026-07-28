# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Camera encoder and decoder modules for MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_depth_anything_3.layers import CamBlock, Mlp, ModuleList
from mlx_depth_anything_3.transform import (
    affine_inverse,
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)


class CameraEnc(nn.Module):
    """Camera encoder: convert extrinsics+intrinsics to conditioning tokens.

    Uses an MLP to project the 9-D pose encoding to token space, then
    refines via transformer blocks.
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            bias=True,
        )
        self.token_norm = nn.LayerNorm(dim_out, eps=1e-6)
        self.trunk = ModuleList([
            CamBlock(dim=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     init_values=init_values)
            for _ in range(trunk_depth)
        ])
        self.trunk_norm = nn.LayerNorm(dim_out, eps=1e-6)

    def __call__(
        self,
        ext: mx.array,
        ixt: mx.array,
        image_size: tuple[int, int],
    ) -> mx.array:
        """Encode camera parameters to conditioning tokens.

        Args:
            ext: (B, S, 4, 4) world-to-camera extrinsics.
            ixt: (B, S, 3, 3) intrinsics.
            image_size: (H, W).

        Returns:
            (B, S, dim_out) camera tokens.
        """
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, ixt, image_size)
        tokens = self.pose_branch(pose_encoding)
        tokens = self.token_norm(tokens)
        for blk in self.trunk:
            tokens = blk(tokens)
        tokens = self.trunk_norm(tokens)
        return tokens


class CameraDec(nn.Module):
    """Camera decoder: predict pose encoding from features.

    Not used for depth-only inference, but included for weight loading.
    """

    def __init__(self, dim_in: int = 1536, **kwargs):
        super().__init__()
        self.backbone_fc1 = nn.Linear(dim_in, dim_in)
        self.backbone_fc2 = nn.Linear(dim_in, dim_in)
        self.fc_t = nn.Linear(dim_in, 3)
        self.fc_qvec = nn.Linear(dim_in, 4)
        self.fc_fov_linear = nn.Linear(dim_in, 2)

    def __call__(self, feat: mx.array) -> mx.array:
        """Predict 9-D pose encoding from features.

        Args:
            feat: (B, S, D) features.

        Returns:
            (B, S, 9) pose encoding [T(3), quat(4), fov(2)].
        """
        B, S = feat.shape[:2]
        x = mx.reshape(feat, (B * S, -1))
        x = nn.relu(self.backbone_fc1(x))
        x = nn.relu(self.backbone_fc2(x))
        x_f = x.astype(mx.float32)
        t = mx.reshape(self.fc_t(x_f), (B, S, 3))
        qvec = mx.reshape(self.fc_qvec(x_f), (B, S, 4))
        fov = nn.relu(mx.reshape(self.fc_fov_linear(x_f), (B, S, 2)))
        return mx.concatenate([t, qvec, fov], axis=-1)
