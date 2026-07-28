# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Depth Anything 3 main model for MLX – depth-only output."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_depth_anything_3.backbone import DinoV2
from mlx_depth_anything_3.camera import CameraDec, CameraEnc
from mlx_depth_anything_3.dpt import DualDPT
from mlx_depth_anything_3.transform import affine_inverse, pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# Model configurations (matching PyTorch YAML configs)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "da3-small": dict(
        backbone=dict(name="vits", out_layers=[5, 7, 9, 11], alt_start=4,
                      qknorm_start=4, rope_start=4, cat_token=True),
        head=dict(dim_in=768, output_dim=2, features=64,
                  out_channels=[48, 96, 192, 384]),
        cam_enc=dict(dim_out=384),
        cam_dec=dict(dim_in=768),
    ),
    "da3-base": dict(
        backbone=dict(name="vitb", out_layers=[5, 7, 9, 11], alt_start=4,
                      qknorm_start=4, rope_start=4, cat_token=True),
        head=dict(dim_in=1536, output_dim=2, features=128,
                  out_channels=[96, 192, 384, 768]),
        cam_enc=dict(dim_out=768),
        cam_dec=dict(dim_in=1536),
    ),
    "da3-large": dict(
        backbone=dict(name="vitl", out_layers=[11, 15, 19, 23], alt_start=8,
                      qknorm_start=8, rope_start=8, cat_token=True),
        head=dict(dim_in=2048, output_dim=2, features=256,
                  out_channels=[256, 512, 1024, 1024]),
        cam_enc=dict(dim_out=1024),
        cam_dec=dict(dim_in=2048),
    ),
    "da3-giant": dict(
        backbone=dict(name="vitg", out_layers=[19, 27, 33, 39], alt_start=13,
                      qknorm_start=13, rope_start=13, cat_token=True),
        head=dict(dim_in=3072, output_dim=2, features=256,
                  out_channels=[256, 512, 1024, 1024]),
        cam_enc=dict(dim_out=1536),
        cam_dec=dict(dim_in=3072),
    ),
}


class DepthAnything3Net(nn.Module):
    """Depth Anything 3 network for depth estimation.

    Supports:
    - Multiple input images (B=1, S=1..N views).
    - Optional extrinsics/intrinsics conditioning.
    - Dynamic spatial resolution (must be multiple of 14).
    - Outputs only predicted depth (and confidence) for each image.
    """

    PATCH_SIZE = 14

    def __init__(self, config_name: str = "da3-small"):
        super().__init__()
        cfg = MODEL_CONFIGS[config_name]

        # Backbone
        bcfg = cfg["backbone"]
        self.backbone = DinoV2(
            name=bcfg["name"],
            out_layers=bcfg["out_layers"],
            alt_start=bcfg["alt_start"],
            qknorm_start=bcfg["qknorm_start"],
            rope_start=bcfg["rope_start"],
            cat_token=bcfg["cat_token"],
        )

        # Depth head
        hcfg = cfg["head"]
        self.head = DualDPT(
            dim_in=hcfg["dim_in"],
            output_dim=hcfg["output_dim"],
            features=hcfg["features"],
            out_channels=hcfg["out_channels"],
        )

        # Camera encoder (for extrinsics/intrinsics conditioning)
        ecfg = cfg["cam_enc"]
        self.cam_enc = CameraEnc(dim_out=ecfg["dim_out"])

        # Camera decoder (for pose prediction; kept for weight compat)
        dcfg = cfg["cam_dec"]
        self.cam_dec = CameraDec(dim_in=dcfg["dim_in"])

    def __call__(
        self,
        x: mx.array,
        extrinsics: mx.array | None = None,
        intrinsics: mx.array | None = None,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, mx.array]:
        """Forward pass – returns predicted depth for each view.

        Args:
            x: (B, S, H, W, 3) input images in NHWC, float32, ImageNet-normalised.
            extrinsics: (B, S, 4, 4) optional world-to-camera matrices.
            intrinsics: (B, S, 3, 3) optional camera intrinsic matrices.
            ref_view_strategy: reference view selection strategy.

        Returns:
            {"depth": (B, S, H, W), "depth_conf": (B, S, H, W)}
        """
        B, S, H, W, C = x.shape

        # Camera encoding
        cam_token = None
        if extrinsics is not None and intrinsics is not None:
            cam_token = self.cam_enc(
                extrinsics.astype(mx.float32),
                intrinsics.astype(mx.float32),
                (H, W),
            )

        # Backbone feature extraction
        feats, _ = self.backbone(
            x, cam_token=cam_token,
            ref_view_strategy=ref_view_strategy,
        )

        # Depth head
        output = self.head(feats, H, W, patch_start_idx=0)

        # Refined camera parameters
        # Take the camera token from the last backbone layer
        cam_token_refined = feats[-1][1]
        pose_encoding = self.cam_dec(cam_token_refined)
        refined_ext, refined_int = pose_encoding_to_extri_intri(pose_encoding, (H, W))

        output["intrinsics"] = refined_int
        output["extrinsics"] = refined_ext

        return output
