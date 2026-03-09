# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""High-level inference API for Depth Anything 3 on MLX."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_depth_anything_3.model import DepthAnything3Net


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthAnything3:
    """Depth Anything 3 inference wrapper for MLX.

    Supports:
    - Multiple input images (variable number of views).
    - Optional extrinsics/intrinsics conditioning.
    - Dynamic spatial resolution (any multiple of 14).
    - Returns predicted depth for each input image.

    Example::

        da3 = DepthAnything3("da3-small")
        da3.load_weights("weights.safetensors")
        images = [np.array(img1), np.array(img2)]  # uint8 or float32, HWC
        result = da3.predict(images)
        depth_maps = result["depth"]  # (N, H, W) numpy array
    """

    def __init__(self, config_name: str = "da3-small"):
        self.config_name = config_name
        self.model = DepthAnything3Net(config_name)

    def load_weights(self, path: str | Path) -> None:
        """Load model weights from a safetensors or npz file."""
        path = str(path)
        if path.endswith(".safetensors"):
            weights = mx.load(path)
        elif path.endswith(".npz"):
            weights = dict(mx.load(path))
        else:
            raise ValueError(f"Unsupported weight format: {path}")

        # The weight keys use dot-separated paths matching the model structure
        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())

    def predict(
        self,
        images: list[np.ndarray],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, np.ndarray]:
        """Run depth prediction on a list of images.

        Args:
            images: list of N images as numpy arrays (H, W, 3) uint8 or float32.
            extrinsics: optional (N, 4, 4) world-to-camera matrices.
            intrinsics: optional (N, 3, 3) camera intrinsic matrices.
            process_res: processing resolution (must be multiple of 14).
            ref_view_strategy: reference view selection strategy.

        Returns:
            {"depth": (N, H, W) numpy float32 array}
        """
        # Preprocess images
        processed = self._preprocess_images(images, process_res)
        x = mx.array(processed)  # (1, N, H, W, 3)

        # Prepare camera matrices
        ext_mx = None
        int_mx = None
        if extrinsics is not None:
            ext_np = extrinsics.astype(np.float32)
            if ext_np.ndim == 3:
                ext_np = ext_np[np.newaxis]  # (1, N, 4, 4)
            ext_mx = mx.array(ext_np)
        if intrinsics is not None:
            int_np = intrinsics.astype(np.float32)
            if int_np.ndim == 3:
                int_np = int_np[np.newaxis]  # (1, N, 3, 3)
            int_mx = mx.array(int_np)

        # Forward pass
        output = self.model(x, extrinsics=ext_mx, intrinsics=int_mx,
                           ref_view_strategy=ref_view_strategy)
        mx.eval(output["depth"])

        # Convert to numpy
        depth = np.array(output["depth"][0])  # (N, H, W)
        return {"depth": depth}

    def _preprocess_images(
        self,
        images: list[np.ndarray],
        process_res: int,
    ) -> np.ndarray:
        """Preprocess a list of images to model input format.

        Args:
            images: list of (H, W, 3) images.
            process_res: target resolution (must be multiple of 14).

        Returns:
            (1, N, H, W, 3) float32 array, ImageNet-normalised, NHWC.
        """
        assert process_res % 14 == 0, f"process_res must be multiple of 14, got {process_res}"

        processed = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            # Resize to process_res x process_res
            img = self._resize_image(img, process_res, process_res)
            # ImageNet normalise
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            processed.append(img)

        stacked = np.stack(processed, axis=0)  # (N, H, W, 3)
        return stacked[np.newaxis]  # (1, N, H, W, 3)

    @staticmethod
    def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize image using bilinear interpolation (numpy only, no OpenCV dependency)."""
        h, w = img.shape[:2]
        if h == target_h and w == target_w:
            return img

        # Simple bilinear resize
        y_ratio = h / target_h
        x_ratio = w / target_w

        y_coords = np.arange(target_h).astype(np.float32) * y_ratio
        x_coords = np.arange(target_w).astype(np.float32) * x_ratio

        y0 = np.clip(np.floor(y_coords).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(np.floor(x_coords).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)

        wy = y_coords - y0.astype(np.float32)
        wx = x_coords - x0.astype(np.float32)

        top_left = img[y0][:, x0]
        top_right = img[y0][:, x1]
        bot_left = img[y1][:, x0]
        bot_right = img[y1][:, x1]

        wy = wy[:, np.newaxis, np.newaxis]
        wx = wx[np.newaxis, :, np.newaxis]

        result = (top_left * (1 - wy) * (1 - wx) +
                  top_right * (1 - wy) * wx +
                  bot_left * wy * (1 - wx) +
                  bot_right * wy * wx)
        return result.astype(np.float32)
