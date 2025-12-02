# Copyright (c) 2025 Delanoe Pirard
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPU-accelerated input processor using Kornia.

This processor eliminates CPU→GPU transfers by performing all preprocessing
operations directly on GPU using Kornia ops (resize, crop, normalize).
Falls back to CPU-based InputProcessor if GPU is unavailable.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import cv2
import kornia
import kornia.geometry.transform as K
import numpy as np
import torch
from PIL import Image

from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.parallel_utils import parallel_execution


class GPUInputProcessor(InputProcessor):
    """GPU-accelerated preprocessing using Kornia.

    Performs all preprocessing operations on GPU to eliminate CPU→GPU transfer overhead.
    Inherits from InputProcessor and overrides key methods with GPU implementations.

    Key differences:
    - Loads images to GPU immediately after loading
    - Uses Kornia for resize/crop/normalize on GPU
    - Only transfers final batch to CPU if needed

    Fallback:
    - Automatically falls back to CPU processing if GPU unavailable
    - Detects device from first tensor in batch
    """

    def __init__(self, device: str | torch.device | None = None):
        """Initialize GPU processor.

        Args:
            device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect).
                   If None, uses cuda if available, else mps if available, else cpu.

        Note:
            On MPS (Apple Silicon), GPU preprocessing is automatically disabled because
            CPU preprocessing is faster due to optimized OpenCV/Accelerate routines.
            The GPU will still be used for model inference where it provides real gains.
        """
        super().__init__()
        self._device = self._resolve_device(device)

        # MPS: Force CPU preprocessing (benchmarks show CPU is faster on Apple Silicon)
        # The overhead of MPS kernel launches + synchronization exceeds the benefit.
        # GPU should be reserved for model inference where it provides 5-10x speedup.
        if self._device.type == "mps":
            self._use_gpu = False
            logger.info(
                "MPS detected: GPU preprocessing disabled (CPU is faster on Apple Silicon). "
                "GPU will be used for model inference only."
            )
        elif self._device.type == "cuda":
            self._use_gpu = True
            logger.info(f"GPUInputProcessor initialized with device=cuda (NVJPEG enabled)")
        else:
            self._use_gpu = False
            logger.warn(
                f"GPUInputProcessor initialized with device={self._device.type}. "
                "GPU preprocessing disabled. Consider using InputProcessor instead."
            )

        # Pre-create Kornia normalize transform on GPU
        if self._use_gpu:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
            self._kornia_mean = mean
            self._kornia_std = std

    # -----------------------------
    # Device management
    # -----------------------------
    def _resolve_device(self, device: str | torch.device | None) -> torch.device:
        """Resolve device string/object to torch.device."""
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    @property
    def use_gpu(self) -> bool:
        """Whether GPU preprocessing is enabled."""
        return self._use_gpu

    # -----------------------------
    # Override: _process_one (GPU path)
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
        perform_normalization: bool = True,
    ) -> tuple[torch.Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        """Process single image with GPU acceleration.

        If GPU is enabled, performs all operations on GPU.
        Otherwise falls back to CPU path from parent class.
        """
        if not self._use_gpu:
            # Fallback to CPU
            return super()._process_one(
                img, 
                extrinsic, 
                intrinsic, 
                process_res=process_res, 
                process_res_method=process_res_method,
                perform_normalization=perform_normalization
            )

        orig_w, orig_h = 0, 0
        img_tensor = None

        # Try GPU/Accelerated decoding if input is a file path and device is CUDA or MPS
        if isinstance(img, str) and self._device.type in ("cuda", "mps"):
            import torchvision.io
            import os
            
            try:
                # Read raw bytes from file
                with open(img, "rb") as f:
                    # Read bytes -> numpy array (uint8) -> torch tensor
                    file_bytes = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8))
                
                ext = os.path.splitext(img)[1].lower()
                
                # 1. CUDA Optimized Path (NVJPEG)
                if self._device.type == "cuda" and ext in (".jpg", ".jpeg"):
                    img_tensor = torchvision.io.decode_jpeg(file_bytes, device=self._device)
                
                # 2. Generic Path (MPS or non-JPG on CUDA)
                # decode_image is generally faster than PIL for loading into tensors
                else:
                    if ext in (".png", ".jpg", ".jpeg"):
                         # decode_image supports many formats
                         # We move to device immediately after decoding
                         img_tensor = torchvision.io.decode_image(img).to(self._device)
                    else:
                         # Fallback for exotic formats
                         img_tensor = None

                if img_tensor is not None:
                    # Ensure (1, 3, H, W) float32 [0, 1]
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0) # Add batch dim
                    
                    _, c, h, w = img_tensor.shape
                    orig_h, orig_w = h, w
                    
                    # Handle RGBA or Grayscale
                    if c == 4:
                        img_tensor = img_tensor[:, :3, :, :]
                    elif c == 1:
                        img_tensor = img_tensor.repeat(1, 3, 1, 1)
                    
                    img_tensor = img_tensor.float() / 255.0
                
            except Exception as e:
                logger.warn(f"Accelerated decoding failed for {img} on {self._device}, falling back to PIL: {e}")
                img_tensor = None

        # Fallback to PIL loading if GPU decoding failed or not applicable
        if img_tensor is None:
            pil_img = self._load_image(img)
            orig_w, orig_h = pil_img.size
            img_tensor = self._pil_to_tensor_gpu(pil_img)

        # Boundary resize (on GPU)
        img_tensor = self._resize_image_gpu(img_tensor, process_res, process_res_method)
        _, _, h_resized, w_resized = img_tensor.shape
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w_resized, h_resized)

        # Enforce divisibility by PATCH_SIZE (on GPU)
        if process_res_method.endswith("resize"):
            img_tensor = self._make_divisible_by_resize_gpu(img_tensor, self.PATCH_SIZE)
            _, _, h_final, w_final = img_tensor.shape
            intrinsic = self._resize_ixt(intrinsic, w_resized, h_resized, w_final, h_final)
        elif process_res_method.endswith("crop"):
            img_tensor = self._make_divisible_by_crop_gpu(img_tensor, self.PATCH_SIZE)
            _, _, h_final, w_final = img_tensor.shape
            intrinsic = self._crop_ixt(intrinsic, w_resized, h_resized, w_final, h_final)
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Normalize (on GPU) if requested
        if perform_normalization:
            img_tensor = self._normalize_image_gpu(img_tensor)

        # Remove batch dimension: (1, 3, H, W) → (3, H, W)
        img_tensor = img_tensor.squeeze(0)

        return img_tensor, (h_final, w_final), intrinsic, extrinsic

    # -----------------------------
    # GPU preprocessing ops (Kornia)
    # -----------------------------
    def _pil_to_tensor_gpu(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to GPU tensor (1, 3, H, W) float32 [0,1]."""
        # PIL → numpy → torch → GPU
        arr = np.array(img)  # (H, W, 3) uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
        tensor = tensor.unsqueeze(0).to(self._device)  # (1, 3, H, W) on GPU
        return tensor

    def _normalize_image_gpu(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization on GPU.

        Args:
            img_tensor: (1, 3, H, W) tensor on GPU

        Returns:
            Normalized tensor (1, 3, H, W)
        """
        # Manual normalization: (x - mean) / std
        return (img_tensor - self._kornia_mean) / self._kornia_std

    def _resize_image_gpu(
        self, img_tensor: torch.Tensor, target_size: int, method: str
    ) -> torch.Tensor:
        """Resize image tensor on GPU.

        Args:
            img_tensor: (1, 3, H, W) tensor on GPU
            target_size: target size for longest/shortest side
            method: resize method string

        Returns:
            Resized tensor (1, 3, H', W')
        """
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side_gpu(img_tensor, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side_gpu(img_tensor, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side_gpu(
        self, img_tensor: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """Resize so longest side = target_size (preserving aspect ratio)."""
        _, _, h, w = img_tensor.shape
        longest = max(w, h)
        if longest == target_size:
            return img_tensor

        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Kornia resize with antialiasing
        # Note: MPS doesn't fully support 'area' mode, use 'bilinear' instead
        if self._device.type == "mps":
            mode = "bilinear"
        else:
            mode = "bilinear" if scale > 1.0 else "area"
        return K.resize(img_tensor, (new_h, new_w), interpolation=mode, antialias=True)

    def _resize_shortest_side_gpu(
        self, img_tensor: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """Resize so shortest side = target_size (preserving aspect ratio)."""
        _, _, h, w = img_tensor.shape
        shortest = min(w, h)
        if shortest == target_size:
            return img_tensor

        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Note: MPS doesn't fully support 'area' mode, use 'bilinear' instead
        if self._device.type == "mps":
            mode = "bilinear"
        else:
            mode = "bilinear" if scale > 1.0 else "area"
        return K.resize(img_tensor, (new_h, new_w), interpolation=mode, antialias=True)

    def _make_divisible_by_crop_gpu(
        self, img_tensor: torch.Tensor, patch: int
    ) -> torch.Tensor:
        """Floor each dimension to nearest multiple of PATCH_SIZE via center crop (GPU)."""
        _, _, h, w = img_tensor.shape
        new_h = (h // patch) * patch
        new_w = (w // patch) * patch
        if new_h == h and new_w == w:
            return img_tensor

        # Kornia center_crop
        return K.center_crop(img_tensor, (new_h, new_w))

    def _make_divisible_by_resize_gpu(
        self, img_tensor: torch.Tensor, patch: int
    ) -> torch.Tensor:
        """Round each dimension to nearest multiple of PATCH_SIZE via resize (GPU)."""
        _, _, h, w = img_tensor.shape

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_h = max(patch, nearest_multiple(h, patch))
        new_w = max(patch, nearest_multiple(w, patch))
        if new_h == h and new_w == w:
            return img_tensor

        # Note: MPS doesn't fully support 'area' mode, use 'bilinear' instead
        if self._device.type == "mps":
            mode = "bilinear"
        else:
            upscale = (new_h > h) or (new_w > w)
            mode = "bilinear" if upscale else "area"
        return K.resize(img_tensor, (new_h, new_w), interpolation=mode, antialias=True)

    # -----------------------------
    # Override: _unify_batch_shapes (GPU version)
    # -----------------------------
    def _unify_batch_shapes(
        self,
        processed_images: list[torch.Tensor],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all tensors to smallest H, W on GPU."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)
        logger.warn(
            f"Images in batch have different sizes {out_sizes}; "
            f"center-cropping all to smallest ({min_h},{min_w})"
        )

        new_imgs, new_sizes, new_ixts = [], [], []
        for img_t, (H, W), K_np in zip(processed_images, out_sizes, out_intrinsics):
            if H == min_h and W == min_w:
                new_imgs.append(img_t)
                new_sizes.append((min_h, min_w))
                new_ixts.append(K_np)
                continue

            # Crop on GPU using Kornia
            # img_t is (3, H, W), need (1, 3, H, W) for Kornia
            img_t_4d = img_t.unsqueeze(0)
            cropped = K.center_crop(img_t_4d, (min_h, min_w))
            new_imgs.append(cropped.squeeze(0))
            new_sizes.append((min_h, min_w))

            # Adjust intrinsics
            if K_np is None:
                new_ixts.append(None)
            else:
                crop_top = max(0, (H - min_h) // 2)
                crop_left = max(0, (W - min_w) // 2)
                K_adj = K_np.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)

        return new_imgs, new_sizes, new_ixts


# Backward compatibility alias
GPUInputAdapter = GPUInputProcessor
