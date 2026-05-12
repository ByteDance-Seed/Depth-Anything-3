# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
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

"""Numpy geometry helpers used by the ONNX inference path.

Re-exports the shared helpers from `utils.numpy_geometry` and adds the
inference-only `normalize_extrinsics_np`.
"""

from __future__ import annotations

import numpy as np

from depth_anything_3.utils.numpy_geometry import (
    affine_inverse_np,
    transpose_last_two_axes,
)


def as_homogeneous_np(ext: np.ndarray) -> np.ndarray:
    """(..., 3, 4) or (..., 4, 4) extrinsics -> (..., 4, 4) homogeneous."""
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        pad = np.zeros((*ext.shape[:-2], 1, 4), dtype=ext.dtype)
        pad[..., 0, 3] = 1.0
        return np.concatenate([ext, pad], axis=-2)
    raise ValueError(f"as_homogeneous_np: invalid shape {ext.shape}")


def normalize_extrinsics_np(ex: np.ndarray | None) -> np.ndarray | None:
    """Median-translation normalization, mirrors ``DepthAnything3._normalize_extrinsics``.

    ``ex`` has shape (B, N, 4, 4). Returns the normalized extrinsics in the
    same shape, anchored so that the first view is the identity and translations
    are scaled by the median camera distance from that anchor.
    """
    if ex is None:
        return None
    transform = affine_inverse_np(ex[:, :1])
    ex_norm = ex @ transform
    c2ws = affine_inverse_np(ex_norm)
    translations = c2ws[..., :3, 3]
    dists = np.linalg.norm(translations, axis=-1)
    median_dist = np.median(dists)
    median_dist = max(median_dist, 1e-1)
    ex_norm = ex_norm.copy()
    ex_norm[..., :3, 3] = ex_norm[..., :3, 3] / median_dist
    return ex_norm


__all__ = [
    "affine_inverse_np",
    "as_homogeneous_np",
    "normalize_extrinsics_np",
    "transpose_last_two_axes",
]
