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

"""NumPy-only post-processing.

Ports the alignment and sky-handling code that lives inside
`NestedDepthAnything3Net.forward` and `DepthAnything3Net._process_mono_sky_estimation`
so that it can run on numpy arrays after onnxruntime has produced the raw
predictions. Keeping this out of the ONNX graph also dodges several
`torch.quantile` / `torch.randint`-on-empty-tensor edge cases that don't
ONNX-export cleanly.
"""

from __future__ import annotations

import numpy as np

# Default copied from `NestedDepthAnything3Net._apply_metric_scaling`.
_METRIC_SCALE_FACTOR = 300.0


def compute_sky_mask(sky: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """True where the pixel is NOT sky (mirrors the torch helper)."""
    return sky < threshold


def least_squares_scale_scalar(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Scalar least-squares scale s such that ``a ≈ s * b``."""
    a_f = a.reshape(-1).astype(np.float64)
    b_f = b.reshape(-1).astype(np.float64)
    num = float(np.dot(a_f, b_f))
    den = max(float(np.dot(b_f, b_f)), eps)
    return num / den


def apply_metric_scaling(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Scale a metric-branch depth map by ``focal / 300``.

    ``depth``      (B, N, H, W) or (B, N, 1, H, W)
    ``intrinsics`` (B, N, 3, 3)
    """
    focal = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2.0
    scale = focal / _METRIC_SCALE_FACTOR
    return depth * scale[:, :, None, None]


def set_sky_regions_to_max_depth(
    depth: np.ndarray,
    depth_conf: np.ndarray | None,
    non_sky_mask: np.ndarray,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Set sky pixels to ``max_depth`` and (if conf is given) bump them to 1.0."""
    depth = depth.copy()
    depth[~non_sky_mask] = max_depth
    if depth_conf is None:
        return depth, None
    depth_conf = depth_conf.copy()
    depth_conf[~non_sky_mask] = 1.0
    return depth, depth_conf


def _safe_quantile(a: np.ndarray, q: float, max_samples: int = 100_000) -> float:
    """Bounded-memory quantile, mirroring `sample_tensor_for_quantile`."""
    flat = a.reshape(-1)
    if flat.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, flat.size, size=max_samples)
        flat = flat[idx]
    return float(np.quantile(flat, q))


def process_mono_sky(
    depth: np.ndarray,
    sky: np.ndarray | None,
    threshold: float = 0.3,
) -> np.ndarray:
    """Mirror of `DepthAnything3Net._process_mono_sky_estimation`.

    ``depth`` shape (N, H, W). ``sky`` (N, H, W) or None.
    """
    if sky is None:
        return depth
    non_sky = compute_sky_mask(sky, threshold)
    if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
        return depth
    non_sky_depth = depth[non_sky]
    max_depth = _safe_quantile(non_sky_depth, 0.99)
    out, _ = set_sky_regions_to_max_depth(depth, None, non_sky, max_depth=max_depth)
    return out


def apply_nested_metric_alignment(
    *,
    depth: np.ndarray,  # (B, N, H, W)  -- main-branch
    depth_conf: np.ndarray,  # (B, N, H, W)
    extrinsics: np.ndarray,  # (B, N, 4, 4)
    intrinsics: np.ndarray,  # (B, N, 3, 3)
    metric_depth: np.ndarray,  # (B, N, H, W)  -- metric-branch raw
    metric_sky: np.ndarray,  # (B, N, H, W)  -- metric-branch sky probabilities
    sky_depth_cap: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """NumPy version of NestedDepthAnything3Net._apply_metric_scaling +
    _apply_depth_alignment + _handle_sky_regions.

    Returns ``(depth_aligned, depth_conf_aligned, extrinsics_scaled, scale_factor)``.
    """
    # 1) Metric scaling using camera intrinsics from the main branch
    metric_depth = apply_metric_scaling(metric_depth, intrinsics)

    # 2) Depth alignment via least-squares scale on confident, non-sky pixels
    non_sky = compute_sky_mask(metric_sky, threshold=0.3)
    if non_sky.sum() <= 10:
        raise RuntimeError("Insufficient non-sky pixels for nested metric alignment.")

    depth_conf_ns = depth_conf[non_sky]
    median_conf = _safe_quantile(depth_conf_ns, 0.5)
    align_mask = (
        (depth_conf >= median_conf)
        & non_sky
        & (metric_depth > 1e-2)
        & (depth > 1e-3)
    )
    if align_mask.sum() == 0:
        raise RuntimeError("Empty alignment mask for nested metric alignment.")

    scale_factor = least_squares_scale_scalar(
        metric_depth[align_mask], depth[align_mask]
    )

    depth_aligned = depth * scale_factor
    extrinsics_scaled = extrinsics.copy()
    extrinsics_scaled[..., :3, 3] *= scale_factor

    # 3) Sky -> capped max depth
    non_sky_depth = depth_aligned[non_sky]
    non_sky_max = min(_safe_quantile(non_sky_depth, 0.99), sky_depth_cap)
    depth_final, depth_conf_final = set_sky_regions_to_max_depth(
        depth_aligned, depth_conf, non_sky, max_depth=non_sky_max
    )
    return depth_final, depth_conf_final, extrinsics_scaled, float(scale_factor)


def unletterbox_depth(
    depth: np.ndarray,
    infos: list,                       # list[PreprocessInfo]
    interpolation: str = "linear",
) -> list[np.ndarray]:
    """Invert the preprocessor transform for each image.

    For every image:
      1) crop ``depth[i]`` to the unpadded ``scaled_size`` rectangle
         starting at ``(pad_top, pad_left)``;
      2) resize that crop to the input's original ``(H, W)``.

    Works for ``letterbox`` (real pad), ``square_resize`` (no pad), and
    the legacy ``upper_bound_resize`` family (no pad, canvas already
    matches the scaled size).

    Returns a list of per-image arrays (each at its own ``(H_i, W_i)``);
    callers stack them when shapes agree.
    """
    import cv2

    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    assert depth.ndim == 3, f"expected (N, H, W) depth, got {depth.shape}"
    assert len(infos) == depth.shape[0], (
        f"infos ({len(infos)}) must match batch ({depth.shape[0]})"
    )

    interp = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation]

    out = []
    for i, info in enumerate(infos):
        sh, sw = info.scaled_size
        cropped = depth[i, info.pad_top : info.pad_top + sh, info.pad_left : info.pad_left + sw]
        oh, ow = info.original_size
        out.append(cv2.resize(cropped, (ow, oh), interpolation=interp))
    return out


def unletterbox_mask(
    mask: np.ndarray,
    infos: list,                       # list[PreprocessInfo]
) -> list[np.ndarray]:
    """Nearest-neighbour version of `unletterbox_depth` for boolean / class
    masks. Keeps the input dtype."""
    import cv2

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    assert mask.ndim == 3, f"expected (N, H, W) mask, got {mask.shape}"

    out = []
    for i, info in enumerate(infos):
        sh, sw = info.scaled_size
        cropped = mask[i, info.pad_top : info.pad_top + sh, info.pad_left : info.pad_left + sw]
        m = cropped.astype(np.uint8)
        oh, ow = info.original_size
        m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
        out.append(m.astype(mask.dtype))
    return out


__all__ = [
    "apply_metric_scaling",
    "apply_nested_metric_alignment",
    "compute_sky_mask",
    "least_squares_scale_scalar",
    "process_mono_sky",
    "set_sky_regions_to_max_depth",
    "unletterbox_depth",
    "unletterbox_mask",
]
