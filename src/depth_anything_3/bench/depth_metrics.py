"""depth_anything_3.bench.depth_metrics

Depth / inverse-depth quality metrics used by the VGB (Visual Geometry Benchmark).

This module is intended to back the two new evaluator modes:

* **metric_depth**: evaluate *metric* depth quality directly in meters (no alignment).
* **rel_depth**: evaluate *relative* depth quality by removing global scale and/or
  affine ambiguity (scale or scale+shift), then scoring the aligned predictions.

Why both?

* Multi-view systems often produce depth that is *accurate up to a global scale*
  (or sometimes scale+shift, depending on representation).
* Systems claiming *metric depth* should also be measured without any alignment,
  and their *scale accuracy* should be reported separately ("Metric Scale rel"),
  because a method can have good relative depth but poor absolute scale.

Metric definitions (per-pixel on valid GT depth):

* **abs_rel**:
    mean( |d̂ - d| / d )
  This matches the "Reld" (relative depth error) used in MoGe-2/MapAnything.
* **delta@t**:
    100 * mean( max(d̂/d, d/d̂) < t )
  This matches the δ depth inlier metric used in many depth benchmarks and in
  MoGe-2 / moge_eval (t=1.25 by default there). MapAnything additionally reports
  a tighter threshold around t=1.03 ("Depth inliers @ 1.03").

Alignments (for rel_depth or auxiliary reporting in metric_depth):

* **median scale alignment** ("_scale_med"): scale s = median(d)/median(d̂),
  then d̂' = s·d̂.
* **affine depth alignment** ("_affine_depth"): find (a,b) minimizing a·d̂ + b ≈ d
  (weighted least-squares with w=1/d to better match relative error).
* **affine disparity alignment** ("_affine_disp"): find (a,b) minimizing
  a·(1/d̂)+b ≈ (1/d), then convert back to depth.
  This follows the disparity-affine alignment described in MoGe-2 Appendix A.3.

Code references:

* MoGe-2 evaluation implementation (alignment + depth metrics):
  - moge_eval/utils/metrics.py (rel_depth, delta metrics)
  - moge_eval/utils/alignment.py (scale / affine / disparity-affine alignments)
* MapAnything uses median scale alignment to make scale-invariant metrics.

This file is self-contained (NumPy + OpenCV) so that evaluator can import it
without pulling training-time dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class DepthEvalConfig:
    """Configuration for depth evaluation."""

    min_depth: float = 1e-3
    max_depth: float = float("inf")
    delta_thresholds: Tuple[float, float] = (1.03, 1.25)
    clamp_min: float = 1e-6


def resize_depth_nearest(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a single-channel depth map to (H,W) using nearest-neighbor."""
    h, w = out_hw
    if depth.shape[0] == h and depth.shape[1] == w:
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def _valid_mask(
    pred: np.ndarray,
    gt: np.ndarray,
    gt_valid: np.ndarray,
    cfg: DepthEvalConfig,
) -> np.ndarray:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    m = np.asarray(gt_valid).astype(bool)
    m &= np.isfinite(pred) & np.isfinite(gt)
    m &= gt > cfg.min_depth
    if np.isfinite(cfg.max_depth):
        m &= gt < cfg.max_depth
    m &= pred > cfg.clamp_min
    return m


def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, EPS)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def rmse_log(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.maximum(pred, EPS)
    gt = np.maximum(gt, EPS)
    return float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))


def si_log(pred: np.ndarray, gt: np.ndarray) -> float:
    """Scale-invariant log RMSE (Eigen et al.)."""
    pred = np.maximum(pred, EPS)
    gt = np.maximum(gt, EPS)
    diff = np.log(pred) - np.log(gt)
    return float(np.sqrt(np.mean(diff**2) - (np.mean(diff) ** 2)))


def delta(pred: np.ndarray, gt: np.ndarray, thr: float) -> float:
    pred = np.maximum(pred, EPS)
    gt = np.maximum(gt, EPS)
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < thr) * 100.0)


def median_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    """Robust global scale s so that median(s·pred) == median(gt)."""
    p_med = float(np.median(pred))
    g_med = float(np.median(gt))
    return g_med / max(p_med, EPS)


def fit_affine_l2(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Fit y ≈ a·x + b by (weighted) least squares."""
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    if w is None:
        A = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        return a, b

    w = w.reshape(-1).astype(np.float64)
    w = np.maximum(w, 0.0)
    # Weighted normal equations: (A^T W A) θ = A^T W y
    A0 = x
    A1 = np.ones_like(x)
    # Compute components explicitly to avoid huge matrices.
    Sw = w.sum() + EPS
    Sx = (w * A0).sum()
    Sy = (w * y).sum()
    Sxx = (w * A0 * A0).sum()
    Sxy = (w * A0 * y).sum()
    # Solve 2x2:
    # [Sxx Sx] [a] = [Sxy]
    # [Sx  Sw] [b]   [Sy ]
    det = Sxx * Sw - Sx * Sx
    if abs(det) < 1e-12:
        return 1.0, 0.0
    a = (Sxy * Sw - Sy * Sx) / det
    b = (Sxx * Sy - Sx * Sxy) / det
    return float(a), float(b)


def apply_affine_depth(pred: np.ndarray, a: float, b: float, clamp_min: float = 1e-6) -> np.ndarray:
    out = a * pred + b
    return np.maximum(out, clamp_min)


def apply_affine_disp(pred: np.ndarray, a: float, b: float, clamp_min: float = 1e-6) -> np.ndarray:
    disp = 1.0 / np.maximum(pred, clamp_min)
    disp2 = a * disp + b
    disp2 = np.maximum(disp2, clamp_min)
    return 1.0 / disp2


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_valid_mask: np.ndarray,
    cfg: DepthEvalConfig = DepthEvalConfig(),
) -> Dict[str, float]:
    """Compute a bundle of depth metrics for a single image."""
    pred = np.asarray(pred_depth).astype(np.float32)
    gt = np.asarray(gt_depth).astype(np.float32)
    valid = _valid_mask(pred, gt, gt_valid_mask, cfg)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {}

    p = pred[valid]
    g = gt[valid]

    out: Dict[str, float] = {}
    # Percentage of pixels used for evaluation after applying:
    # (1) dataset GT valid mask, (2) GT depth range checks, (3) finite/positive checks.
    # Useful for sparse GT depth (e.g., ETH3D); clarifies this is the eval mask (GT∩pred).
    out["valid_pixels_pct"] = float(100.0 * n_valid / float(max(valid.size, 1)))

    # --- Raw / metric metrics (no alignment) ---
    out["abs_rel"] = abs_rel(p, g)
    out["rmse"] = rmse(p, g)
    out["rmse_log"] = rmse_log(p, g)
    out["si_log"] = si_log(p, g)
    for t in cfg.delta_thresholds:
        out[f"delta_{t}"] = delta(p, g, t)

    # --- Median scale alignment (relative depth quality) ---
    s_med = median_scale(p, g)
    out["scale_med"] = float(s_med)
    out["metric_scale_rel"] = float(abs(s_med - 1.0))
    out["metric_scale_log"] = float(abs(np.log(max(s_med, EPS))))

    p_med = p * s_med
    out["abs_rel_scale_med"] = abs_rel(p_med, g)
    for t in cfg.delta_thresholds:
        out[f"delta_{t}_scale_med"] = delta(p_med, g, t)

    # --- Affine depth alignment (scale+shift invariance) ---
    w = 1.0 / np.maximum(g, EPS)  # match relative error emphasis
    a_d, b_d = fit_affine_l2(p, g, w=w)
    out["affine_depth_scale"] = float(a_d)
    out["affine_depth_shift"] = float(b_d)
    p_aff_d = apply_affine_depth(p, a_d, b_d, clamp_min=cfg.clamp_min)
    out["abs_rel_affine_depth"] = abs_rel(p_aff_d, g)
    for t in cfg.delta_thresholds:
        out[f"delta_{t}_affine_depth"] = delta(p_aff_d, g, t)

    # --- Affine disparity alignment (scale+shift invariance in inverse depth) ---
    p_disp = 1.0 / np.maximum(p, cfg.clamp_min)
    g_disp = 1.0 / np.maximum(g, cfg.clamp_min)
    a_id, b_id = fit_affine_l2(p_disp, g_disp, w=None)
    out["affine_disp_scale"] = float(a_id)
    out["affine_disp_shift"] = float(b_id)
    p_aff_id = apply_affine_disp(p, a_id, b_id, clamp_min=cfg.clamp_min)
    out["abs_rel_affine_disp"] = abs_rel(p_aff_id, g)
    for t in cfg.delta_thresholds:
        out[f"delta_{t}_affine_disp"] = delta(p_aff_id, g, t)

    return out
