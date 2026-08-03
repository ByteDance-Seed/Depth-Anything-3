"""Sanity-check: verify the GT depth convention matches what the pretrained
DA3METRIC-LARGE produces out of the box.

Runs a monocular forward on one dataset sample and compares the predicted
meters (after the canonical focal-px rescaling from scripts/run_metric_depth.py)
to the GT meters pixel-wise. If the median ratio is within 25% of 1.0, the
training unit convention is correct and the main training loop can proceed.

Usage:
    python -m scripts.finetune.sanity_check [--index N]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

from .config import FinetuneConfig


CANONICAL_FOCAL = 300.0


def _load_sample(cfg: FinetuneConfig, index: int) -> tuple[str, np.ndarray]:
    depths_dir = cfg.data_root / cfg.train_split / "depths"
    images_dir = cfg.data_root / cfg.train_split / "images"
    candidates = sorted(depths_dir.glob("*.npy"))
    chosen_stem = None
    chosen_depth = None
    seen = 0
    for p in candidates:
        d = np.load(p)
        if d.max() <= 0:
            continue
        if not (images_dir / f"{p.stem}.jpeg").is_file():
            continue
        if seen == index:
            chosen_stem = p.stem
            chosen_depth = d
            break
        seen += 1
    if chosen_stem is None:
        raise RuntimeError(f"Could not find a non-empty sample at index {index}")
    return chosen_stem, chosen_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0,
                        help="Which non-empty sample to use (0-based)")
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = FinetuneConfig()
    stem, gt_depth_orig = _load_sample(cfg, args.index)
    img_path = cfg.data_root / cfg.train_split / "images" / f"{stem}.jpeg"
    print(f"[sanity] sample: {stem}")
    print(f"[sanity] gt (256x256)  min={gt_depth_orig.min():.3f} "
          f"max={gt_depth_orig.max():.3f} "
          f"median(valid)={np.median(gt_depth_orig[gt_depth_orig > 0]):.3f}")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()
    pred = model.inference([str(img_path)])
    raw_depth = pred.depth[0]  # (proc_h, proc_w), pre-scale
    proc_h, proc_w = raw_depth.shape
    print(f"[sanity] processed shape: {proc_w}x{proc_h}")

    focal_px_processed = cfg.dataset_focal_at_orig * (proc_w / cfg.orig_res)
    pred_m = focal_px_processed * raw_depth / CANONICAL_FOCAL
    print(
        f"[sanity] focal_px input={cfg.dataset_focal_at_orig:.1f} "
        f"processed={focal_px_processed:.1f}"
    )
    print(f"[sanity] pred_m min={pred_m.min():.3f} max={pred_m.max():.3f} "
          f"median={np.median(pred_m):.3f}")

    # Resize GT to processed shape for per-pixel comparison
    gt_resized = cv2.resize(
        gt_depth_orig.astype(np.float32), (proc_w, proc_h), interpolation=cv2.INTER_LINEAR
    )
    valid_orig = (gt_depth_orig > 0).astype(np.uint8)
    valid_resized = cv2.resize(valid_orig, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid_resized &= (gt_resized > 0) & np.isfinite(gt_resized) & (pred_m > 0) & np.isfinite(pred_m)

    ratios = pred_m[valid_resized] / gt_resized[valid_resized]
    median_ratio = float(np.median(ratios))
    print(f"[sanity] n_valid_px={valid_resized.sum()}  "
          f"median(pred/gt)={median_ratio:.3f}")
    log_err = abs(math.log(median_ratio)) if median_ratio > 0 else math.inf
    threshold = math.log(1.25)
    if log_err < threshold:
        print(f"[sanity] PASS  |log(median_ratio)|={log_err:.3f} < {threshold:.3f}")
    else:
        print(
            f"[sanity] FAIL  |log(median_ratio)|={log_err:.3f} >= {threshold:.3f}. "
            f"The GT convention probably doesn't match focal={cfg.dataset_focal_at_orig} "
            f"at {cfg.orig_res}x{cfg.orig_res}. Investigate before training."
        )


if __name__ == "__main__":
    main()
