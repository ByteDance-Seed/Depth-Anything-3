"""Run DA3METRIC-LARGE on a single image, convert to meters, save outputs + legend.

Usage:
    python scripts/run_metric_depth.py IMAGE_PATH --focal-px 7246 [--out-dir workspace/out]

Notes:
- --focal-px is the focal length in pixels at the resolution of IMAGE_PATH.
  Cropping does NOT change focal_px, but resizing does.
- The script rescales focal_px to match the resolution DA3 actually processes
  (the network typically resizes to 504x504), then applies
      metric_depth_m = focal_px_processed * raw_depth / 300
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

CANONICAL_FOCAL = 300.0
DEFAULT_MODEL = "depth-anything/DA3METRIC-LARGE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--focal-px",
        type=float,
        required=True,
        help="Focal length in pixels at the resolution of the input image",
    )
    parser.add_argument("--out-dir", default="workspace/metric_depth")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with Image.open(args.image) as im:
        input_w, input_h = im.size
    print(f"Input image: {args.image}  ({input_w}x{input_h})")

    device = torch.device(args.device)
    print(f"Loading {args.model} on {device}...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()

    pred = model.inference([args.image])
    raw_depth = pred.depth[0]
    rgb = pred.processed_images[0]
    proc_h, proc_w = raw_depth.shape
    print(f"Processed size: {proc_w}x{proc_h}")

    # focal_px scales linearly with resize (cropping alone would not change it).
    # Use width ratio; model produces square output so height ratio matches.
    focal_px_processed = args.focal_px * (proc_w / input_w)
    print(
        f"focal_px at input res: {args.focal_px:.1f}  "
        f"-> at processed res: {focal_px_processed:.1f}"
    )

    depth_m = focal_px_processed * raw_depth / CANONICAL_FOCAL
    print(
        f"Metric depth (m): min={depth_m.min():.3f}  "
        f"mean={depth_m.mean():.3f}  max={depth_m.max():.3f}"
    )

    np.savez_compressed(
        os.path.join(args.out_dir, "depth.npz"),
        depth_m=depth_m,
        raw_depth=raw_depth,
        focal_px_processed=focal_px_processed,
    )
    Image.fromarray(rgb).save(os.path.join(args.out_dir, "processed_rgb.png"))

    lo, hi = np.percentile(depth_m, 2), np.percentile(depth_m, 98)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(depth_m, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_title("Metric depth (meters)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("depth (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "depth_vis.png"), dpi=150)
    plt.close(fig)

    print(f"Saved outputs to {args.out_dir}/")
    for f in sorted(os.listdir(args.out_dir)):
        size = os.path.getsize(os.path.join(args.out_dir, f))
        print(f"  {f}  ({size} bytes)")


if __name__ == "__main__":
    main()
