#!/usr/bin/env python3
"""
Single Image Depth Estimation Example

This example demonstrates basic depth estimation from a single image.

Usage:
    python examples/single_image.py
    python examples/single_image.py --image path/to/image.jpg
    python examples/single_image.py --image image.jpg --output-dir results/
    python examples/single_image.py --model da3-small  # Use smaller model
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser(
        description="Single image depth estimation example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/examples/SOH/00.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="da3-small",
        choices=["da3-small", "da3-base", "da3-large", "da3-giant"],
        help="Model size to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/single_image",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detect if not specified.",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model}...")
    model_name_map = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
        "da3-giant": "depth-anything/DA3-GIANT",
    }

    model = DepthAnything3.from_pretrained(model_name_map[args.model])
    model = model.to(device)
    print("Model loaded successfully!")

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        print("Using default example image instead...")
        args.image = "assets/examples/SOH/00.png"

    # Run inference
    print(f"Processing image: {args.image}")
    prediction = model.inference([args.image])

    # Extract results
    depth = prediction.depth[0]  # First (and only) image
    processed_image = prediction.processed_images[0]

    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")

    # Save depth map as NPZ
    npz_path = output_dir / "depth.npz"
    np.savez(npz_path, depth=depth)
    print(f"Saved depth map to: {npz_path}")

    # Save depth visualization
    # Normalize depth to 0-255 for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_normalized * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_vis)
    depth_vis_path = output_dir / "depth_visualization.png"
    depth_img.save(depth_vis_path)
    print(f"Saved depth visualization to: {depth_vis_path}")

    # Save processed input image
    input_img = Image.fromarray(processed_image)
    input_path = output_dir / "input_image.png"
    input_img.save(input_path)
    print(f"Saved processed input to: {input_path}")

    # Create side-by-side comparison
    comparison = Image.new("RGB", (processed_image.shape[1] * 2, processed_image.shape[0]))
    comparison.paste(input_img, (0, 0))
    # Convert grayscale depth to RGB for concatenation
    depth_rgb = Image.fromarray(depth_vis).convert("RGB")
    comparison.paste(depth_rgb, (processed_image.shape[1], 0))
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)
    print(f"Saved comparison to: {comparison_path}")

    print("\n✅ Done! Results saved to:", output_dir)
    print("\nFiles created:")
    print(f"  - depth.npz (raw depth values)")
    print(f"  - depth_visualization.png (grayscale visualization)")
    print(f"  - input_image.png (processed input)")
    print(f"  - comparison.png (side-by-side)")


if __name__ == "__main__":
    main()
