#!/usr/bin/env python3
"""
Batch Image Processing Example

This example demonstrates efficient processing of multiple images with memory management.

Usage:
    python examples/batch_processing.py --images-dir folder/
    python examples/batch_processing.py --images-dir folder/ --batch-size 4
    python examples/batch_processing.py --images-dir folder/ --mixed-precision fp16
    python examples/batch_processing.py --pattern "*.jpg" --batch-size 2
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser(
        description="Batch image processing example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="assets/examples/SOH",
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="File pattern to match (e.g., '*.jpg', '*.png')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process at once (controls memory usage)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32", "bf16"],
        help="Precision mode (fp16 saves memory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_processing",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="da3-small",
        choices=["da3-small", "da3-base", "da3-large"],
        help="Model size",
    )
    args = parser.parse_args()

    # Find all images
    pattern = str(Path(args.images_dir) / args.pattern)
    image_paths = sorted(glob.glob(pattern))

    if not image_paths:
        print(f"❌ No images found matching pattern: {pattern}")
        print(f"   Current directory: {Path.cwd()}")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model with memory-optimized settings
    print(f"Loading model: {args.model}...")
    model_name_map = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
    }

    model = DepthAnything3.from_pretrained(
        model_name_map[args.model],
        batch_size=args.batch_size,  # Sub-batching for memory control
        mixed_precision=None if args.mixed_precision == "auto" else args.mixed_precision,
    )
    model = model.to(device)
    print("Model loaded!")

    # Process images with progress bar
    print("\nProcessing images...")
    start_time = time.time()

    all_depths = []
    all_images = []

    # Process in batches (model handles sub-batching internally)
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Batches"):
        batch_paths = image_paths[i : i + args.batch_size]

        # Run inference on batch
        prediction = model.inference(batch_paths)

        # Store results
        for j, depth in enumerate(prediction.depth):
            all_depths.append(depth)
            all_images.append(prediction.processed_images[j])

            # Save individual depth map
            img_name = Path(batch_paths[j]).stem
            depth_path = output_dir / f"{img_name}_depth.npz"
            np.savez_compressed(depth_path, depth=depth)

            # Save visualization
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            vis_path = output_dir / f"{img_name}_depth.png"
            Image.fromarray(depth_vis).save(vis_path)

    elapsed = time.time() - start_time

    # Print statistics
    print(f"\n✅ Processing complete!")
    print(f"   Total images: {len(image_paths)}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {len(image_paths) / elapsed:.2f} images/sec")
    print(f"   Results saved to: {output_dir}")

    # Save batch summary
    summary_path = output_dir / "batch_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Mixed precision: {args.mixed_precision}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Processing time: {elapsed:.2f}s\n")
        f.write(f"Speed: {len(image_paths) / elapsed:.2f} images/sec\n\n")
        f.write(f"Depth statistics:\n")
        for i, depth in enumerate(all_depths):
            img_name = Path(image_paths[i]).stem
            f.write(
                f"  {img_name}: min={depth.min():.2f}, max={depth.max():.2f}, "
                f"mean={depth.mean():.2f}\n"
            )

    print(f"   Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
