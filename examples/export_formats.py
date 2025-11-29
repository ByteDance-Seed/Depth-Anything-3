#!/usr/bin/env python3
"""
Export Formats Example

This example demonstrates how to export depth estimation results in various formats.

Usage:
    python examples/export_formats.py --format glb
    python examples/export_formats.py --format ply
    python examples/export_formats.py --format npz
    python examples/export_formats.py --format all  # Export all formats
"""

import argparse
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser(
        description="Export formats example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=["assets/examples/SOH/00.png", "assets/examples/SOH/01.png"],
        help="Paths to input images",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["glb", "ply", "npz", "colmap", "all"],
        help="Export format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/export_formats",
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

    # Load model
    print(f"Loading model: {args.model}...")
    model_name_map = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
    }

    model = DepthAnything3.from_pretrained(model_name_map[args.model])
    model = model.to(device)
    print("Model loaded!")

    # Determine formats to export
    if args.format == "all":
        formats = ["npz", "glb"]
    else:
        formats = [args.format]

    print(f"\nProcessing {len(args.images)} images...")
    print(f"Export formats: {', '.join(formats)}")

    # Process for each format
    for fmt in formats:
        print(f"\n{'='*60}")
        print(f"Exporting as: {fmt.upper()}")
        print(f"{'='*60}")

        format_dir = output_dir / fmt
        format_dir.mkdir(exist_ok=True)

        if fmt == "npz":
            # NPZ: NumPy arrays (depth, confidence, intrinsics, extrinsics)
            print("NPZ format contains:")
            print("  - depth: (N, H, W) float32 array")
            print("  - conf: (N, H, W) float32 confidence map")
            print("  - intrinsics: (N, 3, 3) camera intrinsics")
            print("  - extrinsics: (N, 3, 4) camera extrinsics")

            prediction = model.inference(
                args.images,
                export_dir=str(format_dir),
                export_format="npz",
            )

            print(f"\n✅ NPZ saved to: {format_dir}")
            print(f"   Files: {list(format_dir.glob('*.npz'))}")

        elif fmt == "glb":
            # GLB: 3D mesh (GLTF binary format)
            print("GLB format is a 3D mesh file:")
            print("  - Viewable in Blender, web viewers")
            print("  - Contains geometry + camera positions")
            print("  - Suitable for 3D visualization")

            prediction = model.inference(
                args.images,
                export_dir=str(format_dir),
                export_format="glb",
            )

            print(f"\n✅ GLB saved to: {format_dir}")
            print(f"   Files: {list(format_dir.glob('*.glb'))}")
            print("\n   To view:")
            print("   • Open in Blender: File > Import > glTF 2.0")
            print("   • Online: https://gltf-viewer.donmccurdy.com/")

        elif fmt == "ply":
            # PLY: Point cloud format
            print("PLY format is a point cloud:")
            print("  - Viewable in MeshLab, CloudCompare")
            print("  - Contains 3D points with colors")
            print("  - Suitable for point cloud processing")

            # PLY export requires specific flag
            prediction = model.inference(
                args.images,
                export_dir=str(format_dir),
                export_format="ply",
            )

            print(f"\n✅ PLY saved to: {format_dir}")

        elif fmt == "colmap":
            # COLMAP: Reconstruction format
            print("COLMAP format for 3D reconstruction:")
            print("  - cameras.txt: Camera parameters")
            print("  - images.txt: Image poses")
            print("  - points3D.txt: 3D point cloud")

            # COLMAP export requires image paths (not PIL images)
            prediction = model.inference(
                args.images,
                export_dir=str(format_dir),
                export_format="colmap",
            )

            print(f"\n✅ COLMAP saved to: {format_dir}")

    print(f"\n{'='*60}")
    print("ALL EXPORTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print("\nFormat comparison:")
    print("  NPZ:    Best for Python processing (arrays)")
    print("  GLB:    Best for 3D visualization (meshes)")
    print("  PLY:    Best for point cloud tools")
    print("  COLMAP: Best for 3D reconstruction pipelines")


if __name__ == "__main__":
    main()
