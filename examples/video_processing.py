#!/usr/bin/env python3
"""
Video Depth Estimation Example

This example demonstrates depth estimation from video files.

Usage:
    python examples/video_processing.py --video input.mp4
    python examples/video_processing.py --video input.mp4 --fps 15
    python examples/video_processing.py --video input.mp4 --export-format glb
    python examples/video_processing.py --video input.mp4 --max-frames 100
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def extract_frames(video_path, fps=None, max_frames=None):
    """Extract frames from video at specified FPS."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info:")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / video_fps:.2f}s")

    # Calculate frame sampling interval
    if fps is None:
        frame_interval = 1
    else:
        frame_interval = int(video_fps / fps)

    frames = []
    frame_idx = 0
    extracted = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted += 1

                if max_frames and extracted >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Extracted {len(frames)} frames (every {frame_interval} frames)")

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Video depth estimation example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/examples/robot_unitree.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="FPS for frame extraction (lower = fewer frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/video_processing",
        help="Directory to save results",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="frames",
        choices=["frames", "video", "glb"],
        help="Output format (frames: individual PNGs, video: depth video, glb: 3D mesh)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="da3-small",
        choices=["da3-small", "da3-base", "da3-large"],
        help="Model size",
    )
    args = parser.parse_args()

    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    print(f"\nExtracting frames from: {args.video}")
    frames = extract_frames(args.video, fps=args.fps, max_frames=args.max_frames)

    if not frames:
        print("❌ No frames extracted")
        return

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nUsing device: {device}")

    # Load model
    print(f"Loading model: {args.model}...")
    model_name_map = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
    }

    model = DepthAnything3.from_pretrained(
        model_name_map[args.model],
        batch_size=args.batch_size,
    )
    model = model.to(device)
    print("Model loaded!")

    # Process frames
    print(f"\nProcessing {len(frames)} frames...")
    start_time = time.time()

    all_depths = []
    all_frames = []

    # Save frames temporarily
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(frames):
        frame_path = temp_dir / f"frame_{i:06d}.png"
        Image.fromarray(frame).save(frame_path)

    # Process all frames at once (model handles batching)
    frame_paths = sorted(temp_dir.glob("frame_*.png"))
    prediction = model.inference([str(p) for p in frame_paths])

    elapsed = time.time() - start_time

    # Save results based on format
    if args.export_format == "frames":
        frames_dir = output_dir / "depth_frames"
        frames_dir.mkdir(exist_ok=True)

        for i, depth in enumerate(tqdm(prediction.depth, desc="Saving frames")):
            # Normalize and save
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            frame_path = frames_dir / f"depth_{i:06d}.png"
            Image.fromarray(depth_vis).save(frame_path)

        print(f"✅ Saved {len(prediction.depth)} depth frames to: {frames_dir}")

    elif args.export_format == "video":
        # Create video from depth frames
        output_video = output_dir / "depth_video.mp4"
        height, width = prediction.depth[0].shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video), fourcc, args.fps, (width, height), False)

        for depth in tqdm(prediction.depth, desc="Creating video"):
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            out.write(depth_vis)

        out.release()
        print(f"✅ Saved depth video to: {output_video}")

    elif args.export_format == "glb":
        # Export as 3D mesh (requires full prediction with camera params)
        print("Exporting as GLB...")
        glb_dir = output_dir / "glb"
        prediction_glb = model.inference(
            [str(p) for p in frame_paths],
            export_dir=str(glb_dir),
            export_format="glb",
        )
        print(f"✅ Saved GLB to: {glb_dir}")

    # Clean up temp frames
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()

    # Print statistics
    print(f"\n✅ Processing complete!")
    print(f"   Total frames: {len(frames)}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {len(frames) / elapsed:.2f} frames/sec")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
