#!/usr/bin/env python3
"""
Generate a 3D point-cloud map (GLB) from video frames using Depth Anything V3.

V3 jointly predicts depth, confidence, camera poses, and intrinsics.
The GLB exporter back-projects depth into a colored point cloud and
adds camera wireframes — viewable in any 3D viewer (e.g. VS Code, Blender,
https://gltf-viewer.donmccurdy.com/).

Usage:
    # From pre-extracted frames (e.g. the 50 hei_chole frames)
    python generate_3d_map.py \
        --image-dir workspace/hei_chole_dav2_large/input_images \
        --output-dir workspace/3d_map \
        --max-frames 20 --step 2

    # From a video file directly
    python generate_3d_map.py \
        --video assets/examples/robot_unitree.mp4 \
        --output-dir workspace/3d_map_robot \
        --fps 2 --max-frames 20
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from depth_anything_3.api import DepthAnything3


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------

def load_frames_from_dir(
    image_dir: str, max_frames: int | None = None, step: int = 1,
) -> list[str]:
    """Load image paths from a directory, optionally subsampling."""
    d = Path(image_dir)
    paths = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    paths = paths[::step]
    if max_frames is not None:
        paths = paths[:max_frames]
    return [str(p) for p in paths]


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    max_frames: int | None = None,
) -> tuple[list[str], dict]:
    """Extract frames from video at the given FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(vid_fps / fps))

    frames_dir = Path(output_dir) / "input_images"
    frames_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            p = frames_dir / f"{saved:06d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(str(p))
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()

    info = {
        "source_video": video_path,
        "video_fps": vid_fps,
        "video_frames": total,
        "resolution": [w, h],
        "extraction_fps": vid_fps / interval,
        "extracted_frames": saved,
    }
    print(f"Extracted {saved} frames at ~{info['extraction_fps']:.1f} fps → {frames_dir}")
    return paths, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D point cloud (GLB) from images/video using DA3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", type=str,
                     help="Directory with pre-extracted frames (.png/.jpg)")
    src.add_argument("--video", type=str,
                     help="Path to a video file")

    parser.add_argument("--output-dir", type=str, default="workspace/3d_map",
                        help="Output directory (default: workspace/3d_map)")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-LARGE",
                        help="Model name (default: DA3-LARGE)")

    # Frame selection
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Extraction FPS when using --video (default: 2)")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Maximum frames to use (default: 20)")
    parser.add_argument("--step", type=int, default=1,
                        help="Take every Nth frame from --image-dir (default: 1)")

    # GLB export tuning
    parser.add_argument("--process-res", type=int, default=504,
                        help="Model processing resolution (default: 504)")
    parser.add_argument("--num-max-points", type=int, default=2_000_000,
                        help="Max points in the GLB point cloud (default: 2M)")
    parser.add_argument("--conf-percentile", type=float, default=30.0,
                        help="Confidence filter percentile (default: 30; lower keeps more)")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Omit camera wireframes from the GLB")

    args = parser.parse_args()

    # ---- Gather frames ----
    video_info = None
    if args.video:
        image_paths, video_info = extract_frames_from_video(
            args.video, args.output_dir, fps=args.fps, max_frames=args.max_frames,
        )
    else:
        image_paths = load_frames_from_dir(
            args.image_dir, max_frames=args.max_frames, step=args.step,
        )

    n = len(image_paths)
    print(f"\nUsing {n} frames for 3D reconstruction")
    if n < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model).to("cuda")

    # ---- Run inference + GLB export ----
    print(f"Running inference (process_res={args.process_res})...")
    prediction = model.inference(
        image=image_paths,
        export_dir=args.output_dir,
        export_format="mini_npz-glb-depth_vis",
        process_res=args.process_res,
        conf_thresh_percentile=args.conf_percentile,
        num_max_points=args.num_max_points,
        show_cameras=not args.no_cameras,
    )

    # ---- Save metadata ----
    out = Path(args.output_dir)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "num_frames": n,
        "process_res": args.process_res,
        "num_max_points": args.num_max_points,
        "conf_thresh_percentile": args.conf_percentile,
        "image_paths": image_paths,
        "depth_shape": list(prediction.depth.shape),
        "depth_range": [float(prediction.depth.min()), float(prediction.depth.max())],
    }
    if video_info:
        meta["video_info"] = video_info

    with open(out / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Save raw .npy depth ---
    depth_raw_dir = out / "depth_raw"
    depth_raw_dir.mkdir(exist_ok=True)
    for i in range(n):
        np.save(depth_raw_dir / f"depth_{i:06d}.npy", prediction.depth[i])

    glb_path = out / "scene.glb"
    print(f"\nDone!")
    print(f"  GLB point cloud : {glb_path}  ({glb_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Depth vis       : {out / 'depth_vis'}")
    print(f"  Raw depth (.npy): {depth_raw_dir}")
    print(f"  Config          : {out / 'config.json'}")


if __name__ == "__main__":
    main()
