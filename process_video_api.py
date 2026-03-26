#!/usr/bin/env python3
"""
Process video using Depth Anything 3 Python API
Extract frames and generate point cloud
"""

import os
import cv2
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from depth_anything_3.api import DepthAnything3


def extract_frames(video_path, output_dir, fps=5.0, max_frames=None):
    """Extract frames from video"""
    print(f"Extracting frames from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    actual_fps = video_fps / frame_interval
    
    print(f"Video FPS: {video_fps:.2f}, Duration: {duration:.2f}s")
    print(f"Resolution: {width}x{height}")
    print(f"Extracting frames at {actual_fps:.2f} FPS (every {frame_interval} frame(s))")
    
    if max_frames is not None:
        print(f"⚠️  Limiting output to maximum {max_frames} frames")
    
    # Create output directory
    frames_dir = Path(output_dir) / "input_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    saved_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if max_frames is None or saved_count < max_frames:
                frame_path = frames_dir / f"{saved_count:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(str(frame_path))
                saved_count += 1
            else:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {frames_dir}")
    
    video_info = {
        "fps": float(video_fps),
        "total_frames": total_frames,
        "duration_sec": float(duration),
        "resolution": [width, height],
        "extracted_frames": saved_count,
        "actual_extraction_fps": float(actual_fps),
    }
    
    return saved_paths, video_info


def save_config(output_dir, args, video_info=None, result_info=None):
    """Save configuration and results to JSON file"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "video_path": args.video_path,
            "fps_extraction": args.fps,
            "max_frames": args.max_frames,
        },
        "model": {
            "model_dir": args.model_dir,
            "device": args.device,
        },
        "processing": {
            "process_res": args.process_res,
            "export_format": args.export_format,
            "infer_gs": args.infer_gs,
            "num_max_points": args.num_max_points,
        },
    }
    
    if video_info:
        config["video_info"] = video_info
    
    if result_info:
        config["results"] = result_info
    
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Process video with Depth Anything 3")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="workspace/video_output",
                        help="Output directory (default: workspace/video_output)")
    parser.add_argument("--model-dir", type=str, default="depth-anything/DA3-SMALL",
                        help="Model directory (default: depth-anything/DA3-SMALL)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Sampling FPS for frame extraction (default: 5.0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to extract (default: None)")
    parser.add_argument("--export-format", type=str, default="glb-depth_vis",
                        help="Export format (default: glb-depth_vis)")
    parser.add_argument("--process-res", type=int, default=504,
                        help="Processing resolution (default: 504)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--infer-gs", action="store_true",
                        help="Enable Gaussian Splatting (requires DA3-GIANT or DA3NESTED-GIANT-LARGE)")
    parser.add_argument("--num-max-points", type=int, default=1000000,
                        help="Maximum number of points in point cloud (default: 1000000)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    image_paths, video_info = extract_frames(
        args.video_path, 
        output_dir, 
        fps=args.fps,
        max_frames=args.max_frames
    )
    
    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)
    
    # Run inference
    print(f"\nRunning inference on {len(image_paths)} images...")
    print(f"Export format: {args.export_format}")
    print(f"Processing resolution: {args.process_res}")
    
    prediction = model.inference(
        image=image_paths,
        export_dir=str(output_dir),
        export_format=args.export_format,
        process_res=args.process_res,
        process_res_method="lower_bound_resize",
        infer_gs=args.infer_gs,
        num_max_points=args.num_max_points,
        show_cameras=True,
    )
    
    print(f"\n✅ Processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nOutput files:")
    
    # List output files
    output_files = []
    if (output_dir / "scene.glb").exists():
        glb_size = (output_dir / "scene.glb").stat().st_size / (1024**2)
        print(f"  - scene.glb ({glb_size:.2f} MB)")
        output_files.append({"file": "scene.glb", "size_mb": round(glb_size, 2)})
    if (output_dir / "scene.jpg").exists():
        print(f"  - scene.jpg")
        output_files.append({"file": "scene.jpg"})
    if (output_dir / "depth_vis").exists():
        num_depth = len(list((output_dir / "depth_vis").glob("*.jpg")))
        print(f"  - depth_vis/ ({num_depth} images)")
        output_files.append({"file": "depth_vis/", "num_images": num_depth})
    if (output_dir / "gs_video").exists():
        print(f"  - gs_video/")
        output_files.append({"file": "gs_video/"})
    
    print(f"\nDepth shape: {prediction.depth.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")
    if prediction.extrinsics is not None:
        print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    
    # Save configuration with result info
    result_info = {
        "depth_shape": list(prediction.depth.shape),
        "confidence_shape": list(prediction.conf.shape),
        "extrinsics_shape": list(prediction.extrinsics.shape) if prediction.extrinsics is not None else None,
        "intrinsics_shape": list(prediction.intrinsics.shape) if prediction.intrinsics is not None else None,
        "output_files": output_files,
    }
    
    save_config(output_dir, args, video_info, result_info)
    print(f"\n📋 Configuration and results recorded in config.json")


if __name__ == "__main__":
    main()
