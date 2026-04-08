#!/usr/bin/env python3
"""
Process video using Depth Anything 3 in batches to handle memory constraints
Split large video processing into smaller batches and optionally merge results
"""

import os
import cv2
import json
import torch
import argparse
import shutil
import gc
from pathlib import Path
from datetime import datetime
from depth_anything_3.api import DepthAnything3


def extract_frames_batch(video_path, output_dir, fps=5.0, batch_size=50, batch_idx=0):
    """Extract a specific batch of frames from video"""
    print(f"\nExtracting batch {batch_idx} (frames {batch_idx * batch_size} to {(batch_idx + 1) * batch_size - 1})...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    # Calculate batch range
    start_frame_idx = batch_idx * batch_size
    end_frame_idx = min((batch_idx + 1) * batch_size, total_frames // frame_interval)
    
    if start_frame_idx >= total_frames // frame_interval:
        cap.release()
        return [], None
    
    # Create output directory
    frames_dir = Path(output_dir) / f"batch_{batch_idx:03d}" / "input_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    saved_paths = []
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if start_frame_idx <= extracted_count < end_frame_idx:
                frame_path = frames_dir / f"{saved_count:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(str(frame_path))
                saved_count += 1
            extracted_count += 1
            
            if extracted_count >= end_frame_idx:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {frames_dir}")
    
    video_info = {
        "fps": float(video_fps),
        "resolution": [width, height],
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "extracted_frames": saved_count,
        "frame_range": [start_frame_idx, end_frame_idx],
    }
    
    return saved_paths, video_info


def process_batch(model, image_paths, output_dir, args, batch_idx):
    """Process a single batch of images"""
    print(f"\nProcessing batch {batch_idx} ({len(image_paths)} images)...")
    
    batch_dir = Path(output_dir) / f"batch_{batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    prediction = model.inference(
        image=image_paths,
        export_dir=str(batch_dir),
        export_format=args.export_format,
        process_res=args.process_res,
        process_res_method="lower_bound_resize",
        infer_gs=args.infer_gs,
        num_max_points=args.num_max_points,
        show_cameras=True,
    )
    
    # Save batch info
    batch_info = {
        "batch_idx": batch_idx,
        "num_frames": len(image_paths),
        "depth_shape": list(prediction.depth.shape),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(batch_dir / "batch_info.json", "w") as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"✅ Batch {batch_idx} complete")
    print(f"   Depth shape: {prediction.depth.shape}")
    
    # Clear GPU memory
    del prediction
    torch.cuda.empty_cache()
    gc.collect()
    
    return batch_info


def merge_results(output_dir, total_batches, args):
    """Merge results from all batches"""
    print(f"\n📦 Merging results from {total_batches} batches...")
    
    merged_dir = Path(output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge input images
    merged_images_dir = merged_dir / "input_images"
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge depth visualizations if available
    merged_depth_vis_dir = merged_dir / "depth_vis"
    has_depth_vis = False
    
    global_frame_idx = 0
    batch_infos = []
    
    for batch_idx in range(total_batches):
        batch_dir = Path(output_dir) / f"batch_{batch_idx:03d}"
        
        # Load batch info
        with open(batch_dir / "batch_info.json", "r") as f:
            batch_info = json.load(f)
            batch_infos.append(batch_info)
        
        # Copy input images
        input_dir = batch_dir / "input_images"
        if input_dir.exists():
            for img_path in sorted(input_dir.glob("*.png")):
                dest_path = merged_images_dir / f"{global_frame_idx:06d}.png"
                shutil.copy2(img_path, dest_path)
                global_frame_idx += 1
        
        # Copy depth visualizations
        depth_vis_dir = batch_dir / "depth_vis"
        if depth_vis_dir.exists():
            if not has_depth_vis:
                merged_depth_vis_dir.mkdir(parents=True, exist_ok=True)
                has_depth_vis = True
            
            for depth_idx, depth_path in enumerate(sorted(depth_vis_dir.glob("*.jpg"))):
                dest_idx = sum(b["num_frames"] for b in batch_infos[:-1]) + depth_idx
                dest_path = merged_depth_vis_dir / f"depth_{dest_idx:04d}.jpg"
                shutil.copy2(depth_path, dest_path)
    
    # Save merged info
    merged_info = {
        "total_batches": total_batches,
        "total_frames": global_frame_idx,
        "batches": batch_infos,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(merged_dir / "merged_info.json", "w") as f:
        json.dump(merged_info, f, indent=2)
    
    print(f"✅ Merged {global_frame_idx} frames")
    print(f"   Merged directory: {merged_dir}")
    
    return merged_info


def main():
    parser = argparse.ArgumentParser(
        description="Process video with Depth Anything 3 in batches to handle memory constraints"
    )
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="workspace/video_batched",
                        help="Output directory (default: workspace/video_batched)")
    parser.add_argument("--model-dir", type=str, default="depth-anything/DA3-LARGE",
                        help="Model directory (default: depth-anything/DA3-LARGE)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Sampling FPS for frame extraction (default: 5.0)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of frames to process in each batch (default: 50)")
    parser.add_argument("--export-format", type=str, default="glb-depth_vis",
                        help="Export format (default: glb-depth_vis)")
    parser.add_argument("--process-res", type=int, default=504,
                        help="Processing resolution (default: 504)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--infer-gs", action="store_true",
                        help="Enable Gaussian Splatting")
    parser.add_argument("--num-max-points", type=int, default=1000000,
                        help="Maximum number of points in point cloud (default: 1000000)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging results after processing")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum number of batches to process (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total number of batches
    cap = cv2.VideoCapture(args.video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    frame_interval = max(1, int(video_fps / args.fps))
    total_extractable_frames = total_frames // frame_interval
    total_batches = (total_extractable_frames + args.batch_size - 1) // args.batch_size
    
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    
    print(f"📹 Video: {args.video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   Extractable frames (at {args.fps} FPS): {total_extractable_frames}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Total batches: {total_batches}")
    
    # Load model once
    print(f"\n🔧 Loading model from {args.model_dir}...")
    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)
    print("✅ Model loaded")
    
    # Process each batch
    batch_infos = []
    for batch_idx in range(total_batches):
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{total_batches}")
        print(f"{'='*60}")
        
        # Extract frames for this batch
        image_paths, video_info = extract_frames_batch(
            args.video_path,
            output_dir,
            fps=args.fps,
            batch_size=args.batch_size,
            batch_idx=batch_idx,
        )
        
        if not image_paths:
            print(f"No more frames to process")
            break
        
        # Process batch
        batch_info = process_batch(model, image_paths, output_dir, args, batch_idx)
        batch_infos.append(batch_info)
    
    print(f"\n{'='*60}")
    print(f"✅ All batches processed!")
    print(f"{'='*60}")
    
    # Merge results
    if not args.no_merge and len(batch_infos) > 0:
        merge_results(output_dir, len(batch_infos), args)
    
    # Save overall config
    config = {
        "timestamp": datetime.now().isoformat(),
        "video_path": args.video_path,
        "model_dir": args.model_dir,
        "fps": args.fps,
        "batch_size": args.batch_size,
        "total_batches": len(batch_infos),
        "export_format": args.export_format,
        "process_res": args.process_res,
        "batches": batch_infos,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📋 Configuration saved to: {output_dir / 'config.json'}")
    print(f"🎉 Processing complete!")


if __name__ == "__main__":
    main()
