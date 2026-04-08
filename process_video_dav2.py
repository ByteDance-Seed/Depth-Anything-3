#!/usr/bin/env python3
"""
Process video using Depth Anything V2 (from transformers) for comparison
"""

import os
import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
from tqdm import tqdm


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


def process_depth_v2(image_paths, output_dir, model_name, device):
    """Process images with Depth Anything V2"""
    print(f"\nLoading Depth Anything V2 model: {model_name}")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Create output directories
    depth_dir = Path(output_dir) / "depth_vis"
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    depth_raw_dir = Path(output_dir) / "depth_raw"
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    all_depths = []
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Prepare image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()
        all_depths.append(depth)
        
        # Save raw depth
        np.save(depth_raw_dir / f"depth_{idx:06d}.npy", depth)
        
        # Normalize and save visualization
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(depth_dir / f"depth_{idx:04d}.jpg"), depth_colored)
    
    print(f"\n✅ Processing complete!")
    print(f"Saved {len(all_depths)} depth maps")
    
    return all_depths


def main():
    parser = argparse.ArgumentParser(
        description="Process video with Depth Anything V2 for comparison"
    )
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="workspace/video_dav2",
                        help="Output directory (default: workspace/video_dav2)")
    parser.add_argument("--model-name", type=str, 
                        default="depth-anything/Depth-Anything-V2-Large-hf",
                        help="Model name from Hugging Face (default: Depth-Anything-V2-Large-hf)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Sampling FPS for frame extraction (default: 5.0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to extract (default: None)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    
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
    
    # Process with Depth Anything V2
    device = torch.device(args.device)
    depths = process_depth_v2(image_paths, output_dir, args.model_name, device)
    
    # Save configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": "Depth Anything V2",
        "model_name": args.model_name,
        "video_info": video_info,
        "processing": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "device": str(device),
        },
        "results": {
            "num_frames": len(depths),
            "depth_shapes": [d.shape for d in depths[:3]],  # First 3 for reference
        }
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📋 Configuration saved to: {output_dir / 'config.json'}")
    print(f"\nOutput files:")
    print(f"  - input_images/ ({len(image_paths)} frames)")
    print(f"  - depth_vis/ ({len(depths)} depth visualizations)")
    print(f"  - depth_raw/ ({len(depths)} raw .npy files)")


if __name__ == "__main__":
    main()
