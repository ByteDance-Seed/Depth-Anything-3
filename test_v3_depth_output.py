#!/usr/bin/env python3
"""
Test V3 depth output using original implementation
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def test_v3_depth():
    device = torch.device("cuda")
    
    # Load model
    print("Loading DA3-LARGE model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to(device=device)
    
    # Load test images (first 5 frames from V2 for comparison)
    image_dir = Path("workspace/hei_chole_dav2_large/input_images")
    image_paths = sorted(list(image_dir.glob("*.png")))[:5]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Run inference
    prediction = model.inference(
        image=[str(p) for p in image_paths],
        process_res=504,
    )
    
    output_dir = Path("workspace/v3_depth_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nPrediction shapes:")
    print(f"  depth: {prediction.depth.shape}")
    print(f"  processed_images: {prediction.processed_images.shape}")
    
    for i in range(len(image_paths)):
        # Save raw depth (.npy)
        depth = prediction.depth[i]
        np.save(output_dir / f"depth_{i:06d}.npy", depth)
        
        # Save V3's official visualization
        depth_vis = visualize_depth(depth)
        cv2.imwrite(str(output_dir / f"depth_vis_official_{i:06d}.jpg"), 
                    cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        
        # Save input image
        input_img = prediction.processed_images[i]
        cv2.imwrite(str(output_dir / f"input_{i:06d}.png"), 
                    cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
        
        print(f"Frame {i}: depth min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")
    
    print(f"\n✅ Results saved to {output_dir}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    
    for i in range(5):
        # Input
        img = cv2.imread(str(image_paths[i]))
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Frame {i}: Input')
        axes[i, 0].axis('off')
        
        # Raw depth (grayscale)
        depth = np.load(output_dir / f"depth_{i:06d}.npy")
        axes[i, 1].imshow(depth, cmap='gray')
        axes[i, 1].set_title(f'Depth (raw)\nmin={depth.min():.2f}, max={depth.max():.2f}')
        axes[i, 1].axis('off')
        
        # Official visualization
        vis = cv2.imread(str(output_dir / f"depth_vis_official_{i:06d}.jpg"))
        axes[i, 2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title('V3 Official Visualization\n(Spectral colormap)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_grid.png", dpi=100, bbox_inches='tight')
    print(f"✅ Comparison grid saved to {output_dir}/comparison_grid.png")


if __name__ == "__main__":
    test_v3_depth()
