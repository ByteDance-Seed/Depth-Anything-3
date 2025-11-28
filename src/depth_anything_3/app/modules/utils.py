# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for Depth Anything 3 Gradio app.

This module contains helper functions for data processing, visualization,
and file operations.
"""


import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def create_depth_visualization(depth: np.ndarray) -> Optional[np.ndarray]:
    """
    Create a colored depth visualization.

    Args:
        depth: Depth array

    Returns:
        Colored depth visualization or None
    """
    if depth is None:
        return None

    # Normalize depth to 0-1 range
    depth_min = depth[depth > 0].min() if (depth > 0).any() else 0
    depth_max = depth.max()

    if depth_max <= depth_min:
        return None

    # Normalize depth
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap (using matplotlib's viridis colormap)
    import matplotlib.cm as cm

    # Convert to colored image
    depth_colored = cm.viridis(depth_norm)[:, :, :3]  # Remove alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored


def save_to_gallery_func(
    target_dir: str, processed_data: Dict[int, Dict[str, Any]], gallery_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Save the current reconstruction results to the gallery directory.

    Args:
        target_dir: Source directory containing reconstruction results
        processed_data: Processed data dictionary
        gallery_name: Name for the gallery folder

    Returns:
        Tuple of (success, message)
    """
    try:
        # Get gallery directory from environment variable or use default
        gallery_dir = os.environ.get(
            "DA3_GALLERY_DIR",
            "workspace/gallery",
        )
        if not os.path.exists(gallery_dir):
            os.makedirs(gallery_dir)

        # Use provided name or create a unique name
        if gallery_name is None or gallery_name.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gallery_name = f"reconstruction_{timestamp}"

        gallery_path = os.path.join(gallery_dir, gallery_name)

        # Check if directory already exists
        if os.path.exists(gallery_path):
            return False, f"Save failed: folder '{gallery_name}' already exists"

        # Create the gallery directory
        os.makedirs(gallery_path, exist_ok=True)

        # Copy GLB file
        glb_source = os.path.join(target_dir, "scene.glb")
        glb_dest = os.path.join(gallery_path, "scene.glb")
        if os.path.exists(glb_source):
            shutil.copy2(glb_source, glb_dest)

        # Copy depth visualization images
        depth_vis_dir = os.path.join(target_dir, "depth_vis")
        if os.path.exists(depth_vis_dir):
            gallery_depth_vis = os.path.join(gallery_path, "depth_vis")
            shutil.copytree(depth_vis_dir, gallery_depth_vis)

        # Copy original images
        images_source = os.path.join(target_dir, "images")
        if os.path.exists(images_source):
            gallery_images = os.path.join(gallery_path, "images")
            shutil.copytree(images_source, gallery_images)

        scene_preview_source = os.path.join(target_dir, "scene.jpg")
        scene_preview_dest = os.path.join(gallery_path, "scene.jpg")
        shutil.copy2(scene_preview_source, scene_preview_dest)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_images": len(processed_data) if processed_data else 0,
            "gallery_name": gallery_name,
        }

        with open(os.path.join(gallery_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved reconstruction to gallery: {gallery_path}")
        return True, f"Save successful: saved to {gallery_path}"

    except Exception as e:
        print(f"Error saving to gallery: {e}")
        return False, f"Save failed: {str(e)}"


def get_scene_info(examples_dir: str) -> List[Dict[str, Any]]:
    """
    Get information about scenes in the examples directory.

    Args:
        examples_dir: Path to examples directory

    Returns:
        List of scene information dictionaries
    """
    import glob

    scenes = []
    if not os.path.exists(examples_dir):
        return scenes

    for scene_folder in sorted(os.listdir(examples_dir)):
        scene_path = os.path.join(examples_dir, scene_folder)
        if os.path.isdir(scene_path):
            # Find all image files in the scene folder
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(scene_path, ext)))
                image_files.extend(glob.glob(os.path.join(scene_path, ext.upper())))

            if image_files:
                # Sort images and get the first one for thumbnail
                image_files = sorted(image_files)
                first_image = image_files[0]
                num_images = len(image_files)

                scenes.append(
                    {
                        "name": scene_folder,
                        "path": scene_path,
                        "thumbnail": first_image,
                        "num_images": num_images,
                        "image_files": image_files,
                    }
                )

    return scenes


# NOTE: cleanup was moved to a single canonical helper in
# `depth_anything_3.utils.memory.cleanup_cuda_memory`.
# Callers should import and call that directly instead of using this module.


def get_logo_base64() -> Optional[str]:
    """
    Convert WAI logo to base64 for embedding in HTML.

    Returns:
        Base64 encoded logo string or None
    """
    import base64

    logo_path = "examples/WAI-Logo/wai_logo.png"
    try:
        with open(logo_path, "rb") as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{base64_str}"
    except FileNotFoundError:
        return None
