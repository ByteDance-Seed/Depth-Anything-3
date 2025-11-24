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
Model inference module for Depth Anything 3 Gradio app.

This module handles all model-related operations including inference,
data processing, and result preparation.
"""

import glob
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.gs import export_to_gs_video


class ModelInference:
    """
    Handles model inference and data processing for Depth Anything 3.
    """

    def __init__(self):
        """Initialize the model inference handler."""
        self.model = None
        self.model_opts: Dict[str, Any] = {
            "batch_size": None,
            "mixed_precision": None,
        }

    def initialize_model(
        self,
        device: str = "cuda",
        *,
        batch_size: Optional[int] = None,
        mixed_precision: Optional[str] = "auto",
    ) -> None:
        """
        Initialize the DepthAnything3 model.

        Args:
            device: Device to load the model on
        """
        desired_opts = {
            "batch_size": batch_size,
            "mixed_precision": None if mixed_precision == "auto" else mixed_precision,
        }

        should_reload = (self.model is None) or (desired_opts != self.model_opts)

        if should_reload:
            # Get model directory from environment variable or use default
            model_dir = os.environ.get(
                "DA3_MODEL_DIR", "/dev/shm/da3_models/DA3HF-VITG-METRIC_VITL"
            )
            self.model = DepthAnything3.from_pretrained(
                model_dir,
                batch_size=desired_opts["batch_size"],
                mixed_precision=desired_opts["mixed_precision"],
            )
            self.model_opts = desired_opts
            self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)

        self.model.eval()

    def run_inference(
        self,
        target_dir: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        batch_size: Optional[int] = None,
        mixed_precision: str = "auto",
        show_camera: bool = True,
        selected_first_frame: Optional[str] = None,
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
    ) -> Tuple[Any, Dict[int, Dict[str, Any]]]:
        """
        Run DepthAnything3 model inference on images.

        Args:
            target_dir: Directory containing images
            apply_mask: Whether to apply mask for ambiguous depth classes
            mask_edges: Whether to mask edges
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images
            show_camera: Whether to show camera in 3D view
            selected_first_frame: Selected first frame filename
            save_percentage: Percentage of points to save (0-100)
            infer_gs: Whether to infer 3D Gaussian Splatting

        Returns:
            Tuple of (prediction, processed_data)
        """
        print(f"Processing images from {target_dir}")

        # Device check: prioritize CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Initialize model if needed
        self.initialize_model(
            device,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
        )

        # Get image paths
        print("Loading images...")
        image_folder_path = os.path.join(target_dir, "images")
        all_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*")))

        # Filter for image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        all_image_paths = [
            path
            for path in all_image_paths
            if any(path.lower().endswith(ext) for ext in image_extensions)
        ]

        print(f"Found {len(all_image_paths)} images")
        print(f"All image paths: {all_image_paths}")

        # Apply first frame selection logic
        if selected_first_frame:
            # Find the image with matching filename
            selected_path = None
            for path in all_image_paths:
                if os.path.basename(path) == selected_first_frame:
                    selected_path = path
                    break

            if selected_path:
                # Move selected frame to the front
                image_paths = [selected_path] + [
                    path for path in all_image_paths if path != selected_path
                ]
                print(f"User selected first frame: {selected_first_frame} -> {selected_path}")
                print(f"Reordered image paths: {image_paths}")
            else:
                # Use default order if no match found
                image_paths = all_image_paths
                print(
                    f"Selected frame '{selected_first_frame}' not found in image paths. "
                    "Using default order."
                )
                first_frame_display = image_paths[0] if image_paths else "No images"
                print(f"Using default order (first frame): {first_frame_display}")
        else:
            # Use default order (sorted)
            image_paths = all_image_paths
            first_frame_display = image_paths[0] if image_paths else "No images"
            print(f"Using default order (first frame): {first_frame_display}")

        if len(image_paths) == 0:
            raise ValueError("No images found. Check your upload.")

        # Map UI options to actual method names
        method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}
        actual_method = method_mapping.get(process_res_method, "upper_bound_crop")

        # Run model inference
        print(f"Running inference with method: {actual_method}")
        effective_batch = batch_size if batch_size is not None else min(len(image_paths), 4)
        try:
            with torch.no_grad():
                prediction = self.model.inference(
                    image_paths,
                    export_dir=None,
                    process_res_method=actual_method,
                    infer_gs=infer_gs,
                    export_feat_layers=[],
                )
        except RuntimeError as e:
            # Friendly message for OOM / invalid buffer
            msg = str(e).lower()
            if "invalid buffer size" in msg or "out of memory" in msg:
                raise ValueError(
                    "Inference failed: insufficient memory. "
                    "Essayez de réduire le nombre d'images, de choisir 'low_res', "
                    "ou de définir un batch_size plus petit."
                ) from e
            raise
        # num_max_points: int = 1_000_000,
        export_to_glb(
            prediction,
            filter_black_bg=filter_black_bg,
            filter_white_bg=filter_white_bg,
            export_dir=target_dir,
            show_cameras=show_camera,
            conf_thresh_percentile=save_percentage,
            num_max_points=int(num_max_points),
        )

        # export to gs video if needed
        if infer_gs:
            mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
            print(f"GS mode: {gs_trj_mode}; Backend mode: {mode_mapping[gs_trj_mode]}")
            export_to_gs_video(
                prediction,
                export_dir=target_dir,
                chunk_size=4,
                trj_mode=mode_mapping.get(gs_trj_mode, "extend"),
                enable_tqdm=True,
                vis_depth="hcat",
                video_quality=gs_video_quality,
            )

        # Save predictions.npz for caching metric depth data
        self._save_predictions_cache(target_dir, prediction)

        # Process results
        processed_data = self._process_results(target_dir, prediction, image_paths)

        # Clean up using centralized memory utilities for consistency with backend
        cleanup_cuda_memory()

        return prediction, processed_data

    def _save_predictions_cache(self, target_dir: str, prediction: Any) -> None:
        """
        Save predictions data to predictions.npz for caching.

        Args:
            target_dir: Directory to save the cache
            prediction: Model prediction object
        """
        try:
            output_file = os.path.join(target_dir, "predictions.npz")

            # Build save dict with prediction data
            save_dict = {}

            # Save processed images if available
            if prediction.processed_images is not None:
                save_dict["images"] = prediction.processed_images

            # Save depth data
            if prediction.depth is not None:
                save_dict["depths"] = np.round(prediction.depth, 6)

            # Save confidence if available
            if prediction.conf is not None:
                save_dict["conf"] = np.round(prediction.conf, 2)

            # Save camera parameters
            if prediction.extrinsics is not None:
                save_dict["extrinsics"] = prediction.extrinsics
            if prediction.intrinsics is not None:
                save_dict["intrinsics"] = prediction.intrinsics

            # Save to file
            np.savez_compressed(output_file, **save_dict)
            print(f"Saved predictions cache to: {output_file}")

        except Exception as e:
            print(f"Warning: Failed to save predictions cache: {e}")

    def _process_results(
        self, target_dir: str, prediction: Any, image_paths: list
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process model results into structured data.

        Args:
            target_dir: Directory containing results
            prediction: Model prediction object
            image_paths: List of input image paths

        Returns:
            Dictionary containing processed data for each view
        """
        processed_data = {}

        # Read generated depth visualization files
        depth_vis_dir = os.path.join(target_dir, "depth_vis")

        if os.path.exists(depth_vis_dir):
            depth_files = sorted(glob.glob(os.path.join(depth_vis_dir, "*.jpg")))
            for i, depth_file in enumerate(depth_files):
                # Use processed images directly from API
                processed_image = None
                if prediction.processed_images is not None and i < len(
                    prediction.processed_images
                ):
                    processed_image = prediction.processed_images[i]

                processed_data[i] = {
                    "depth_image": depth_file,
                    "image": processed_image,
                    "original_image_path": image_paths[i] if i < len(image_paths) else None,
                    "depth": prediction.depth[i] if i < len(prediction.depth) else None,
                    "intrinsics": (
                        prediction.intrinsics[i]
                        if prediction.intrinsics is not None and i < len(prediction.intrinsics)
                        else None
                    ),
                    "mask": None,  # No mask information available
                }

        return processed_data

    # cleanup() removed: call cleanup_cuda_memory() directly where needed.
