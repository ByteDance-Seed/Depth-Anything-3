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
Event handling module for Depth Anything 3 Gradio app.

This module handles all event callbacks and user interactions.
"""

import os
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import numpy as np
import torch

from depth_anything_3.app.modules.file_handlers import FileHandler
from depth_anything_3.app.modules.model_inference import ModelInference
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.app.modules.visualization import VisualizationHandler


class EventHandlers:
    """
    Handles all event callbacks and user interactions for the Gradio app.
    """

    def __init__(self):
        """Initialize the event handlers."""
        self.model_inference = ModelInference()
        self.file_handler = FileHandler()
        self.visualization_handler = VisualizationHandler()

    def clear_fields(self) -> None:
        """
        Clears the 3D viewer, the stored target_dir, and empties the gallery.
        """
        return None

    def update_log(self) -> str:
        """
        Display a quick log message while waiting.
        """
        return "Loading and Reconstructing..."

    def save_current_visualization(
        self,
        target_dir: str,
        save_percentage: float,
        show_cam: bool,
        filter_black_bg: bool,
        filter_white_bg: bool,
        processed_data: Optional[Dict],
        scene_name: str = "",
    ) -> str:
        """
        Save current visualization results to gallery with specified save percentage.

        Args:
            target_dir: Directory containing results
            save_percentage: Percentage of points to save (0-100)
            show_cam: Whether to show cameras
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            processed_data: Processed data from reconstruction

        Returns:
            Status message
        """
        if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
            return "No reconstruction available. Please run 'Reconstruct' first."

        if processed_data is None:
            return "No processed data available. Please run 'Reconstruct' first."

        try:
            # Add debug information
            print("[DEBUG] save_current_visualization called with:")
            print(f"  target_dir: {target_dir}")
            print(f"  save_percentage: {save_percentage}")
            print(f"  show_cam: {show_cam}")
            print(f"  filter_black_bg: {filter_black_bg}")
            print(f"  filter_white_bg: {filter_white_bg}")
            print(f"  processed_data: {processed_data is not None}")

            # Import the gallery save function
            # Create gallery name with user input or auto-generated
            import datetime

            from .utils import save_to_gallery_func

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if scene_name and scene_name.strip():
                gallery_name = f"{scene_name.strip()}_{timestamp}_pct{save_percentage:.0f}"
            else:
                gallery_name = f"save_{timestamp}_pct{save_percentage:.0f}"

            print(f"[DEBUG] Saving to gallery with name: {gallery_name}")

            # Save entire process folder to gallery
            success, message = save_to_gallery_func(
                target_dir=target_dir, processed_data=processed_data, gallery_name=gallery_name
            )

            if success:
                print(f"[DEBUG] Gallery save completed successfully: {message}")
                return (
                    "Successfully saved to gallery!\n"
                    f"Gallery name: {gallery_name}\n"
                    f"Save percentage: {save_percentage}%\n"
                    f"Show cameras: {show_cam}\n"
                    f"Filter black bg: {filter_black_bg}\n"
                    f"Filter white bg: {filter_white_bg}\n\n"
                    f"{message}"
                )
            else:
                print(f"[DEBUG] Gallery save failed: {message}")
                return f"Failed to save to gallery: {message}"

        except Exception as e:
            return f"Error saving visualization: {str(e)}"

    def gradio_demo(
        self,
        target_dir: str,
        show_cam: bool = True,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        batch_size: Optional[int] = None,
        mixed_precision: str = "auto",
        selected_first_frame: str = "",
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
    ) -> Tuple[
        Optional[str],
        str,
        Optional[Dict],
        Optional[np.ndarray],
        Optional[np.ndarray],
        str,
        gr.Dropdown,
        Optional[str],  # gs video path
        gr.update,  # gs video visibility update
        gr.update,  # gs info visibility update
    ]:
        """
        Perform reconstruction using the already-created target_dir/images.

        Args:
            target_dir: Directory containing images
            show_cam: Whether to show camera
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images
            selected_first_frame: Selected first frame filename
            infer_gs: Whether to infer 3D Gaussian Splatting

        Returns:
            Tuple of reconstruction results
        """
        if not os.path.isdir(target_dir) or target_dir == "None":
            return (
                None,
                "No valid target directory found. Please upload first.",
                None,
                None,
                None,
                "",
                None,
                None,
                gr.update(visible=False),  # gs_video
                gr.update(visible=True),  # gs_info
            )

        start_time = time.time()
        cleanup_cuda_memory()

        # Get image files for logging
        target_dir_images = os.path.join(target_dir, "images")
        all_files = (
            sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
        )

        print("Running DepthAnything3 model...")
        print(f"Selected first frame: {selected_first_frame}")

        # Validate selected_first_frame against current image list
        if selected_first_frame and target_dir_images:
            current_files = (
                sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
            )
            if selected_first_frame not in current_files:
                print(
                    f"Selected first frame '{selected_first_frame}' not found in "
                    "current images. Using default order."
                )
                selected_first_frame = ""  # Reset to use default order

        with torch.no_grad():
            prediction, processed_data = self.model_inference.run_inference(
                target_dir,
                process_res_method=process_res_method,
                batch_size=batch_size,
                mixed_precision=mixed_precision,
                show_camera=show_cam,
                selected_first_frame=selected_first_frame,
                save_percentage=save_percentage,
                num_max_points=int(num_max_points * 1000),  # Convert K to actual count
                infer_gs=infer_gs,
                gs_trj_mode=gs_trj_mode,
                gs_video_quality=gs_video_quality,
            )

        # The GLB file is already generated by the API
        glbfile = os.path.join(target_dir, "scene.glb")

        # Handle 3DGS video based on infer_gs flag
        gsvideo_path = None
        gs_video_visible = False
        gs_info_visible = True

        if infer_gs:
            try:
                gsvideo_path = sorted(glob(os.path.join(target_dir, "gs_video", "*.mp4")))[-1]
                gs_video_visible = True
                gs_info_visible = False
            except IndexError:
                gsvideo_path = None
                print("3DGS video not found, but infer_gs was enabled")

        # Cleanup
        cleanup_cuda_memory()

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

        # Populate visualization tabs with processed data
        depth_vis, measure_img, measure_depth_vis, measure_pts = (
            self.visualization_handler.populate_visualization_tabs(processed_data)
        )

        # Update view selectors based on available views
        depth_selector, measure_selector = self.visualization_handler.update_view_selectors(
            processed_data
        )

        return (
            glbfile,
            log_msg,
            processed_data,
            measure_img,  # measure_image
            measure_depth_vis,  # measure_depth_image
            "",  # measure_text (empty initially)
            measure_selector,  # measure_view_selector
            gsvideo_path,
            gr.update(visible=gs_video_visible),  # gs_video visibility
            gr.update(visible=gs_info_visible),  # gs_info visibility
        )

    def update_visualization(
        self,
        target_dir: str,
        show_cam: bool,
        is_example: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
    ) -> Tuple[gr.update, str]:
        """
        Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
        and return it for the 3D viewer.

        Args:
            target_dir: Directory containing results
            show_cam: Whether to show camera
            is_example: Whether this is an example scene
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images

        Returns:
            Tuple of (glb_file, log_message)
        """
        if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
            return (
                gr.update(),
                "No reconstruction available. Please click the Reconstruct button first.",
            )

        # Check if GLB exists (could be cached example or reconstructed scene)
        glbfile = os.path.join(target_dir, "scene.glb")
        if os.path.exists(glbfile):
            return (
                glbfile,
                (
                    "Visualization loaded from cache."
                    if is_example == "True"
                    else "Visualization updated."
                ),
            )

        # If no GLB but it's an example that hasn't been reconstructed yet
        if is_example == "True":
            return (
                gr.update(),
                "No reconstruction available. Please click the Reconstruct button first.",
            )

        # For non-examples, check predictions.npz
        predictions_path = os.path.join(target_dir, "predictions.npz")
        if not os.path.exists(predictions_path):
            error_message = (
                f"No reconstruction available at {predictions_path}. "
                "Please run 'Reconstruct' first."
            )
            return gr.update(), error_message

        loaded = np.load(predictions_path, allow_pickle=True)
        predictions = {key: loaded[key] for key in loaded.keys()}  # noqa: F841

        return (
            glbfile,
            "Visualization updated.",
        )

    def handle_uploads(
        self,
        input_video: Optional[str],
        input_images: Optional[List],
        s_time_interval: float = 10.0,
    ) -> Tuple[Optional[str], Optional[str], Optional[List], Optional[str]]:
        """
        Handle file uploads and update gallery.

        Args:
            input_video: Path to input video file
            input_images: List of input image files
            s_time_interval: Sampling FPS (frames per second) for frame extraction

        Returns:
            Tuple of (reconstruction_output, target_dir, image_paths, log_message)
        """
        return self.file_handler.update_gallery_on_upload(
            input_video, input_images, s_time_interval
        )

    def load_example_scene(self, scene_name: str, examples_dir: str = None) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[List],
        str,
        Optional[Dict],
        gr.Dropdown,
        Optional[str],
        gr.update,
        gr.update,
    ]:
        """
        Load a scene from examples directory.

        Args:
            scene_name: Name of the scene to load
            examples_dir: Path to examples directory (if None, uses workspace_dir/examples)

        Returns:
            Tuple of (reconstruction_output, target_dir, image_paths, log_message, processed_data, measure_view_selector, gs_video, gs_video_vis, gs_info_vis)  # noqa: E501
        """
        if examples_dir is None:
            # Get workspace directory from environment variable
            workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "gradio_workspace")
            examples_dir = os.path.join(workspace_dir, "examples")

        reconstruction_output, target_dir, image_paths, log_message = (
            self.file_handler.load_example_scene(scene_name, examples_dir)
        )

        # Try to load cached processed data if available
        processed_data = None
        measure_view_selector = gr.Dropdown(choices=["View 1"], value="View 1")
        gs_video_path = None
        gs_video_visible = False
        gs_info_visible = True

        if target_dir and target_dir != "None":
            predictions_path = os.path.join(target_dir, "predictions.npz")
            if os.path.exists(predictions_path):
                try:
                    # Load predictions from cache
                    loaded = np.load(predictions_path, allow_pickle=True)
                    predictions = {key: loaded[key] for key in loaded.keys()}

                    # Reconstruct processed_data structure
                    num_images = len(predictions.get("images", []))
                    processed_data = {}

                    for i in range(num_images):
                        processed_data[i] = {
                            "image": predictions["images"][i] if "images" in predictions else None,
                            "depth": predictions["depths"][i] if "depths" in predictions else None,
                            "depth_image": os.path.join(
                                target_dir, "depth_vis", f"{i:04d}.jpg"  # Fixed: use .jpg not .png
                            ),
                            "intrinsics": (
                                predictions["intrinsics"][i]
                                if "intrinsics" in predictions
                                and i < len(predictions["intrinsics"])
                                else None
                            ),
                            "mask": None,
                        }

                    # Update measure view selector
                    choices = [f"View {i + 1}" for i in range(num_images)]
                    measure_view_selector = gr.Dropdown(choices=choices, value=choices[0])

                except Exception as e:
                    print(f"Error loading cached data: {e}")

            # Check for cached 3DGS video
            gs_video_dir = os.path.join(target_dir, "gs_video")
            if os.path.exists(gs_video_dir):
                try:
                    from glob import glob

                    gs_videos = sorted(glob(os.path.join(gs_video_dir, "*.mp4")))
                    if gs_videos:
                        gs_video_path = gs_videos[-1]
                        gs_video_visible = True
                        gs_info_visible = False
                        print(f"Loaded cached 3DGS video: {gs_video_path}")
                except Exception as e:
                    print(f"Error loading cached 3DGS video: {e}")

        return (
            reconstruction_output,
            target_dir,
            image_paths,
            log_message,
            processed_data,
            measure_view_selector,
            gs_video_path,
            gr.update(visible=gs_video_visible),
            gr.update(visible=gs_info_visible),
        )

    def navigate_depth_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector: str,
        direction: int,
    ) -> Tuple[str, Optional[str]]:
        """
        Navigate depth view.

        Args:
            processed_data: Processed data dictionary
            current_selector: Current selector value
            direction: Direction to navigate

        Returns:
            Tuple of (new_selector_value, depth_vis)
        """
        return self.visualization_handler.navigate_depth_view(
            processed_data, current_selector, direction
        )

    def update_depth_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[str]:
        """
        Update depth view for a specific view index.

        Args:
            processed_data: Processed data dictionary
            view_index: Index of the view to update

        Returns:
            Path to depth visualization image or None
        """
        return self.visualization_handler.update_depth_view(processed_data, view_index)

    def navigate_measure_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector: str,
        direction: int,
    ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], List]:
        """
        Navigate measure view.

        Args:
            processed_data: Processed data dictionary
            current_selector: Current selector value
            direction: Direction to navigate

        Returns:
            Tuple of (new_selector_value, measure_image, depth_right_half, measure_points)
        """
        return self.visualization_handler.navigate_measure_view(
            processed_data, current_selector, direction
        )

    def update_measure_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        """
        Update measure view for a specific view index.

        Args:
            processed_data: Processed data dictionary
            view_index: Index of the view to update

        Returns:
            Tuple of (measure_image, depth_right_half, measure_points)
        """
        return self.visualization_handler.update_measure_view(processed_data, view_index)

    def measure(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        measure_points: List,
        current_view_selector: str,
        event: gr.SelectData,
    ) -> List:
        """
        Handle measurement on images.

        Args:
            processed_data: Processed data dictionary
            measure_points: List of current measure points
            current_view_selector: Current view selector value
            event: Gradio select event

        Returns:
            List of [image, depth_right_half, measure_points, text]
        """
        return self.visualization_handler.measure(
            processed_data, measure_points, current_view_selector, event
        )

    def select_first_frame(
        self, image_gallery: List, selected_index: int = 0
    ) -> Tuple[List, str, str]:
        """
        Select the first frame from the image gallery.

        Args:
            image_gallery: List of images in the gallery
            selected_index: Index of the selected image (default: 0)

        Returns:
            Tuple of (updated_image_gallery, log_message, selected_frame_path)
        """
        try:
            if not image_gallery or len(image_gallery) == 0:
                return image_gallery, "No images available to select as first frame.", ""

            # Handle None or invalid selected_index
            if (
                selected_index is None
                or selected_index < 0
                or selected_index >= len(image_gallery)
            ):
                selected_index = 0
                print(f"Invalid selected_index: {selected_index}, using default: 0")

            # Get the selected image based on index
            selected_image = image_gallery[selected_index]
            print(f"Selected image index: {selected_index}")
            print(f"Total images: {len(image_gallery)}")

            # Extract the file path from the selected image
            selected_frame_path = ""
            print(f"Selected image type: {type(selected_image)}")
            print(f"Selected image: {selected_image}")

            if isinstance(selected_image, tuple):
                # Gradio Gallery returns tuple (path, None)
                selected_frame_path = selected_image[0]
            elif isinstance(selected_image, str):
                selected_frame_path = selected_image
            elif hasattr(selected_image, "name"):
                selected_frame_path = selected_image.name
            elif isinstance(selected_image, dict):
                if "name" in selected_image:
                    selected_frame_path = selected_image["name"]
                elif "path" in selected_image:
                    selected_frame_path = selected_image["path"]
                elif "src" in selected_image:
                    selected_frame_path = selected_image["src"]
            else:
                # Try to convert to string
                selected_frame_path = str(selected_image)

            print(f"Extracted path: {selected_frame_path}")

            # Extract filename from the path for matching
            import os

            selected_filename = os.path.basename(selected_frame_path)
            print(f"Selected filename: {selected_filename}")

            # Move the selected image to the front
            updated_gallery = [selected_image] + [
                img for img in image_gallery if img != selected_image
            ]

            log_message = (
                f"Selected frame: {selected_filename}. "
                f"Moved to first position. Total frames: {len(updated_gallery)}"
            )
            return updated_gallery, log_message, selected_filename

        except Exception as e:
            print(f"Error selecting first frame: {e}")
            return image_gallery, f"Error selecting first frame: {e}", ""
