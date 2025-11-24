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
UI components module for Depth Anything 3 Gradio app.

This module contains UI component definitions and layout functions.
"""

import os
from typing import Any, Dict, List, Tuple
import gradio as gr

from depth_anything_3.app.modules.utils import get_logo_base64, get_scene_info


class UIComponents:
    """
    Handles UI component creation and layout for the Gradio app.
    """

    def __init__(self):
        """Initialize the UI components handler."""

    def create_upload_section(self) -> Tuple[gr.Video, gr.Slider, gr.File, gr.Gallery, gr.Button]:
        """
        Create the upload section with video, images, and gallery components.

        Returns:
            A tuple of Gradio components: (input_video, s_time_interval, input_images,
            image_gallery, select_first_frame_btn).
        """
        input_video = gr.Video(label="Upload Video", interactive=True)
        s_time_interval = gr.Slider(
            minimum=0.1,
            maximum=60,
            value=10,
            step=0.1,
            label="Sampling FPS (Frames Per Second)",
            interactive=True,
            visible=True,
        )
        input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
        image_gallery = gr.Gallery(
            label="Preview",
            columns=4,
            height="300px",
            show_download_button=True,
            object_fit="contain",
            preview=True,
            interactive=False,
        )

        # Select first frame button (moved below image gallery)
        select_first_frame_btn = gr.Button("Select First Frame", scale=1)

        return input_video, s_time_interval, input_images, image_gallery, select_first_frame_btn

    def create_3d_viewer_section(self) -> gr.Model3D:
        """
        Create the 3D viewer component.

        Returns:
            3D model viewer component
        """
        return gr.Model3D(
            height=520,
            zoom_speed=0.5,
            pan_speed=0.5,
            clear_color=[0.0, 0.0, 0.0, 0.0],
            key="persistent_3d_viewer",
            elem_id="reconstruction_3d_viewer",
        )

    def create_nvs_video(self) -> Tuple[gr.Video, gr.Markdown]:
        """
        Create the 3DGS rendered video display component and info message.

        Returns:
            Tuple of (video component, info message component)
        """
        with gr.Column():
            gs_info = gr.Markdown(
                (
                    "‼️ **3D Gaussian Splatting rendering is currently DISABLED.** <br><br><br>"
                    "To render novel views from 3DGS, "
                    "enable **Infer 3D Gaussian Splatting** below. <br>"
                    "Next, in **Visualization Options**, "
                    "*optionally* configure the **rendering trajectory** (default: smooth) "
                    "and **video quality** (default: low), "
                    "then click **Reconstruct**."
                ),
                visible=True,
                height=520,
            )
            gs_video = gr.Video(
                height=520,
                label="3DGS Rendered NVS Video (depth shown for reference only)",
                interactive=False,
                visible=False,
            )
        return gs_video, gs_info

    def create_depth_section(self) -> Tuple[gr.Button, gr.Dropdown, gr.Button, gr.Image]:
        """
        Create the depth visualization section.

        Returns:
            A tuple of (prev_depth_btn, depth_view_selector, next_depth_btn, depth_map)
        """
        with gr.Row(elem_classes=["navigation-row"]):
            prev_depth_btn = gr.Button("◀ Previous", size="sm", scale=1)
            depth_view_selector = gr.Dropdown(
                choices=["View 1"],
                value="View 1",
                label="Select View",
                scale=2,
                interactive=True,
                allow_custom_value=True,
            )
            next_depth_btn = gr.Button("Next ▶", size="sm", scale=1)
        depth_map = gr.Image(
            type="numpy",
            label="Colorized Depth Map",
            format="png",
            interactive=False,
        )

        return prev_depth_btn, depth_view_selector, next_depth_btn, depth_map

    def create_measure_section(
        self,
    ) -> Tuple[gr.Button, gr.Dropdown, gr.Button, gr.Image, gr.Image, gr.Markdown]:
        """
        Create the measurement section.

        Returns:
            A tuple of (prev_measure_btn, measure_view_selector, next_measure_btn, measure_image,
            measure_depth_image, measure_text)
        """
        from depth_anything_3.app.css_and_html import MEASURE_INSTRUCTIONS_HTML

        gr.Markdown(MEASURE_INSTRUCTIONS_HTML)
        with gr.Row(elem_classes=["navigation-row"]):
            prev_measure_btn = gr.Button("◀ Previous", size="sm", scale=1)
            measure_view_selector = gr.Dropdown(
                choices=["View 1"],
                value="View 1",
                label="Select View",
                scale=2,
                interactive=True,
                allow_custom_value=True,
            )
            next_measure_btn = gr.Button("Next ▶", size="sm", scale=1)
        with gr.Row():
            measure_image = gr.Image(
                type="numpy",
                show_label=False,
                format="webp",
                interactive=False,
                sources=[],
                label="RGB Image",
                scale=1,
                height=275,
            )
            measure_depth_image = gr.Image(
                type="numpy",
                show_label=False,
                format="webp",
                interactive=False,
                sources=[],
                label="Depth Visualization (Right Half)",
                scale=1,
                height=275,
            )
        gr.Markdown(
            "**Note:** Images have been adjusted to model processing size. "
            "Click two points on the RGB image to measure distance."
        )
        measure_text = gr.Markdown("")

        return (
            prev_measure_btn,
            measure_view_selector,
            next_measure_btn,
            measure_image,
            measure_depth_image,
            measure_text,
        )

    def create_inference_control_section(
        self,
    ) -> Tuple[gr.Dropdown, gr.Checkbox, gr.Number, gr.Dropdown]:
        """
        Create the inference control section (before inference).

        Returns:
            Tuple of (process_res_method_dropdown, infer_gs, batch_size, mixed_precision)
        """
        with gr.Row():
            process_res_method_dropdown = gr.Dropdown(
                choices=["high_res", "low_res"],
                value="low_res",
                label="Image Processing Method",
                info="low_res for much more images",
                scale=1,
            )
            infer_gs = gr.Checkbox(
                label="Infer 3D Gaussian Splatting",
                value=False,
                info=(
                    'Enable novel view rendering from 3DGS (<i class="fas fa-triangle-exclamation '
                    'fa-color-red"></i> requires extra processing time)'
                ),
                scale=1,
            )

        with gr.Row():
            batch_size = gr.Number(
                label="Batch size (sub-batching)",
                value=None,
                precision=0,
                minimum=1,
                maximum=64,
                step=1,
                info="Process images in chunks to limit memory. Leave empty for all-at-once.",
            )
            mixed_precision = gr.Dropdown(
                choices=["auto", "fp16", "fp32", "bf16"],
                value="auto",
                label="Mixed precision",
                info="MPS: default fp32; fp16 opt-in. CUDA: auto uses bf16/fp16.",
            )

        return (process_res_method_dropdown, infer_gs, batch_size, mixed_precision)

    def create_display_control_section(
        self,
    ) -> Tuple[
        gr.Checkbox,
        gr.Checkbox,
        gr.Checkbox,
        gr.Slider,
        gr.Slider,
        gr.Dropdown,
        gr.Dropdown,
        gr.Button,
        gr.ClearButton,
    ]:
        """
        Create the display control section (options for visualization).

        Returns:
            Tuple of display control components including buttons
        """
        with gr.Column():
            # 3DGS options at the top
            with gr.Row():
                gs_trj_mode = gr.Dropdown(
                    choices=["smooth", "extend"],
                    value="smooth",
                    label=("Rendering trajectory for 3DGS viewpoints (requires n_views ≥ 2)"),
                    info=("'smooth' for view interpolation; 'extend' for longer trajectory"),
                    visible=False,  # initially hidden
                )
                gs_video_quality = gr.Dropdown(
                    choices=["low", "medium", "high"],
                    value="low",
                    label=("Video quality for 3DGS rendered outputs"),
                    info=("'low' for faster loading speed; 'high' for better visual quality"),
                    visible=False,  # initially hidden
                )

            # Reconstruct and Clear buttons (before Visualization Options)
            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(scale=1)

            gr.Markdown("### Visualization Options: (Click Reconstruct to update)")
            show_cam = gr.Checkbox(label="Show Camera", value=True)
            filter_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
            filter_white_bg = gr.Checkbox(label="Filter White Background", value=False)
            save_percentage = gr.Slider(
                minimum=0,
                maximum=100,
                value=10,
                step=1,
                label="Filter Percentage",
                info="Confidence Threshold (%): Higher values filter more points.",
            )
            num_max_points = gr.Slider(
                minimum=1000,
                maximum=100000,
                value=1000,
                step=1000,
                label="Max Points (K points)",
                info="Maximum number of points to export to GLB (in thousands)",
            )

        return (
            show_cam,
            filter_black_bg,
            filter_white_bg,
            save_percentage,
            num_max_points,
            gs_trj_mode,
            gs_video_quality,
            submit_btn,
            clear_btn,
        )

    def create_control_section(
        self,
    ) -> Tuple[
        gr.Button,
        gr.ClearButton,
        gr.Dropdown,
        gr.Checkbox,
        gr.Checkbox,
        gr.Checkbox,
        gr.Checkbox,
        gr.Checkbox,
        gr.Dropdown,
        gr.Checkbox,
        gr.Textbox,
    ]:
        """
        Create the control section with buttons and options.

        Returns:
            Tuple of control components
        """
        with gr.Row():
            submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
            clear_btn = gr.ClearButton(
                scale=1,
            )

        with gr.Row():
            frame_filter = gr.Dropdown(
                choices=["All"], value="All", label="Show Points from Frame"
            )
            with gr.Column():
                gr.Markdown("### Visualization Option: (Click Reconstruct to update)")
                show_cam = gr.Checkbox(label="Show Camera", value=True)
                show_mesh = gr.Checkbox(label="Show Mesh", value=True)
                filter_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                filter_white_bg = gr.Checkbox(label="Filter White Background", value=False)
                gr.Markdown("### Reconstruction Options: (updated on next run)")
                apply_mask_checkbox = gr.Checkbox(
                    label="Apply mask for predicted ambiguous depth classes & edges",
                    value=True,
                )
                process_res_method_dropdown = gr.Dropdown(
                    choices=[
                        "upper_bound_resize",
                        "upper_bound_crop",
                        "lower_bound_resize",
                        "lower_bound_crop",
                    ],
                    value="upper_bound_resize",
                    label="Image Processing Method",
                    info="Method for resizing input images",
                )
                save_to_gallery_checkbox = gr.Checkbox(
                    label="Save to Gallery",
                    value=False,
                    info="Save current reconstruction results to gallery directory",
                )
                gallery_name_input = gr.Textbox(
                    label="Gallery Name",
                    placeholder="Enter a name for the gallery folder",
                    value="",
                    info="Leave empty for auto-generated name with timestamp",
                )

        return (
            submit_btn,
            clear_btn,
            frame_filter,
            show_cam,
            show_mesh,
            filter_black_bg,
            filter_white_bg,
            apply_mask_checkbox,
            process_res_method_dropdown,
            save_to_gallery_checkbox,
            gallery_name_input,
        )

    def create_example_scenes_section(self) -> List[Dict[str, Any]]:
        """
        Create the example scenes section.

        Returns:
            List of scene information dictionaries
        """
        # Get workspace directory from environment variable
        workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "gradio_workspace")
        examples_dir = os.path.join(workspace_dir, "examples")

        # Get scene information
        scenes = get_scene_info(examples_dir)

        return scenes

    def create_example_scene_grid(self, scenes: List[Dict[str, Any]]) -> List[gr.Image]:
        """
        Create the example scene grid.

        Args:
            scenes: List of scene information dictionaries

        Returns:
            List of scene image components
        """
        scene_components = []

        if scenes:
            for i in range(0, len(scenes), 4):  # Process 4 scenes per row
                with gr.Row():
                    for j in range(4):
                        scene_idx = i + j
                        if scene_idx < len(scenes):
                            scene = scenes[scene_idx]
                            with gr.Column(scale=1, elem_classes=["clickable-thumbnail"]):
                                # Clickable thumbnail
                                scene_img = gr.Image(
                                    value=scene["thumbnail"],
                                    height=150,
                                    interactive=False,
                                    show_label=False,
                                    elem_id=f"scene_thumb_{scene['name']}",
                                    sources=[],
                                )
                                scene_components.append(scene_img)

                                # Scene name and image count as text below thumbnail
                                gr.Markdown(
                                    f"**{scene['name']}** \n {scene['num_images']} images",
                                    elem_classes=["scene-info"],
                                )
                        else:
                            # Empty column to maintain grid structure
                            with gr.Column(scale=1):
                                pass

        return scene_components

    def create_header_section(self) -> gr.HTML:
        """
        Create the header section with logo and title.

        Returns:
            Header HTML component
        """
        from depth_anything_3.app.css_and_html import get_header_html

        return gr.HTML(get_header_html(get_logo_base64()))

    def create_description_section(self) -> gr.HTML:
        """
        Create the description section.

        Returns:
            Description HTML component
        """
        from depth_anything_3.app.css_and_html import get_description_html

        return gr.HTML(get_description_html())

    def create_acknowledgements_section(self) -> gr.HTML:
        """
        Create the acknowledgements section.

        Returns:
            Acknowledgements HTML component
        """
        from depth_anything_3.app.css_and_html import get_acknowledgements_html

        return gr.HTML(get_acknowledgements_html())
