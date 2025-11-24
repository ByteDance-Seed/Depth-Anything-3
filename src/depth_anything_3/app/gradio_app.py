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
Refactored Gradio App for Depth Anything 3.

This is the main application file that orchestrates all components.
The original functionality has been split into modular components for better maintainability.
"""

import argparse
import os
from typing import Any, Dict, List
import gradio as gr

from depth_anything_3.app.css_and_html import GRADIO_CSS, get_gradio_theme
from depth_anything_3.app.modules.event_handlers import EventHandlers
from depth_anything_3.app.modules.ui_components import UIComponents

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DepthAnything3App:
    """
    Main application class for Depth Anything 3 Gradio app.
    """

    def __init__(self, model_dir: str = None, workspace_dir: str = None, gallery_dir: str = None):
        """
        Initialize the application.

        Args:
            model_dir: Path to the model directory
            workspace_dir: Path to the workspace directory
            gallery_dir: Path to the gallery directory
        """
        self.model_dir = model_dir
        self.workspace_dir = workspace_dir
        self.gallery_dir = gallery_dir

        # Set environment variables for directories
        if self.model_dir:
            os.environ["DA3_MODEL_DIR"] = self.model_dir
        if self.workspace_dir:
            os.environ["DA3_WORKSPACE_DIR"] = self.workspace_dir
        if self.gallery_dir:
            os.environ["DA3_GALLERY_DIR"] = self.gallery_dir

        self.event_handlers = EventHandlers()
        self.ui_components = UIComponents()

    def cache_examples(
        self,
        show_cam: bool = True,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        save_percentage: float = 20.0,
        num_max_points: int = 1000,
        cache_gs_tag: str = "",
        gs_trj_mode: str = "smooth",
        gs_video_quality: str = "low",
    ) -> None:
        """
        Pre-cache all example scenes at startup.

        Args:
            show_cam: Whether to show camera in visualization
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            save_percentage: Filter percentage for point cloud
            num_max_points: Maximum number of points
            cache_gs_tag: Tag to match scene names for high-res+3DGS caching (e.g., "dl3dv")
            gs_trj_mode: Trajectory mode for 3DGS
            gs_video_quality: Video quality for 3DGS
        """
        from depth_anything_3.app.modules.utils import get_scene_info

        examples_dir = os.path.join(self.workspace_dir, "examples")
        if not os.path.exists(examples_dir):
            print(f"Examples directory not found: {examples_dir}")
            return

        scenes = get_scene_info(examples_dir)
        if not scenes:
            print("No example scenes found to cache.")
            return

        print(f"\n{'='*60}")
        print(f"Caching {len(scenes)} example scenes...")
        print(f"{'='*60}\n")

        for i, scene in enumerate(scenes, 1):
            scene_name = scene["name"]

            # Check if scene name matches the gs tag for high-res+3DGS caching
            use_high_res_gs = cache_gs_tag and cache_gs_tag.lower() in scene_name.lower()

            if use_high_res_gs:
                print(f"[{i}/{len(scenes)}] Caching scene: {scene_name} (HIGH-RES + 3DGS)")
                print(f"  - Number of images: {scene['num_images']}")
                print(f"  - Matched tag: '{cache_gs_tag}' - using high_res + 3DGS")
            else:
                print(f"[{i}/{len(scenes)}] Caching scene: {scene_name} (LOW-RES)")
                print(f"  - Number of images: {scene['num_images']}")

            try:
                # Load example scene
                _, target_dir, _, _, _, _, _, _, _ = self.event_handlers.load_example_scene(
                    scene_name
                )

                if target_dir and target_dir != "None":
                    # Run reconstruction with appropriate settings
                    print("  - Running reconstruction...")
                    result = self.event_handlers.gradio_demo(
                        target_dir=target_dir,
                        show_cam=show_cam,
                        filter_black_bg=filter_black_bg,
                        filter_white_bg=filter_white_bg,
                        process_res_method="high_res" if use_high_res_gs else "low_res",
                        selected_first_frame="",
                        save_percentage=save_percentage,
                        num_max_points=num_max_points,
                        infer_gs=use_high_res_gs,
                        gs_trj_mode=gs_trj_mode,
                        gs_video_quality=gs_video_quality,
                    )

                    # Check if successful
                    if result[0] is not None:  # reconstruction_output
                        print(f"  ✓ Scene '{scene_name}' cached successfully")
                    else:
                        print(f"  ✗ Scene '{scene_name}' caching failed: {result[1]}")
                else:
                    print(f"  ✗ Scene '{scene_name}' loading failed")

            except Exception as e:
                print(f"  ✗ Error caching scene '{scene_name}': {str(e)}")

            print()

        print("=" * 60)
        print("Example scene caching completed!")
        print("=" * 60 + "\n")

    def create_app(self) -> gr.Blocks:
        """
        Create and configure the Gradio application.

        Returns:
            Configured Gradio Blocks interface
        """

        # Initialize theme
        def get_theme():
            return get_gradio_theme()

        with gr.Blocks(theme=get_theme(), css=GRADIO_CSS) as demo:
            # State variables for the tabbed interface
            is_example = gr.Textbox(label="is_example", visible=False, value="None")
            processed_data_state = gr.State(value=None)
            measure_points_state = gr.State(value=[])
            selected_first_frame_state = gr.State(value="")
            selected_image_index_state = gr.State(value=0)  # Track selected image index
            # current_view_index = gr.State(value=0)  # noqa: F841 Track current view index

            # Header and description
            self.ui_components.create_header_section()
            self.ui_components.create_description_section()

            target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

            # Main content area
            with gr.Row():
                with gr.Column(scale=2):
                    # Upload section
                    (
                        input_video,
                        s_time_interval,
                        input_images,
                        image_gallery,
                        select_first_frame_btn,
                    ) = self.ui_components.create_upload_section()

                with gr.Column(scale=4):
                    with gr.Column():
                        # gr.Markdown("**Metric 3D Reconstruction (Point Cloud and Camera Poses)**")
                        # Reconstruction control section (buttons) - moved below tabs

                        log_output = gr.Markdown(
                            "Please upload a video or images, then click Reconstruct.",
                            elem_classes=["custom-log"],
                        )

                        # Tabbed interface
                        with gr.Tabs():
                            with gr.Tab("Point Cloud & Cameras"):
                                reconstruction_output = (
                                    self.ui_components.create_3d_viewer_section()
                                )

                            with gr.Tab("Metric Depth"):
                                (
                                    prev_measure_btn,
                                    measure_view_selector,
                                    next_measure_btn,
                                    measure_image,
                                    measure_depth_image,
                                    measure_text,
                                ) = self.ui_components.create_measure_section()

                            with gr.Tab("3DGS Rendered Novel Views"):
                                gs_video, gs_info = self.ui_components.create_nvs_video()

                        # Inference control section (before inference)
                        (
                            process_res_method_dropdown,
                            infer_gs,
                            batch_size_ctrl,
                            mixed_precision_ctrl,
                        ) = self.ui_components.create_inference_control_section()

                        # Display control section - includes 3DGS options, buttons, and Visualization Options  # noqa: E501
                        (
                            show_cam,
                            filter_black_bg,
                            filter_white_bg,
                            save_percentage,
                            num_max_points,
                            gs_trj_mode,
                            gs_video_quality,
                            submit_btn,
                            clear_btn,
                        ) = self.ui_components.create_display_control_section()

                        # bind visibility of gs_trj_mode to infer_gs
                        infer_gs.change(
                            fn=lambda checked: (
                                gr.update(visible=checked),
                                gr.update(visible=checked),
                                gr.update(visible=checked),
                                gr.update(visible=(not checked)),
                            ),
                            inputs=infer_gs,
                            outputs=[gs_trj_mode, gs_video_quality, gs_video, gs_info],
                        )

            # Example scenes section
            gr.Markdown("## Example Scenes")

            scenes = self.ui_components.create_example_scenes_section()
            scene_components = self.ui_components.create_example_scene_grid(scenes)

            # Set up event handlers
            self._setup_event_handlers(
                demo,
                is_example,
                processed_data_state,
                measure_points_state,
                target_dir_output,
                input_video,
                input_images,
                s_time_interval,
                image_gallery,
                reconstruction_output,
                log_output,
                show_cam,
                filter_black_bg,
                filter_white_bg,
                process_res_method_dropdown,
                batch_size_ctrl,
                mixed_precision_ctrl,
                save_percentage,
                submit_btn,
                clear_btn,
                num_max_points,
                infer_gs,
                select_first_frame_btn,
                selected_first_frame_state,
                selected_image_index_state,
                measure_view_selector,
                measure_image,
                measure_depth_image,
                measure_text,
                prev_measure_btn,
                next_measure_btn,
                scenes,
                scene_components,
                gs_video,
                gs_info,
                gs_trj_mode,
                gs_video_quality,
            )

            # Acknowledgements
            self.ui_components.create_acknowledgements_section()

        return demo

    def _setup_event_handlers(
        self,
        demo: gr.Blocks,
        is_example: gr.Textbox,
        processed_data_state: gr.State,
        measure_points_state: gr.State,
        target_dir_output: gr.Textbox,
        input_video: gr.Video,
        input_images: gr.File,
        s_time_interval: gr.Slider,
        image_gallery: gr.Gallery,
        reconstruction_output: gr.Model3D,
        log_output: gr.Markdown,
        show_cam: gr.Checkbox,
        filter_black_bg: gr.Checkbox,
        filter_white_bg: gr.Checkbox,
        process_res_method_dropdown: gr.Dropdown,
        batch_size_ctrl: gr.Number,
        mixed_precision_ctrl: gr.Dropdown,
        save_percentage: gr.Slider,
        submit_btn: gr.Button,
        clear_btn: gr.ClearButton,
        num_max_points: gr.Slider,
        infer_gs: gr.Checkbox,
        select_first_frame_btn: gr.Button,
        selected_first_frame_state: gr.State,
        selected_image_index_state: gr.State,
        measure_view_selector: gr.Dropdown,
        measure_image: gr.Image,
        measure_depth_image: gr.Image,
        measure_text: gr.Markdown,
        prev_measure_btn: gr.Button,
        next_measure_btn: gr.Button,
        scenes: List[Dict[str, Any]],
        scene_components: List[gr.Image],
        gs_video: gr.Video,
        gs_info: gr.Markdown,
        gs_trj_mode: gr.Dropdown,
        gs_video_quality: gr.Dropdown,
    ) -> None:
        """
        Set up all event handlers for the application.

        Args:
            demo: Gradio Blocks interface
            All other arguments: Gradio components to connect
        """
        # Configure clear button
        clear_btn.add(
            [
                input_video,
                input_images,
                reconstruction_output,
                log_output,
                target_dir_output,
                image_gallery,
                gs_video,
            ]
        )

        # Main reconstruction button
        submit_btn.click(
            fn=self.event_handlers.clear_fields, inputs=[], outputs=[reconstruction_output]
        ).then(fn=self.event_handlers.update_log, inputs=[], outputs=[log_output]).then(
            fn=self.event_handlers.gradio_demo,
            inputs=[
                target_dir_output,
                show_cam,
                filter_black_bg,
                filter_white_bg,
                process_res_method_dropdown,
                batch_size_ctrl,
                mixed_precision_ctrl,
                selected_first_frame_state,
                save_percentage,
                # pass num_max_points
                num_max_points,
                infer_gs,
                gs_trj_mode,
                gs_video_quality,
            ],
            outputs=[
                reconstruction_output,
                log_output,
                processed_data_state,
                measure_image,
                measure_depth_image,
                measure_text,
                measure_view_selector,
                gs_video,
                gs_video,  # gs_video visibility
                gs_info,  # gs_info visibility
            ],
        ).then(
            fn=lambda: "False",
            inputs=[],
            outputs=[is_example],  # set is_example to "False"
        )

        # Real-time visualization updates
        self._setup_visualization_handlers(
            show_cam,
            filter_black_bg,
            filter_white_bg,
            process_res_method_dropdown,
            batch_size_ctrl,
            mixed_precision_ctrl,
            target_dir_output,
            is_example,
            reconstruction_output,
            log_output,
        )

        # File upload handlers
        input_video.change(
            fn=self.event_handlers.handle_uploads,
            inputs=[input_video, input_images, s_time_interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=self.event_handlers.handle_uploads,
            inputs=[input_video, input_images, s_time_interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        # Image gallery click handler (for selecting first frame)
        def handle_image_selection(evt: gr.SelectData):
            if evt is None or evt.index is None:
                return "No image selected", 0
            selected_index = evt.index
            return f"Selected image {selected_index} as potential first frame", selected_index

        image_gallery.select(
            fn=handle_image_selection,
            outputs=[log_output, selected_image_index_state],
        )

        # Select first frame handler
        select_first_frame_btn.click(
            fn=self.event_handlers.select_first_frame,
            inputs=[image_gallery, selected_image_index_state],
            outputs=[image_gallery, log_output, selected_first_frame_state],
        )

        # Navigation handlers
        self._setup_navigation_handlers(
            prev_measure_btn,
            next_measure_btn,
            measure_view_selector,
            measure_image,
            measure_depth_image,
            measure_points_state,
            processed_data_state,
        )

        # Measurement handler
        measure_image.select(
            fn=self.event_handlers.measure,
            inputs=[processed_data_state, measure_points_state, measure_view_selector],
            outputs=[measure_image, measure_depth_image, measure_points_state, measure_text],
        )

        # Example scene handlers
        self._setup_example_scene_handlers(
            scenes,
            scene_components,
            reconstruction_output,
            target_dir_output,
            image_gallery,
            log_output,
            is_example,
            processed_data_state,
            measure_view_selector,
            measure_image,
            measure_depth_image,
            gs_video,
            gs_info,
        )

    def _setup_visualization_handlers(
        self,
        show_cam: gr.Checkbox,
        filter_black_bg: gr.Checkbox,
        filter_white_bg: gr.Checkbox,
        process_res_method_dropdown: gr.Dropdown,
        batch_size_ctrl: gr.Number,
        mixed_precision_ctrl: gr.Dropdown,
        target_dir_output: gr.Textbox,
        is_example: gr.Textbox,
        reconstruction_output: gr.Model3D,
        log_output: gr.Markdown,
    ) -> None:
        """Set up visualization update handlers."""
        # Common inputs for visualization updates
        viz_inputs = [
            target_dir_output,
            show_cam,
            is_example,
            filter_black_bg,
            filter_white_bg,
            process_res_method_dropdown,
            batch_size_ctrl,
            mixed_precision_ctrl,
        ]

        # Set up change handlers for all visualization controls
        for component in [show_cam, filter_black_bg, filter_white_bg]:
            component.change(
                fn=self.event_handlers.update_visualization,
                inputs=viz_inputs,
                outputs=[reconstruction_output, log_output],
            )

    def _setup_navigation_handlers(
        self,
        prev_measure_btn: gr.Button,
        next_measure_btn: gr.Button,
        measure_view_selector: gr.Dropdown,
        measure_image: gr.Image,
        measure_depth_image: gr.Image,
        measure_points_state: gr.State,
        processed_data_state: gr.State,
    ) -> None:
        """Set up navigation handlers for measure tab."""
        # Measure tab navigation
        prev_measure_btn.click(
            fn=lambda processed_data, current_selector: self.event_handlers.navigate_measure_view(
                processed_data, current_selector, -1
            ),
            inputs=[processed_data_state, measure_view_selector],
            outputs=[
                measure_view_selector,
                measure_image,
                measure_depth_image,
                measure_points_state,
            ],
        )

        next_measure_btn.click(
            fn=lambda processed_data, current_selector: self.event_handlers.navigate_measure_view(
                processed_data, current_selector, 1
            ),
            inputs=[processed_data_state, measure_view_selector],
            outputs=[
                measure_view_selector,
                measure_image,
                measure_depth_image,
                measure_points_state,
            ],
        )

        measure_view_selector.change(
            fn=lambda processed_data, selector_value: (
                self.event_handlers.update_measure_view(
                    processed_data, int(selector_value.split()[1]) - 1
                )
                if selector_value
                else (None, None, [])
            ),
            inputs=[processed_data_state, measure_view_selector],
            outputs=[measure_image, measure_depth_image, measure_points_state],
        )

    def _setup_example_scene_handlers(
        self,
        scenes: List[Dict[str, Any]],
        scene_components: List[gr.Image],
        reconstruction_output: gr.Model3D,
        target_dir_output: gr.Textbox,
        image_gallery: gr.Gallery,
        log_output: gr.Markdown,
        is_example: gr.Textbox,
        processed_data_state: gr.State,
        measure_view_selector: gr.Dropdown,
        measure_image: gr.Image,
        measure_depth_image: gr.Image,
        gs_video: gr.Video,
        gs_info: gr.Markdown,
    ) -> None:
        """Set up example scene handlers."""

        def load_and_update_measure(name):
            result = self.event_handlers.load_example_scene(name)
            # result = (reconstruction_output, target_dir, image_paths, log_message, processed_data, measure_view_selector, gs_video, gs_video_vis, gs_info_vis)  # noqa: E501

            # Update measure view if processed_data is available
            measure_img = None
            measure_depth = None
            if result[4] is not None:  # processed_data exists
                measure_img, measure_depth, _ = (
                    self.event_handlers.visualization_handler.update_measure_view(result[4], 0)
                )

            return result + ("True", measure_img, measure_depth)

        for i, scene in enumerate(scenes):
            if i < len(scene_components):
                scene_components[i].select(
                    fn=lambda name=scene["name"]: load_and_update_measure(name),
                    outputs=[
                        reconstruction_output,
                        target_dir_output,
                        image_gallery,
                        log_output,
                        processed_data_state,
                        measure_view_selector,
                        gs_video,
                        gs_video,  # gs_video_visibility
                        gs_info,  # gs_info_visibility
                        is_example,
                        measure_image,
                        measure_depth_image,
                    ],
                )

    def launch(self, host: str = "127.0.0.1", port: int = 7860, **kwargs) -> None:
        """
        Launch the application.

        Args:
            host: Host address to bind to
            port: Port number to bind to
            **kwargs: Additional arguments for demo.launch()
        """
        demo = self.create_app()
        demo.queue(max_size=20).launch(
            show_error=True, ssr_mode=False, server_name=host, server_port=port, **kwargs
        )


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 Gradio Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python gradio_app.py --help
  python gradio_app.py --host 0.0.0.0 --port 8080
  python gradio_app.py --model-dir /path/to/model --workspace-dir /path/to/workspace

  # Cache examples at startup (all low-res)
  python gradio_app.py --cache-examples

  # Cache with selective high-res+3DGS for scenes matching tag
  python gradio_app.py --cache-examples --cache-gs-tag dl3dv
  # This will use high-res + 3DGS for scenes containing "dl3dv" in their name,
  # and low-res only for other scenes
        """,
    )

    # Server configuration
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port number to bind to (default: 7860)"
    )

    # Directory configuration
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3-BASE",
        help="Path to the model directory (default: depth-anything/DA3-BASE)",
    )
    parser.add_argument(
        "--workspace-dir",
        default="workspace/gradio",  # noqa: E501
        help="Path to the workspace directory (default: workspace/gradio)",  # noqa: E501
    )
    parser.add_argument(
        "--gallery-dir",
        default="workspace/gallery",
        help="Path to the gallery directory (default: workspace/gallery)",  # noqa: E501
    )

    # Additional Gradio options
    parser.add_argument("--share", action="store_true", help="Create a public link for the app")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Example caching options
    parser.add_argument(
        "--cache-examples",
        action="store_true",
        help="Pre-cache all example scenes at startup for faster loading",
    )
    parser.add_argument(
        "--cache-gs-tag",
        type=str,
        default="",
        help="Tag to match scene names for high-res+3DGS caching (e.g., 'dl3dv'). Scenes containing this tag will use high_res and infer_gs=True; others will use low_res only.",  # noqa: E501
    )

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.workspace_dir, exist_ok=True)
    os.makedirs(args.gallery_dir, exist_ok=True)

    # Initialize and launch the application
    app = DepthAnything3App(
        model_dir=args.model_dir, workspace_dir=args.workspace_dir, gallery_dir=args.gallery_dir
    )

    # Prepare launch arguments
    launch_kwargs = {"share": args.share, "debug": args.debug}

    print("Starting Depth Anything 3 Gradio App...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Workspace Directory: {args.workspace_dir}")
    print(f"Gallery Directory: {args.gallery_dir}")
    print(f"Share: {args.share}")
    print(f"Debug: {args.debug}")
    print(f"Cache Examples: {args.cache_examples}")
    if args.cache_examples:
        if args.cache_gs_tag:
            print(
                f"Cache GS Tag: '{args.cache_gs_tag}' (scenes matching this tag will use high-res + 3DGS)"  # noqa: E501
            )  # noqa: E501
        else:
            print("Cache GS Tag: None (all scenes will use low-res only)")

    # Pre-cache examples if requested
    if args.cache_examples:
        print("\n" + "=" * 60)
        print("Pre-caching mode enabled")
        if args.cache_gs_tag:
            print(f"Scenes containing '{args.cache_gs_tag}' will use HIGH-RES + 3DGS")
            print("Other scenes will use LOW-RES only")
        else:
            print("All scenes will use LOW-RES only")
        print("=" * 60)
        app.cache_examples(
            show_cam=True,
            filter_black_bg=False,
            filter_white_bg=False,
            save_percentage=5.0,
            num_max_points=1000,
            cache_gs_tag=args.cache_gs_tag,
            gs_trj_mode="smooth",
            gs_video_quality="low",
        )

    app.launch(host=args.host, port=args.port, **launch_kwargs)


if __name__ == "__main__":
    main()
