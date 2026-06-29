# flake8: noqa: E402
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
Refactored Depth Anything 3 CLI
Clean, modular command-line interface
"""

from __future__ import annotations

import os
import typer

from depth_anything_3.services import start_server
from depth_anything_3.services.gallery import gallery as gallery_main
from depth_anything_3.services.inference_service import run_inference
from depth_anything_3.services.input_handlers import (
    ColmapHandler,
    ImageHandler,
    ImagesHandler,
    InputHandler,
    VideoHandler,
    parse_export_feat,
)
from depth_anything_3.utils.constants import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_GALLERY_DIR,
    DEFAULT_GRADIO_DIR,
    DEFAULT_MODEL,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

app = typer.Typer(help="Depth Anything 3 - Video depth estimation CLI", add_completion=False)


# ============================================================================
# Input type detection utilities
# ============================================================================

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def detect_input_type(input_path: str) -> str:
    """
    Detect input type from path.

    Returns:
        - "image": Single image file
        - "images": Directory containing images
        - "video": Video file
        - "colmap": COLMAP directory structure
        - "unknown": Cannot determine type
    """
    if not os.path.exists(input_path):
        return "unknown"

    # Check if it's a file
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

    # Check if it's a directory
    if os.path.isdir(input_path):
        # Check for COLMAP structure
        images_dir = os.path.join(input_path, "images")
        sparse_dir = os.path.join(input_path, "sparse")

        if os.path.isdir(images_dir) and os.path.isdir(sparse_dir):
            return "colmap"

        # Check if directory contains image files
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    return "images"

        return "unknown"

    return "unknown"


# ============================================================================
# Common parameters and configuration
# ============================================================================

# ============================================================================
# Inference commands
# ============================================================================


@app.command()
def auto(
    input_path: str = typer.Argument(
        ..., help="Path to input (image, directory, video, or COLMAP)"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option("cuda", help="Device to use"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS]Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Video-specific options
    fps: float = typer.Option(1.0, help="[Video] Sampling FPS for frame extraction"),
    # COLMAP-specific options
    sparse_subdir: str = typer.Option(
        "", help="[COLMAP] Sparse reconstruction subdirectory (e.g., '0' for sparse/0/)"
    ),
    align_to_input_ext_scale: bool = typer.Option(
        True, help="[COLMAP] Align prediction to input extrinsics scale"
    ),
    # Pose estimation options
    use_ray_pose: bool = typer.Option(
        False, help="Use ray-based pose estimation instead of camera decoder"
    ),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        help="Reference view selection strategy: empty, first, middle, saddle_balanced, saddle_sim_range",
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """
    Automatically detect input type and run appropriate processing.

    Supports:
    - Single image file (.jpg, .png, etc.)
    - Directory of images
    - Video file (.mp4, .avi, etc.)
    - COLMAP directory (with 'images' and 'sparse' subdirectories)
    """
    # Detect input type
    input_type = detect_input_type(input_path)

    if input_type == "unknown":
        typer.echo(f"❌ Error: Cannot determine input type for: {input_path}", err=True)
        typer.echo("Supported inputs:", err=True)
        typer.echo("  - Single image file (.jpg, .png, etc.)", err=True)
        typer.echo("  - Directory containing images", err=True)
        typer.echo("  - Video file (.mp4, .avi, etc.)", err=True)
        typer.echo("  - COLMAP directory (with 'images/' and 'sparse/' subdirectories)", err=True)
        raise typer.Exit(1)

    # Display detected type
    typer.echo(f"🔍 Detected input type: {input_type.upper()}")
    typer.echo(f"📁 Input path: {input_path}")
    typer.echo()

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Route to appropriate handler
    if input_type == "image":
        typer.echo("Processing single image...")
        # Process input
        image_files = ImageHandler.process(input_path)

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "images":
        typer.echo("Processing directory of images...")
        # Process input - use default extensions
        image_files = ImagesHandler.process(input_path, "png,jpg,jpeg")

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "video":
        typer.echo(f"Processing video with FPS={fps}...")
        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Process input
        image_files = VideoHandler.process(input_path, export_dir, fps)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    elif input_type == "colmap":
        typer.echo(
            f"Processing COLMAP directory (sparse subdirectory: '{sparse_subdir or 'default'}')..."
        )
        # Process input
        image_files, extrinsics, intrinsics = ColmapHandler.process(input_path, sparse_subdir)

        # Handle export directory
        export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

        # Run inference
        run_inference(
            image_paths=image_files,
            export_dir=export_dir,
            model_dir=model_dir,
            device=device,
            backend_url=final_backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )

    typer.echo()
    typer.echo("✅ Processing completed successfully!")


@app.command()
def image(
    image_path: str = typer.Argument(..., help="Path to input image file"),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option("cuda", help="Device to use"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Pose estimation options
    use_ray_pose: bool = typer.Option(
        False, help="Use ray-based pose estimation instead of camera decoder"
    ),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        help="Reference view selection strategy: empty, first, middle, saddle_balanced, saddle_sim_range",
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run camera pose and depth estimation on a single image."""
    # Process input
    image_files = ImageHandler.process(image_path)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


@app.command()
def images(
    images_dir: str = typer.Argument(..., help="Path to directory containing input images"),
    image_extensions: str = typer.Option(
        "png,jpg,jpeg", help="Comma-separated image file extensions to process"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option("cuda", help="Device to use"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Pose estimation options
    use_ray_pose: bool = typer.Option(
        False, help="Use ray-based pose estimation instead of camera decoder"
    ),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        help="Reference view selection strategy: empty, first, middle, saddle_balanced, saddle_sim_range",
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run camera pose and depth estimation on a directory of images."""
    # Process input
    image_files = ImagesHandler.process(images_dir, image_extensions)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


@app.command()
def colmap(
    colmap_dir: str = typer.Argument(
        ..., help="Path to COLMAP directory containing 'images' and 'sparse' subdirectories"
    ),
    sparse_subdir: str = typer.Option(
        "", help="Sparse reconstruction subdirectory (e.g., '0' for sparse/0/, empty for sparse/)"
    ),
    align_to_input_ext_scale: bool = typer.Option(
        True, help="Align prediction to input extrinsics scale"
    ),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option("cuda", help="Device to use"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Pose estimation options
    use_ray_pose: bool = typer.Option(
        False, help="Use ray-based pose estimation instead of camera decoder"
    ),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        help="Reference view selection strategy: empty, first, middle, saddle_balanced, saddle_sim_range",
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run pose conditioned depth estimation on COLMAP data."""
    # Process input
    image_files, extrinsics, intrinsics = ColmapHandler.process(colmap_dir, sparse_subdir)

    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=align_to_input_ext_scale,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


@app.command()
def video(
    video_path: str = typer.Argument(..., help="Path to input video file"),
    fps: float = typer.Option(1.0, help="Sampling FPS for frame extraction"),
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    export_dir: str = typer.Option(DEFAULT_EXPORT_DIR, help="Export directory"),
    export_format: str = typer.Option("glb", help="Export format"),
    device: str = typer.Option("cuda", help="Device to use"),
    use_backend: bool = typer.Option(False, help="Use backend service for inference"),
    backend_url: str = typer.Option(
        "http://localhost:8008", help="Backend URL (default: http://localhost:8008)"
    ),
    process_res: int = typer.Option(504, help="Processing resolution"),
    process_res_method: str = typer.Option(
        "upper_bound_resize", help="Processing resolution method"
    ),
    export_feat: str = typer.Option(
        "",
        help="[FEAT_VIS] Export features from specified layers using comma-separated indices (e.g., '0,1,2').",
    ),
    auto_cleanup: bool = typer.Option(
        False, help="Automatically clean export directory if it exists (no prompt)"
    ),
    # Pose estimation options
    use_ray_pose: bool = typer.Option(
        False, help="Use ray-based pose estimation instead of camera decoder"
    ),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        help="Reference view selection strategy: empty, first, middle, saddle_balanced, saddle_sim_range",
    ),
    # GLB export options
    conf_thresh_percentile: float = typer.Option(
        40.0, help="[GLB] Lower percentile for adaptive confidence threshold"
    ),
    num_max_points: int = typer.Option(
        1_000_000, help="[GLB] Maximum number of points in the point cloud"
    ),
    show_cameras: bool = typer.Option(
        True, help="[GLB] Show camera wireframes in the exported scene"
    ),
    # Feat_vis export options
    feat_vis_fps: int = typer.Option(15, help="[FEAT_VIS] Frame rate for output video"),
):
    """Run depth estimation on video by extracting frames and processing them."""
    # Handle export directory
    export_dir = InputHandler.handle_export_dir(export_dir, auto_cleanup)

    # Process input
    image_files = VideoHandler.process(video_path, export_dir, fps)

    # Parse export_feat parameter
    export_feat_layers = parse_export_feat(export_feat)

    # Determine backend URL based on use_backend flag
    final_backend_url = backend_url if use_backend else None

    # Run inference
    run_inference(
        image_paths=image_files,
        export_dir=export_dir,
        model_dir=model_dir,
        device=device,
        backend_url=final_backend_url,
        export_format=export_format,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )


# ============================================================================
# Service management commands
# ============================================================================


@app.command()
def backend(
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    device: str = typer.Option("cuda", help="Device to use"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8008, help="Port to bind to"),
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery directory path (optional)"),
):
    """Start model backend service with integrated gallery."""
    typer.echo("=" * 60)
    typer.echo("🚀 Starting Depth Anything 3 Backend Server")
    typer.echo("=" * 60)
    typer.echo(f"Model directory: {model_dir}")
    typer.echo(f"Device: {device}")

    # Check if gallery directory exists
    if gallery_dir and os.path.exists(gallery_dir):
        typer.echo(f"Gallery directory: {gallery_dir}")
    else:
        gallery_dir = None  # Disable gallery if directory doesn't exist

    typer.echo()
    typer.echo("📡 Server URLs (Ctrl/CMD+Click to open):")
    typer.echo(f"  🏠 Home:      http://{host}:{port}")
    typer.echo(f"  📊 Dashboard: http://{host}:{port}/dashboard")
    typer.echo(f"  📈 API Status: http://{host}:{port}/status")

    if gallery_dir:
        typer.echo(f"  🎨 Gallery:   http://{host}:{port}/gallery/")

    typer.echo("=" * 60)

    try:
        start_server(model_dir, device, host, port, gallery_dir)
    except KeyboardInterrupt:
        typer.echo("\n👋 Backend server stopped.")
    except Exception as e:
        typer.echo(f"❌ Failed to start backend: {e}")
        raise typer.Exit(1)


# ============================================================================
# Application launch commands
# ============================================================================


@app.command()
def gradio(
    model_dir: str = typer.Option(DEFAULT_MODEL, help="Model directory path"),
    workspace_dir: str = typer.Option(DEFAULT_GRADIO_DIR, help="Workspace directory path"),
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery directory path"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(7860, help="Port number to bind to"),
    share: bool = typer.Option(False, help="Create a public link for the app"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
    cache_examples: bool = typer.Option(
        False, help="Pre-cache all example scenes at startup for faster loading"
    ),
    cache_gs_tag: str = typer.Option(
        "",
        help="Tag to match scene names for high-res+3DGS caching (e.g., 'dl3dv'). Scenes containing this tag will use high_res and infer_gs=True; others will use low_res only.",
    ),
):
    """Launch Depth Anything 3 Gradio interactive web application"""
    from depth_anything_3.app.gradio_app import DepthAnything3App

    # Create necessary directories
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    typer.echo("Launching Depth Anything 3 Gradio application...")
    typer.echo(f"Model directory: {model_dir}")
    typer.echo(f"Workspace directory: {workspace_dir}")
    typer.echo(f"Gallery directory: {gallery_dir}")
    typer.echo(f"Host: {host}")
    typer.echo(f"Port: {port}")
    typer.echo(f"Share: {share}")
    typer.echo(f"Debug mode: {debug}")
    typer.echo(f"Cache examples: {cache_examples}")
    if cache_examples:
        if cache_gs_tag:
            typer.echo(
                f"Cache GS Tag: '{cache_gs_tag}' (scenes matching this tag will use high-res + 3DGS)"
            )
        else:
            typer.echo(f"Cache GS Tag: None (all scenes will use low-res only)")

    try:
        # Initialize and launch application
        app = DepthAnything3App(
            model_dir=model_dir, workspace_dir=workspace_dir, gallery_dir=gallery_dir
        )

        # Pre-cache examples if requested
        if cache_examples:
            typer.echo("\n" + "=" * 60)
            typer.echo("Pre-caching mode enabled")
            if cache_gs_tag:
                typer.echo(f"Scenes containing '{cache_gs_tag}' will use HIGH-RES + 3DGS")
                typer.echo(f"Other scenes will use LOW-RES only")
            else:
                typer.echo(f"All scenes will use LOW-RES only")
            typer.echo("=" * 60)
            app.cache_examples(
                show_cam=True,
                filter_black_bg=False,
                filter_white_bg=False,
                save_percentage=20.0,
                num_max_points=1000,
                cache_gs_tag=cache_gs_tag,
                gs_trj_mode="smooth",
                gs_video_quality="low",
            )

        # Prepare launch arguments
        launch_kwargs = {"share": share, "debug": debug}

        app.launch(host=host, port=port, **launch_kwargs)

    except KeyboardInterrupt:
        typer.echo("\nGradio application stopped.")
    except Exception as e:
        typer.echo(f"Failed to launch Gradio application: {e}")
        raise typer.Exit(1)


@app.command()
def gallery(
    gallery_dir: str = typer.Option(DEFAULT_GALLERY_DIR, help="Gallery root directory"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind to"),
    port: int = typer.Option(8007, help="Port number to bind to"),
    open_browser: bool = typer.Option(False, help="Open browser after launch"),
):
    """Launch Depth Anything 3 Gallery server"""

    # Validate gallery directory
    if not os.path.exists(gallery_dir):
        raise typer.BadParameter(f"Gallery directory not found: {gallery_dir}")

    typer.echo("Launching Depth Anything 3 Gallery server...")
    typer.echo(f"Gallery directory: {gallery_dir}")
    typer.echo(f"Host: {host}")
    typer.echo(f"Port: {port}")
    typer.echo(f"Auto-open browser: {open_browser}")

    try:
        # Set command line arguments
        import sys

        sys.argv = ["gallery", "--dir", gallery_dir, "--host", host, "--port", str(port)]
        if open_browser:
            sys.argv.append("--open")

        # Launch gallery server
        gallery_main()

    except KeyboardInterrupt:
        typer.echo("\nGallery server stopped.")
    except Exception as e:
        typer.echo(f"Failed to launch Gallery server: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
