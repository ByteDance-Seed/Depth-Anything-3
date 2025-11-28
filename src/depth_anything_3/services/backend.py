# flake8: noqa: E501
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
Model backend service for Depth Anything 3.
Provides HTTP API for model inference with persistent model loading.
"""

import os
import posixpath
import time
import uuid

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import numpy as np

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..api import DepthAnything3
from ..utils.memory import (
    get_gpu_memory_info,
    cleanup_cuda_memory,
    check_memory_availability,
    estimate_memory_requirement,
)


class InferenceRequest(BaseModel):
    """Request model for inference API."""

    image_paths: List[str]
    export_dir: Optional[str] = None
    export_format: str = "mini_npz-glb"
    extrinsics: Optional[List[List[List[float]]]] = None
    intrinsics: Optional[List[List[List[float]]]] = None
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    export_feat_layers: List[int] = []
    align_to_input_ext_scale: bool = True
    # GLB export parameters
    conf_thresh_percentile: float = 40.0
    num_max_points: int = 1_000_000
    show_cameras: bool = True
    # Feat_vis export parameters
    feat_vis_fps: int = 15


class InferenceResponse(BaseModel):
    """Response model for inference API."""

    success: bool
    message: str
    task_id: Optional[str] = None
    export_dir: Optional[str] = None
    export_format: str = "mini_npz-glb"
    processing_time: Optional[float] = None


class TaskStatus(BaseModel):
    """Task status model."""

    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    message: str
    progress: Optional[float] = None  # 0.0 to 1.0
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    export_dir: Optional[str] = None
    request: Optional[InferenceRequest] = None  # Store the original request

    # Essential task parameters
    num_images: Optional[int] = None  # Number of input images
    export_format: Optional[str] = None  # Export format
    process_res_method: Optional[str] = None  # Processing resolution method
    video_path: Optional[str] = None  # Source video path


class ModelBackend:
    """Model backend service with persistent model loading."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.model_loaded = False
        self.load_time = None
        self.load_start_time = None  # Time when model loading started
        self.load_completed_time = None  # Time when model loading completed
        self.last_used = None

    def load_model(self):
        """Load model if not already loaded."""
        if self.model_loaded and self.model is not None:
            self.last_used = time.time()
            return self.model

        try:
            print(f"Loading model from {self.model_dir}...")
            self.load_start_time = time.time()
            start_time = time.time()

            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
            self.model.eval()

            self.model_loaded = True
            self.load_time = time.time() - start_time
            self.load_completed_time = time.time()
            self.last_used = time.time()

            print(f"Model loaded successfully in {self.load_time:.2f}s")
            return self.model

        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def get_model(self):
        """Get model, loading if necessary."""
        if not self.model_loaded:
            return self.load_model()
        self.last_used = time.time()
        return self.model

    def get_status(self) -> Dict[str, Any]:
        """Get backend status information."""
        # Calculate uptime from when model loading completed
        uptime = 0
        if self.model_loaded and self.load_completed_time:
            uptime = time.time() - self.load_completed_time

        return {
            "model_loaded": self.model_loaded,
            "model_dir": self.model_dir,
            "device": self.device,
            "load_time": self.load_time,
            "last_used": self.last_used,
            "uptime": uptime,
        }


# Global backend instance
_backend: Optional[ModelBackend] = None
_app: Optional[FastAPI] = None
_tasks: Dict[str, TaskStatus] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # Restrict to single-task execution
_running_task_id: Optional[str] = None  # Currently running task ID
_task_queue: List[str] = []  # Pending task queue

# Task cleanup configuration
MAX_TASK_HISTORY = 100  # Maximum number of tasks to keep in memory
CLEANUP_INTERVAL = 300  # Cleanup interval in seconds (5 minutes)


def _process_next_task():
    """Process the next task in the queue."""
    global _task_queue, _running_task_id

    if not _task_queue or _running_task_id is not None:
        return

    # Get next task from queue
    task_id = _task_queue.pop(0)

    # Get task request from tasks dict (we need to store the request)
    if task_id not in _tasks:
        return

    # Submit task to executor
    _executor.submit(_run_inference_task, task_id)


# get_gpu_memory_info imported from depth_anything_3.utils.memory


# cleanup_cuda_memory imported from depth_anything_3.utils.memory


# check_memory_availability imported from depth_anything_3.utils.memory


# estimate_memory_requirement imported from depth_anything_3.utils.memory


def _run_inference_task(task_id: str):
    """Run inference task in background thread with OOM protection."""
    global _tasks, _backend, _running_task_id, _task_queue

    model = None
    inference_started = False
    start_time = time.time()

    try:
        # Get task request
        if task_id not in _tasks or _tasks[task_id].request is None:
            print(f"[{task_id}] Task not found or request missing")
            return

        request = _tasks[task_id].request
        num_images = len(request.image_paths)

        # Set current running task
        _running_task_id = task_id

        # Update task status to running
        _tasks[task_id].status = "running"
        _tasks[task_id].started_at = start_time
        _tasks[task_id].message = f"[{task_id}] Starting inference on {num_images} frames..."
        print(f"[{task_id}] Starting inference on {num_images} frames")

        # Pre-inference cleanup to ensure maximum available memory
        print(f"[{task_id}] Pre-inference cleanup...")
        cleanup_cuda_memory()

        # Check memory availability
        estimated_memory = estimate_memory_requirement(num_images, request.process_res)
        mem_available, mem_msg = check_memory_availability(estimated_memory)
        print(f"[{task_id}] {mem_msg}")

        if not mem_available:
            # Try aggressive cleanup
            print(f"[{task_id}] Insufficient memory, attempting aggressive cleanup...")
            cleanup_cuda_memory()
            time.sleep(0.5)  # Give system time to reclaim memory

            # Check again
            mem_available, mem_msg = check_memory_availability(estimated_memory)
            if not mem_available:
                raise RuntimeError(
                    f"Insufficient GPU memory after cleanup. {mem_msg}\n"
                    f"Suggestions:\n"
                    f"  1. Reduce process_res (current: {request.process_res})\n"
                    f"  2. Process fewer images at once (current: {num_images})\n"
                    f"  3. Clear other GPU processes"
                )

        # Get model (with error handling)
        print(f"[{task_id}] Loading model...")
        _tasks[task_id].message = f"[{task_id}] Loading model..."
        _tasks[task_id].progress = 0.1

        try:
            model = _backend.get_model()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                cleanup_cuda_memory()
                raise RuntimeError(
                    f"OOM during model loading: {str(e)}\n"
                    f"Try reducing the batch size or resolution."
                )
            raise

        print(f"[{task_id}] Model loaded successfully")
        _tasks[task_id].progress = 0.2

        # Prepare inference parameters
        inference_kwargs = {
            "image": request.image_paths,
            "export_format": request.export_format,
            "process_res": request.process_res,
            "process_res_method": request.process_res_method,
            "export_feat_layers": request.export_feat_layers,
            "align_to_input_ext_scale": request.align_to_input_ext_scale,
            "conf_thresh_percentile": request.conf_thresh_percentile,
            "num_max_points": request.num_max_points,
            "show_cameras": request.show_cameras,
            "feat_vis_fps": request.feat_vis_fps,
        }

        if request.export_dir:
            inference_kwargs["export_dir"] = request.export_dir

        if request.extrinsics:
            inference_kwargs["extrinsics"] = np.array(request.extrinsics, dtype=np.float32)

        if request.intrinsics:
            inference_kwargs["intrinsics"] = np.array(request.intrinsics, dtype=np.float32)

        # Run inference with timing
        inference_start_time = time.time()
        print(f"[{task_id}] Running model inference...")
        _tasks[task_id].message = f"[{task_id}] Running model inference on {num_images} images..."
        _tasks[task_id].progress = 0.3

        inference_started = True

        try:
            model.inference(**inference_kwargs)
            inference_time = time.time() - inference_start_time
            avg_time_per_image = inference_time / num_images if num_images > 0 else 0

            print(
                f"[{task_id}] Inference completed in {inference_time:.2f}s "
                f"({avg_time_per_image:.2f}s per image)"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                cleanup_cuda_memory()
                raise RuntimeError(
                    f"OOM during inference: {str(e)}\n"
                    f"Settings: {num_images} images, resolution={request.process_res}\n"
                    f"Suggestions:\n"
                    f"  1. Reduce process_res to {int(request.process_res * 0.75)}\n"
                    f"  2. Process images in smaller batches\n"
                    f"  3. Use process_res_method='resize' instead of 'upper_bound_resize'"
                )
            raise

        _tasks[task_id].progress = 0.9

        # Post-inference cleanup
        print(f"[{task_id}] Post-inference cleanup...")
        cleanup_cuda_memory()

        # Calculate total processing time
        total_time = time.time() - start_time

        # Update task status to completed
        _tasks[task_id].status = "completed"
        _tasks[task_id].completed_at = time.time()
        _tasks[task_id].message = (
            f"[{task_id}] Completed in {total_time:.2f}s " f"({avg_time_per_image:.2f}s per image)"
        )
        _tasks[task_id].progress = 1.0
        _tasks[task_id].export_dir = request.export_dir

        # Clear running state
        _running_task_id = None

        # Process next task in queue
        _process_next_task()

        print(f"[{task_id}] Task completed successfully")
        print(
            f"[{task_id}] Total time: {total_time:.2f}s, "
            f"Inference time: {inference_time:.2f}s, "
            f"Avg per image: {avg_time_per_image:.2f}s"
        )

    except Exception as e:
        # Update task status to failed
        error_msg = str(e)
        total_time = time.time() - start_time

        print(f"[{task_id}] Task failed after {total_time:.2f}s: {error_msg}")

        # Always attempt cleanup on failure
        cleanup_cuda_memory()

        _tasks[task_id].status = "failed"
        _tasks[task_id].completed_at = time.time()
        _tasks[task_id].message = f"[{task_id}] Failed after {total_time:.2f}s: {error_msg}"

        # Clear running state
        _running_task_id = None

        # Process next task in queue
        _process_next_task()

    finally:
        # Final cleanup in finally block to ensure it always runs
        # This is critical for releasing resources even if unexpected errors occur
        try:
            if inference_started:
                print(f"[{task_id}] Final cleanup in finally block...")
                cleanup_cuda_memory()
        except Exception as e:
            print(f"[{task_id}] Warning: Finally block cleanup failed: {e}")

        # Schedule cleanup after task completion
        _schedule_task_cleanup()


def _cleanup_old_tasks():
    """Clean up old completed/failed tasks to prevent memory buildup."""
    global _tasks

    current_time = time.time()
    tasks_to_remove = []

    # Find tasks to remove - more aggressive cleanup
    for task_id, task in _tasks.items():
        # Remove completed/failed tasks older than 10 minutes (instead of 1 hour)
        if (
            task.status in ["completed", "failed"]
            and task.completed_at
            and current_time - task.completed_at > 600
        ):  # 10 minutes
            tasks_to_remove.append(task_id)

    # Remove old tasks
    for task_id in tasks_to_remove:
        del _tasks[task_id]
        print(f"[CLEANUP] Removed old task: {task_id}")

    # If still too many tasks, remove oldest completed/failed tasks
    if len(_tasks) > MAX_TASK_HISTORY:
        completed_tasks = [
            (task_id, task)
            for task_id, task in _tasks.items()
            if task.status in ["completed", "failed"]
        ]
        completed_tasks.sort(key=lambda x: x[1].completed_at or 0)

        excess_count = len(_tasks) - MAX_TASK_HISTORY
        for i in range(min(excess_count, len(completed_tasks))):
            task_id = completed_tasks[i][0]
            del _tasks[task_id]
            print(f"[CLEANUP] Removed excess task: {task_id}")

    # Count active tasks (only pending and running)
    active_count = sum(1 for task in _tasks.values() if task.status in ["pending", "running"])
    print(
        "[CLEANUP] Task cleanup completed. "
        f"Total tasks: {len(_tasks)}, Active tasks: {active_count}"
    )


def _schedule_task_cleanup():
    """Schedule task cleanup in background."""

    def cleanup_worker():
        try:
            time.sleep(2)  # Small delay to ensure task status is updated
            _cleanup_old_tasks()
        except Exception as e:
            print(f"[CLEANUP] Cleanup worker failed: {e}")

    # Run cleanup in background thread
    _executor.submit(cleanup_worker)


# ============================================================================
# Gallery utilities (extracted from gallery.py)
# ============================================================================

GALLERY_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _load_gallery_html() -> str:
    """
    Load and modify gallery HTML to work under /gallery/ subdirectory.
    Replaces API paths from root to /gallery/ prefix.
    """
    from ..services.gallery import HTML_PAGE

    # Replace API paths to be under /gallery/ subdirectory
    html = (
        HTML_PAGE.replace("fetch('/manifest.json'", "fetch('/gallery/manifest.json'")
        .replace("fetch('/manifest/'+", "fetch('/gallery/manifest/'+")
        .replace(
            "if(location.pathname!=\"/\")history.replaceState(null,'','/'+location.search)",
            "if(!location.pathname.startsWith(\"/gallery\"))history.replaceState(null,'','/gallery/'+location.search)",
        )
    )

    return html


def _gallery_url_join(*parts: str) -> str:
    """Join URL parts safely."""
    norm = posixpath.join(*[p.replace("\\", "/") for p in parts])
    segs = [s for s in norm.split("/") if s not in ("", ".")]
    return "/".join(quote(s) for s in segs)


def _is_plain_name(name: str) -> bool:
    """Check if name is safe for use in paths."""
    return all(c not in name for c in ("/", "\\")) and name not in (".", "..")


def build_group_list(root_dir: str) -> dict:
    """Build list of groups from gallery directory."""
    groups = []
    try:
        for gname in sorted(os.listdir(root_dir)):
            gpath = os.path.join(root_dir, gname)
            if not os.path.isdir(gpath):
                continue
            has_scene = False
            try:
                for sname in os.listdir(gpath):
                    spath = os.path.join(gpath, sname)
                    if not os.path.isdir(spath):
                        continue
                    if os.path.exists(os.path.join(spath, "scene.glb")) and os.path.exists(
                        os.path.join(spath, "scene.jpg")
                    ):
                        has_scene = True
                        break
            except Exception:
                pass
            if has_scene:
                groups.append({"id": gname, "title": gname})
    except Exception as e:
        print(f"[warn] build_group_list failed: {e}")
    return {"groups": groups}


def build_group_manifest(root_dir: str, group: str) -> dict:
    """Build manifest for a specific group."""
    items = []
    gpath = os.path.join(root_dir, group)
    try:
        if not os.path.isdir(gpath):
            return {"group": group, "items": []}
        for sname in sorted(os.listdir(gpath)):
            spath = os.path.join(gpath, sname)
            if not os.path.isdir(spath):
                continue
            glb_fs = os.path.join(spath, "scene.glb")
            jpg_fs = os.path.join(spath, "scene.jpg")
            if not (os.path.exists(glb_fs) and os.path.exists(jpg_fs)):
                continue
            depth_images = []
            dpath = os.path.join(spath, "depth_vis")
            if os.path.isdir(dpath):
                files = [
                    f
                    for f in os.listdir(dpath)
                    if os.path.splitext(f)[1].lower() in GALLERY_IMAGE_EXTS
                ]
                for fn in sorted(files):
                    depth_images.append(
                        "/gallery/" + _gallery_url_join(group, sname, "depth_vis", fn)
                    )
            items.append(
                {
                    "id": sname,
                    "title": sname,
                    "model": "/gallery/" + _gallery_url_join(group, sname, "scene.glb"),
                    "thumbnail": "/gallery/" + _gallery_url_join(group, sname, "scene.jpg"),
                    "depth_images": depth_images,
                }
            )
    except Exception as e:
        print(f"[warn] build_group_manifest failed for {group}: {e}")
    return {"group": group, "items": items}


def create_app(model_dir: str, device: str = "cuda", gallery_dir: Optional[str] = None) -> FastAPI:
    """Create FastAPI application with model backend."""
    global _backend, _app

    _backend = ModelBackend(model_dir, device)
    _app = FastAPI(
        title="Depth Anything 3 Backend",
        description="Model inference service for Depth Anything 3",
        version="1.0.0",
    )

    # Store gallery directory globally for use in routes
    _gallery_dir = gallery_dir

    @_app.get("/", response_class=HTMLResponse)
    async def root():
        """Home page with navigation to dashboard and gallery."""
        html_content = (
            """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 Backend</title>
    <style>
        :root {
            --tech-blue: #00d4ff;
            --tech-cyan: #00ffcc;
            --tech-purple: #7877c6;
        }

        * {
            box-sizing: border-box;
        }

        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
                color: #e8eaed;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow-x: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background:
                    radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
                animation: techPulse 8s ease-in-out infinite;
                z-index: -1;
            }

            .container {
                max-width: 800px;
                padding: 40px;
                text-align: center;
                z-index: 1;
            }

            h1 {
                font-size: 3em;
                margin: 0 0 20px 0;
                background: linear-gradient(45deg, var(--tech-blue), var(--tech-cyan), var(--tech-purple));
                background-size: 400% 400%;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: techGradient 3s ease infinite;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            }

            .subtitle {
                font-size: 1.2em;
                opacity: 0.8;
                margin-bottom: 50px;
                color: #a0a0a0;
            }

            .nav-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 24px;
                margin-top: 40px;
            }

            .nav-card {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 16px;
                padding: 30px;
                text-decoration: none;
                color: inherit;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }

            .nav-card:hover {
                transform: translateY(-4px);
                border-color: var(--tech-blue);
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
            }

            .nav-card h2 {
                margin: 0 0 15px 0;
                font-size: 1.8em;
                color: var(--tech-blue);
            }

            .nav-card p {
                margin: 0;
                opacity: 0.8;
                line-height: 1.6;
            }
        }

        /* Light mode styles */
        @media (prefers-color-scheme: light) {
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
                color: #1e293b;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow-x: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background:
                    radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(0, 102, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(0, 255, 204, 0.08) 0%, transparent 50%);
                animation: techPulse 8s ease-in-out infinite;
                z-index: -1;
            }

            .container {
                max-width: 800px;
                padding: 40px;
                text-align: center;
                z-index: 1;
            }

            h1 {
                font-size: 3em;
                margin: 0 0 20px 0;
                background: linear-gradient(45deg, #0066ff, #00d4ff, #00ffcc);
                background-size: 400% 400%;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: techGradient 3s ease infinite;
                text-shadow: 0 0 20px rgba(0, 102, 255, 0.3);
            }

            .subtitle {
                font-size: 1.2em;
                opacity: 0.8;
                margin-bottom: 50px;
                color: #64748b;
            }

            .nav-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 24px;
                margin-top: 40px;
            }

            .nav-card {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 16px;
                padding: 30px;
                text-decoration: none;
                color: inherit;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .nav-card:hover {
                transform: translateY(-4px);
                border-color: #0066ff;
                box-shadow: 0 8px 25px rgba(0, 102, 255, 0.2);
            }

            .nav-card h2 {
                margin: 0 0 15px 0;
                font-size: 1.8em;
                color: #0066ff;
            }

            .nav-card p {
                margin: 0;
                opacity: 0.8;
                line-height: 1.6;
            }
        }

        @keyframes techPulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 0.8; }
        }

        @keyframes techGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .footer {
            margin-top: 50px;
            opacity: 0.6;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Depth Anything 3</h1>
        <p class="subtitle">Model Backend Service</p>
        <div class="nav-grid">
            <a href="/dashboard" class="nav-card">
                <h2>📊 Dashboard</h2>
                <p>Monitor backend status, model information, and inference tasks in real-time.</p>
            </a>
            """
            + (
                '<a href="/gallery/" class="nav-card">'
                "<h2>🎨 Gallery</h2>"
                "<p>Browse 3D reconstructions and depth visualizations from processed scenes.</p>"
                "</a>"
                if _gallery_dir and os.path.exists(_gallery_dir)
                else ""
            )
            + """
        </div>
        <div class="footer">
            <p>Depth Anything 3 Backend API</p>
        </div>
    </div>
</body>
</html>
        """
        )
        return HTMLResponse(html_content)

    @_app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """HTML dashboard for monitoring backend status and tasks."""
        if _backend is None:
            return HTMLResponse("<h1>Backend not initialized</h1>", status_code=500)

        # Get backend status
        status = _backend.get_status()

        # Safely format status values
        if status["load_time"] is not None:
            load_time_str = f"{status['load_time']:.2f}s"
        else:
            load_time_str = "Not loaded"

        if status["uptime"] is not None:
            uptime_str = f"{status['uptime']:.2f}s"
        else:
            uptime_str = "Not running"

        # Get tasks information
        active_tasks = [task for task in _tasks.values() if task.status in ["pending", "running"]]
        completed_tasks = [
            task for task in _tasks.values() if task.status in ["completed", "failed"]
        ]

        # Generate task HTML
        active_tasks_html = ""
        if active_tasks:
            for task in active_tasks:
                task_details = f"""
                <div class="task-item running">
                    <div class="task-header">
                        <span class="task-id">{task.task_id}</span>
                        <span class="task-status status-{task.status}">{task.status}</span>
                    </div>
                    <div class="task-message">{task.message}</div>
                    <div class="task-params">
                        <small>
                            Images: {task.num_images or 'N/A'} |
                            Format: {task.export_format or 'N/A'} |
                            Method: {task.process_res_method or 'N/A'} |
                            Export Dir: {task.export_dir or 'N/A'}
                        </small>
                        {f'<br><small>Video: {task.video_path}</small>' if task.video_path else ''}
                    </div>
                </div>
                """
                active_tasks_html += task_details
        else:
            active_tasks_html = "<p>No active tasks</p>"

        completed_tasks_html = ""
        if completed_tasks:
            for task in completed_tasks[-10:]:
                task_details = f"""
                <div class="task-item completed">
                    <div class="task-header">
                        <span class="task-id">{task.task_id}</span>
                        <span class="task-status status-{task.status}">{task.status}</span>
                    </div>
                    <div class="task-message">{task.message}</div>
                    <div class="task-params">
                        <small>
                            Images: {task.num_images or 'N/A'} |
                            Format: {task.export_format or 'N/A'} |
                            Method: {task.process_res_method or 'N/A'} |
                            Export Dir: {task.export_dir or 'N/A'}
                        </small>
                        {f'<br><small>Video: {task.video_path}</small>' if task.video_path else ''}
                    </div>
                </div>
                """
                completed_tasks_html += task_details
        else:
            completed_tasks_html = "<p>No completed tasks</p>"

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 Backend Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .status-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .status-card h3 {{
            margin-top: 0;
            color: #333;
        }}
        .status-item {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .status-item:last-child {{
            border-bottom: none;
        }}
        .status-value {{
            font-weight: bold;
            color: #666;
        }}
        .status-online {{
            color: #28a745;
        }}
        .status-offline {{
            color: #dc3545;
        }}
        .tasks-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .task-item {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .task-item.completed {{
            border-left-color: #28a745;
        }}
        .task-item.failed {{
            border-left-color: #dc3545;
        }}
        .task-item.running {{
            border-left-color: #ffc107;
        }}
        .task-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .task-id {{
            font-family: monospace;
            font-size: 12px;
            color: #666;
        }}
        .task-status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-pending {{
            background: #fff3cd;
            color: #856404;
        }}
        .status-running {{
            background: #d4edda;
            color: #155724;
        }}
        .status-completed {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .refresh-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .refresh-btn:hover {{
            background: #0056b3;
        }}
        .auto-refresh {{
            margin-left: 10px;
        }}
        .timestamp {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        .task-message {{
            font-size: 14px;
            color: #333;
            margin-bottom: 8px;
        }}
        .task-params {{
            font-size: 12px;
            color: #666;
            background: #f8f9fa;
            padding: 6px 8px;
            border-radius: 4px;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Depth Anything 3 Backend Dashboard</h1>
            <p>Real-time monitoring of model status and inference tasks</p>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>Model Status</h3>
                <div class="status-item">
                    <span>Status:</span>
                    <span class="status-value {'status-online' if status['model_loaded'] else 'status-offline'}">
                        {'Online' if status['model_loaded'] else 'Offline'}
                    </span>
                </div>
                <div class="status-item">
                    <span>Model Directory:</span>
                    <span class="status-value">{status['model_dir']}</span>
                </div>
                <div class="status-item">
                    <span>Device:</span>
                    <span class="status-value">{status['device']}</span>
                </div>
                <div class="status-item">
                    <span>Load Time:</span>
                    <span class="status-value">{load_time_str}</span>
                </div>
                <div class="status-item">
                    <span>Uptime:</span>
                    <span class="status-value">{uptime_str}</span>
                </div>
            </div>

            <div class="status-card">
                <h3>Task Summary</h3>
                <div class="status-item">
                    <span>Active Tasks:</span>
                    <span class="status-value">{len(active_tasks)}</span>
                </div>
                <div class="status-item">
                    <span>Completed Tasks:</span>
                    <span class="status-value">{len(completed_tasks)}</span>
                </div>
                <div class="status-item">
                    <span>Total Tasks:</span>
                    <span class="status-value">{len(_tasks)}</span>
                </div>
            </div>
        </div>

        <div class="tasks-section">
            <h3>Active Tasks</h3>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
            <label class="auto-refresh">
                <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> Auto-refresh (5s)
            </label>
            <div class="timestamp">Last updated: <span id="lastUpdate">{time.strftime('%Y-%m-%d %H:%M:%S')}</span></div>

            {active_tasks_html}
        </div>

        <div class="tasks-section">
            <h3>Recent Completed Tasks</h3>
            {completed_tasks_html}
        </div>
    </div>

    <script>
        let autoRefreshInterval;

        function toggleAutoRefresh() {{
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {{
                autoRefreshInterval = setInterval(() => {{
                    location.reload();
                }}, 5000);
            }} else {{
                clearInterval(autoRefreshInterval);
            }}
        }}

        // Update timestamp every second
        setInterval(() => {{
            const now = new Date();
            document.getElementById('lastUpdate').textContent = now.toLocaleString();
        }}, 1000);
    </script>
</body>
</html>
        """

        return HTMLResponse(html_content)

    @_app.get("/status")
    async def get_status():
        """Get backend status with GPU memory information."""
        if _backend is None:
            raise HTTPException(status_code=500, detail="Backend not initialized")

        status = _backend.get_status()

        # Add GPU memory information
        gpu_memory = get_gpu_memory_info()
        if gpu_memory:
            status["gpu_memory"] = {
                "total_gb": round(gpu_memory["total_gb"], 2),
                "allocated_gb": round(gpu_memory["allocated_gb"], 2),
                "reserved_gb": round(gpu_memory["reserved_gb"], 2),
                "free_gb": round(gpu_memory["free_gb"], 2),
                "utilization_percent": round(gpu_memory["utilization"], 1),
            }
        else:
            status["gpu_memory"] = None

        return status

    @_app.post("/inference", response_model=InferenceResponse)
    async def run_inference(request: InferenceRequest):
        """Submit inference task and return task ID."""
        global _running_task_id

        if _backend is None:
            raise HTTPException(status_code=500, detail="Backend not initialized")

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Create task status
        if _running_task_id is not None:
            status_msg = f"[{task_id}] Task queued (waiting for {_running_task_id} to complete)"
        else:
            status_msg = f"[{task_id}] Task submitted"

        _tasks[task_id] = TaskStatus(
            task_id=task_id,
            status="pending",
            message=status_msg,
            created_at=time.time(),
            export_dir=request.export_dir,
            request=request,
            # Record essential parameters
            num_images=len(request.image_paths),
            export_format=request.export_format,
            process_res_method=request.process_res_method,
            video_path=(
                request.image_paths[0] if request.image_paths else None
            ),  # Use first image path as video reference
        )

        # Add task to queue
        _task_queue.append(task_id)

        # If no task is running, start processing the queue
        if _running_task_id is None:
            _process_next_task()

        return InferenceResponse(
            success=True,
            message="Task submitted successfully",
            task_id=task_id,
            export_dir=request.export_dir,
            export_format=request.export_format,
        )

    @_app.get("/task/{task_id}", response_model=TaskStatus)
    async def get_task_status(task_id: str):
        """Get task status by task ID."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        return _tasks[task_id]

    @_app.get("/gpu-memory")
    async def get_gpu_memory():
        """Get detailed GPU memory information."""
        gpu_memory = get_gpu_memory_info()
        if gpu_memory is None:
            return {
                "available": False,
                "message": "CUDA not available or memory info cannot be retrieved",
            }

        return {
            "available": True,
            "total_gb": round(gpu_memory["total_gb"], 2),
            "allocated_gb": round(gpu_memory["allocated_gb"], 2),
            "reserved_gb": round(gpu_memory["reserved_gb"], 2),
            "free_gb": round(gpu_memory["free_gb"], 2),
            "utilization_percent": round(gpu_memory["utilization"], 1),
            "status": (
                "healthy"
                if gpu_memory["utilization"] < 80
                else "warning" if gpu_memory["utilization"] < 95 else "critical"
            ),
        }

    @_app.get("/tasks")
    async def list_tasks():
        """List all tasks."""
        # Separate active and completed tasks
        active_tasks = [task for task in _tasks.values() if task.status in ["pending", "running"]]
        completed_tasks = [
            task for task in _tasks.values() if task.status in ["completed", "failed"]
        ]

        return {
            "tasks": list(_tasks.values()),
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "active_count": len(active_tasks),
            "total_count": len(_tasks),
        }

    @_app.post("/cleanup")
    async def manual_cleanup():
        """Manually trigger task cleanup."""
        try:
            _cleanup_old_tasks()
            return {"message": "Cleanup completed", "active_tasks": len(_tasks)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

    @_app.delete("/task/{task_id}")
    async def delete_task(task_id: str):
        """Delete a specific task."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        # Only allow deletion of completed/failed tasks
        if _tasks[task_id].status not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Cannot delete running or pending tasks")

        del _tasks[task_id]
        return {"message": f"Task {task_id} deleted successfully"}

    @_app.post("/reload")
    async def reload_model():
        """Reload the model."""
        if _backend is None:
            raise HTTPException(status_code=500, detail="Backend not initialized")

        try:
            _backend.model = None
            _backend.model_loaded = False
            _backend.load_model()
            return {"message": "Model reloaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

    # ============================================================================
    # Gallery routes
    # ============================================================================

    if _gallery_dir and os.path.exists(_gallery_dir):
        # Load gallery HTML page (with modified paths for /gallery/ subdirectory)
        _gallery_html = _load_gallery_html()

        @_app.get("/gallery/", response_class=HTMLResponse)
        @_app.get("/gallery", response_class=HTMLResponse)
        async def gallery_home():
            """Gallery home page."""
            return HTMLResponse(_gallery_html)

        @_app.get("/gallery/manifest.json")
        async def gallery_manifest():
            """Get gallery group list."""
            try:
                return build_group_list(_gallery_dir)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to build group list: {str(e)}"
                )

        @_app.get("/gallery/manifest/{group}.json")
        async def gallery_group_manifest(group: str):
            """Get manifest for a specific group."""
            if not _is_plain_name(group):
                raise HTTPException(status_code=400, detail="Invalid group name")
            try:
                return build_group_manifest(_gallery_dir, group)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to build group manifest: {str(e)}"
                )

        @_app.get("/gallery/{path:path}")
        async def gallery_files(path: str):
            """Serve gallery static files (GLB, JPG, etc.)."""
            # Security check: prevent directory traversal
            path_parts = path.split("/")
            if any(not _is_plain_name(part) for part in path_parts if part):
                raise HTTPException(status_code=400, detail="Invalid path")

            file_path = os.path.join(_gallery_dir, *path_parts)

            # Ensure the file is within gallery directory
            real_file_path = os.path.realpath(file_path)
            real_gallery_dir = os.path.realpath(_gallery_dir)
            if not real_file_path.startswith(real_gallery_dir):
                raise HTTPException(status_code=403, detail="Access denied")

            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(file_path)

    return _app


def start_server(
    model_dir: str,
    device: str = "cuda",
    host: str = "127.0.0.1",
    port: int = 8000,
    gallery_dir: Optional[str] = None,
):
    """Start the backend server."""
    app = create_app(model_dir, device, gallery_dir)

    print("Starting Depth Anything 3 Backend...")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print(f"Server: http://{host}:{port}")
    print(f"Dashboard: http://{host}:{port}/dashboard")
    print(f"API Status: http://{host}:{port}/status")

    if gallery_dir and os.path.exists(gallery_dir):
        print(f"Gallery: http://{host}:{port}/gallery/")

    print("=" * 60)
    print("Backend is running! You can now:")
    print(f"  • Open home page: http://{host}:{port}")
    print(f"  • Open dashboard: http://{host}:{port}/dashboard")
    print(f"  • Check API status: http://{host}:{port}/status")

    if gallery_dir and os.path.exists(gallery_dir):
        print(f"  • Browse gallery: http://{host}:{port}/gallery/")

    print("  • Submit inference tasks via API")
    print("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth Anything 3 Backend Server")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--gallery-dir", help="Gallery directory path (optional)")

    args = parser.parse_args()
    start_server(args.model_dir, args.device, args.host, args.port, args.gallery_dir)
