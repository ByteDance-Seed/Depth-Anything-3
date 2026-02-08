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

import html
import os
import posixpath
import time
import uuid
import re
import threading

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import numpy as np

import uvicorn
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

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
    task_kind: str = "manual"  # "manual" or "stream"
    session_id: Optional[str] = None  # Streaming session ID
    chunk_id: Optional[int] = None  # Streaming chunk ID
    frame_start: Optional[int] = None  # Inclusive frame start index for stream chunk
    frame_end: Optional[int] = None  # Exclusive frame end index for stream chunk


class SessionCreateRequest(BaseModel):
    """Request model for streaming session creation."""

    robot_id: str = "robot-unknown"
    camera: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    """Response model for streaming session creation."""

    session_id: str
    upload_url: str
    events_url: str
    map_url: str
    created_at: float


class FrameUploadResponse(BaseModel):
    """Response model for frame upload."""

    success: bool
    session_id: str
    frame_index: int
    frame_path: str
    total_frames: int
    queued_tasks: int = 0
    pending_frames: int = 0


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
_stream_sessions: Dict[str, Dict[str, Any]] = {}  # Streaming session metadata
_stream_task_meta: Dict[str, Dict[str, Any]] = {}
_stream_lock = threading.RLock()
_session_root: Optional[str] = None

# Task cleanup configuration
MAX_TASK_HISTORY = 100  # Maximum number of tasks to keep in memory
CLEANUP_INTERVAL = 300  # Cleanup interval in seconds (5 minutes)
MAX_SESSION_EVENTS = 2000
STREAM_DEFAULT_CHUNK_SIZE = 8
STREAM_DEFAULT_MAX_INFLIGHT = 1


def _safe_name(value: Optional[str], fallback: str) -> str:
    """Normalize a user-provided token to a filesystem-safe token."""
    if not value:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return safe or fallback


def _safe_positive_int(value: Any, default: int, minimum: int = 1, maximum: int = 10_000) -> int:
    """Parse positive integer with fallback and bounds."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _append_session_event(session: Dict[str, Any], event_type: str, payload: Dict[str, Any]) -> None:
    """Append one session event and keep a bounded event history."""
    session.setdefault("events", []).append(
        {"event_type": event_type, "timestamp": time.time(), "payload": payload}
    )
    if len(session["events"]) > MAX_SESSION_EVENTS:
        session["events"] = session["events"][-MAX_SESSION_EVENTS:]


def _pending_stream_frames(session: Dict[str, Any]) -> int:
    """Return number of uploaded-but-not-yet-scheduled frames."""
    return max(0, int(session["frame_count"]) - int(session["next_frame_index"]))


def _enqueue_inference_task(
    request: InferenceRequest,
    task_kind: str = "manual",
    session_id: Optional[str] = None,
    chunk_id: Optional[int] = None,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> str:
    """Create and enqueue an inference task."""
    global _running_task_id

    task_id = str(uuid.uuid4())
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
        num_images=len(request.image_paths),
        export_format=request.export_format,
        process_res_method=request.process_res_method,
        video_path=(request.image_paths[0] if request.image_paths else None),
        task_kind=task_kind,
        session_id=session_id,
        chunk_id=chunk_id,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    if task_kind == "stream" and session_id is not None and chunk_id is not None:
        _stream_task_meta[task_id] = {
            "session_id": session_id,
            "chunk_id": chunk_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "export_dir": request.export_dir,
        }

    _task_queue.append(task_id)
    if _running_task_id is None:
        _process_next_task()
    return task_id


def _reserve_stream_chunk(session_id: str, force: bool = False) -> Optional[Dict[str, Any]]:
    """
    Reserve one chunk of frames from a session for inference scheduling.

    Returns chunk metadata if a chunk should be scheduled; otherwise None.
    """
    with _stream_lock:
        session = _stream_sessions.get(session_id)
        if session is None:
            return None

        inflight = len(session["inflight_task_ids"])
        if inflight >= int(session["max_inflight_chunks"]):
            return None

        pending = _pending_stream_frames(session)
        if pending <= 0:
            return None

        chunk_size = int(session["chunk_size"])
        if pending < chunk_size and not force:
            return None

        frame_start = int(session["next_frame_index"])
        take = chunk_size if pending >= chunk_size else pending
        frame_end = frame_start + take
        frame_records = session["frames"][frame_start:frame_end]
        if not frame_records:
            return None

        chunk_id = int(session["next_chunk_id"])
        session["next_chunk_id"] = chunk_id + 1
        session["next_frame_index"] = frame_end

        return {
            "session_id": session_id,
            "chunk_id": chunk_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_paths": [item["frame_path"] for item in frame_records],
            "config": dict(session.get("config") or {}),
            "session_dir": session["session_dir"],
        }


def _schedule_stream_inference(session_id: str, force: bool = False) -> int:
    """
    Schedule as many stream chunks as possible for one session.

    Returns number of newly queued inference tasks.
    """
    queued = 0
    while True:
        chunk = _reserve_stream_chunk(session_id, force=force)
        if chunk is None:
            break

        cfg = chunk["config"]
        export_dir = os.path.join(
            chunk["session_dir"], "inference", f"chunk_{chunk['chunk_id']:06d}"
        )
        request = InferenceRequest(
            image_paths=chunk["frame_paths"],
            export_dir=export_dir,
            export_format=str(cfg.get("export_format", "mini_npz")),
            process_res=_safe_positive_int(cfg.get("process_res"), 504, minimum=64, maximum=2048),
            process_res_method=str(cfg.get("process_res_method", "upper_bound_resize")),
            export_feat_layers=(
                cfg.get("export_feat_layers")
                if isinstance(cfg.get("export_feat_layers"), list)
                else []
            ),
            align_to_input_ext_scale=bool(cfg.get("align_to_input_ext_scale", True)),
            conf_thresh_percentile=float(cfg.get("conf_thresh_percentile", 40.0)),
            num_max_points=_safe_positive_int(
                cfg.get("num_max_points"), 1_000_000, minimum=1_000, maximum=20_000_000
            ),
            show_cameras=bool(cfg.get("show_cameras", True)),
            feat_vis_fps=_safe_positive_int(cfg.get("feat_vis_fps"), 15, minimum=1, maximum=120),
        )

        task_id = _enqueue_inference_task(
            request=request,
            task_kind="stream",
            session_id=chunk["session_id"],
            chunk_id=chunk["chunk_id"],
            frame_start=chunk["frame_start"],
            frame_end=chunk["frame_end"],
        )

        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                continue
            session["inflight_task_ids"].append(task_id)
            session["worker_state"] = "running"
            _append_session_event(
                session,
                "inference_chunk_queued",
                {
                    "task_id": task_id,
                    "chunk_id": chunk["chunk_id"],
                    "frame_start": chunk["frame_start"],
                    "frame_end": chunk["frame_end"],
                    "num_frames": len(chunk["frame_paths"]),
                    "export_dir": export_dir,
                },
            )
        queued += 1

    return queued


def _finalize_stream_task(task_id: str, succeeded: bool, message: str) -> None:
    """Update streaming session state when a stream task completes."""
    meta = _stream_task_meta.pop(task_id, None)
    if meta is None:
        return

    session_id = meta["session_id"]
    with _stream_lock:
        session = _stream_sessions.get(session_id)
        if session is None:
            return

        session["inflight_task_ids"] = [tid for tid in session["inflight_task_ids"] if tid != task_id]
        session["last_inference_at"] = time.time()

        if succeeded:
            session["completed_chunks"] += 1
            session["latest_chunk"] = {
                "task_id": task_id,
                "chunk_id": meta["chunk_id"],
                "frame_start": meta["frame_start"],
                "frame_end": meta["frame_end"],
                "export_dir": meta.get("export_dir"),
            }
            session["map_pointer"] = meta.get("export_dir")
            _append_session_event(
                session,
                "inference_chunk_completed",
                {
                    "task_id": task_id,
                    "chunk_id": meta["chunk_id"],
                    "frame_start": meta["frame_start"],
                    "frame_end": meta["frame_end"],
                    "export_dir": meta.get("export_dir"),
                    "message": message,
                },
            )
            _append_session_event(
                session,
                "map_chunk",
                {
                    "chunk_id": meta["chunk_id"],
                    "map_pointer": session["map_pointer"],
                    "frame_count": session["frame_count"],
                },
            )
        else:
            session["failed_chunks"] += 1
            _append_session_event(
                session,
                "inference_chunk_failed",
                {
                    "task_id": task_id,
                    "chunk_id": meta["chunk_id"],
                    "frame_start": meta["frame_start"],
                    "frame_end": meta["frame_end"],
                    "message": message,
                },
            )

        session["worker_state"] = "running" if session["inflight_task_ids"] else "idle"
        force_next = bool(session.get("flush_requested", False))
        if _pending_stream_frames(session) <= 0:
            session["flush_requested"] = False
            force_next = False

    # Try to keep the pipeline flowing after completion/failure.
    _schedule_stream_inference(session_id, force=force_next)


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
        _finalize_stream_task(task_id, succeeded=True, message=_tasks[task_id].message)

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
        _finalize_stream_task(task_id, succeeded=False, message=_tasks[task_id].message)

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
    global _backend, _app, _stream_sessions, _stream_task_meta, _session_root

    _backend = ModelBackend(model_dir, device)
    _app = FastAPI(
        title="Depth Anything 3 Backend",
        description="Model inference service for Depth Anything 3",
        version="1.0.0",
    )

    # Store gallery directory globally for use in routes
    _gallery_dir = gallery_dir
    _session_root = os.path.join(os.getcwd(), "workspace", "stream_sessions")
    os.makedirs(_session_root, exist_ok=True)
    with _stream_lock:
        _stream_sessions = {}
        _stream_task_meta = {}

    @_app.get("/", response_class=HTMLResponse)
    async def root():
        """Home page with navigation to dashboard and gallery."""
        gallery_card = ""
        if _gallery_dir and os.path.exists(_gallery_dir):
            gallery_card = """
            <a href="/gallery/" class="quick-link">
                <h2>Scene Gallery</h2>
                <p>Inspect generated GLB scenes, thumbnails, and depth visualization assets.</p>
            </a>
            """

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 Backend</title>
    <style>
        :root {{
            --ink: #132238;
            --ink-soft: #425a76;
            --paper: #f7f5f0;
            --panel: rgba(255, 255, 255, 0.78);
            --line: rgba(19, 34, 56, 0.16);
            --accent-a: #d45500;
            --accent-b: #00827f;
            --accent-c: #1b5dbf;
            --focus: #0a7a75;
            --shadow: 0 18px 40px rgba(18, 38, 62, 0.18);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 18%, rgba(212, 85, 0, 0.18), transparent 42%),
                radial-gradient(circle at 82% 0%, rgba(0, 130, 127, 0.18), transparent 36%),
                linear-gradient(155deg, #fdfcf9 0%, #f6f3eb 45%, #e9efe9 100%);
            padding: 28px 18px;
        }}

        .shell {{
            max-width: 980px;
            margin: 0 auto;
            padding: 30px;
            border-radius: 24px;
            border: 1px solid var(--line);
            background: var(--panel);
            backdrop-filter: blur(10px);
            box-shadow: var(--shadow);
            animation: rise 600ms ease-out both;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            border: 1px solid rgba(27, 93, 191, 0.24);
            color: var(--accent-c);
            background: rgba(27, 93, 191, 0.08);
            padding: 6px 12px;
            font-size: 12px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
        }}

        h1 {{
            margin: 18px 0 10px;
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            letter-spacing: -0.02em;
        }}

        .lede {{
            margin: 0;
            color: var(--ink-soft);
            font-size: clamp(1rem, 1.8vw, 1.2rem);
            max-width: 62ch;
        }}

        .quick-grid {{
            margin-top: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 14px;
        }}

        .quick-link {{
            text-decoration: none;
            color: inherit;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.78);
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
            animation: rise 700ms ease-out both;
        }}

        .quick-link:hover {{
            transform: translateY(-2px);
            border-color: rgba(212, 85, 0, 0.42);
            box-shadow: 0 10px 20px rgba(18, 38, 62, 0.14);
        }}

        .quick-link:focus-visible {{
            outline: 3px solid var(--focus);
            outline-offset: 2px;
        }}

        .quick-link h2 {{
            margin: 0 0 8px;
            font-size: 1.15rem;
            color: var(--accent-a);
            letter-spacing: -0.01em;
        }}

        .quick-link p {{
            margin: 0;
            color: var(--ink-soft);
            line-height: 1.45;
            font-size: 0.96rem;
        }}

        .footer {{
            margin-top: 28px;
            color: #5a6e87;
            font-size: 0.88rem;
        }}

        @keyframes rise {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 720px) {{
            body {{
                padding: 14px;
            }}

            .shell {{
                padding: 22px 18px;
                border-radius: 20px;
            }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <span class="badge">Control Plane</span>
        <h1>Depth Anything 3 Backend</h1>
        <p class="lede">
            Use the dashboard to monitor model health, queue activity, and active streaming sessions from one place.
        </p>

        <section class="quick-grid">
            <a href="/dashboard" class="quick-link">
                <h2>Operations Dashboard</h2>
                <p>Track model status, active jobs, recent outputs, and streaming sessions.</p>
            </a>
            <a href="/status" class="quick-link">
                <h2>API Status JSON</h2>
                <p>Inspect machine-readable status payloads for uptime and memory diagnostics.</p>
            </a>
            {gallery_card}
        </section>

        <p class="footer">Depth Anything 3 Backend Service</p>
    </main>
</body>
</html>
        """

        return HTMLResponse(html_content)

    @_app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """HTML dashboard for monitoring backend status and tasks."""
        if _backend is None:
            return HTMLResponse("<h1>Backend not initialized</h1>", status_code=500)

        status = _backend.get_status()
        active_tasks = [task for task in _tasks.values() if task.status in ["pending", "running"]]
        completed_tasks = [task for task in _tasks.values() if task.status in ["completed", "failed"]]
        with _stream_lock:
            sessions = sorted(
                [
                    {
                        "session_id": s.get("session_id"),
                        "robot_id": s.get("robot_id"),
                        "created_at": s.get("created_at"),
                        "frame_count": s.get("frame_count", 0),
                        "latest_frame_path": s.get("latest_frame_path"),
                        "event_count": len(s.get("events", [])) if isinstance(s.get("events"), list) else 0,
                    }
                    for s in _stream_sessions.values()
                ],
                key=lambda s: float(s.get("created_at", 0.0)),
                reverse=True,
            )

        def _fmt_seconds(value: Any) -> str:
            if value is None:
                return "-"
            try:
                return f"{float(value):.2f}s"
            except Exception:
                return "-"

        def _fmt_ts(value: Any) -> str:
            if value is None:
                return "-"
            try:
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))
            except Exception:
                return "-"

        def _render_task_card(task: TaskStatus, tone: str) -> str:
            safe_task_id = html.escape(task.task_id)
            safe_message = html.escape(task.message or "-")
            safe_video = html.escape(task.video_path) if task.video_path else "-"
            safe_export_dir = html.escape(task.export_dir or "-")
            status_name = task.status if task.status in {"pending", "running", "completed", "failed"} else "pending"

            progress_html = ""
            if task.progress is not None:
                progress_pct = max(0, min(100, int(float(task.progress) * 100)))
                progress_html = f"""
                <div class="progress-track" aria-label="task progress">
                    <div class="progress-fill" style="width: {progress_pct}%"></div>
                </div>
                """

            return f"""
            <article class="task-card task-{tone}">
                <header>
                    <code>{safe_task_id}</code>
                    <span class="chip chip-{status_name}">{status_name}</span>
                </header>
                <p class="task-message">{safe_message}</p>
                {progress_html}
                <div class="meta-grid">
                    <span>Images: <strong>{task.num_images or "-"}</strong></span>
                    <span>Format: <strong>{html.escape(task.export_format or "-")}</strong></span>
                    <span>Method: <strong>{html.escape(task.process_res_method or "-")}</strong></span>
                    <span>Video: <strong>{safe_video}</strong></span>
                    <span>Export: <strong>{safe_export_dir}</strong></span>
                    <span>Created: <strong>{_fmt_ts(task.created_at)}</strong></span>
                </div>
            </article>
            """

        active_tasks_html = (
            "".join(_render_task_card(task, "active") for task in active_tasks)
            if active_tasks
            else '<p class="empty-state">No active tasks.</p>'
        )

        completed_tasks_sorted = sorted(
            completed_tasks, key=lambda t: float(t.completed_at or t.created_at or 0.0), reverse=True
        )
        completed_tasks_html = (
            "".join(_render_task_card(task, "completed") for task in completed_tasks_sorted[:12])
            if completed_tasks_sorted
            else '<p class="empty-state">No completed tasks yet.</p>'
        )

        session_cards_html = ""
        if sessions:
            for session in sessions[:12]:
                safe_session_id = html.escape(str(session.get("session_id", "unknown")))
                safe_robot_id = html.escape(str(session.get("robot_id", "robot-unknown")))
                safe_latest = html.escape(
                    os.path.basename(str(session.get("latest_frame_path") or "-"))
                )
                event_count = int(session.get("event_count", 0))
                frame_count = int(session.get("frame_count", 0))
                session_cards_html += f"""
                <article class="session-card">
                    <header>
                        <code>{safe_session_id}</code>
                        <span class="chip chip-session">streaming</span>
                    </header>
                    <div class="meta-grid">
                        <span>Robot: <strong>{safe_robot_id}</strong></span>
                        <span>Created: <strong>{_fmt_ts(session.get("created_at"))}</strong></span>
                        <span>Frames: <strong>{frame_count}</strong></span>
                        <span>Events: <strong>{event_count}</strong></span>
                        <span>Latest Frame: <strong>{safe_latest}</strong></span>
                    </div>
                </article>
                """
        else:
            session_cards_html = '<p class="empty-state">No active streaming sessions.</p>'

        load_time_str = _fmt_seconds(status.get("load_time"))
        uptime_str = _fmt_seconds(status.get("uptime"))
        model_state = "online" if status.get("model_loaded") else "offline"
        model_class = "chip-ok" if status.get("model_loaded") else "chip-alert"
        model_dir = html.escape(str(status.get("model_dir", "-")))
        device = html.escape(str(status.get("device", "-")))
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Depth Anything 3 Dashboard</title>
    <style>
        :root {{
            --ink: #172a3e;
            --ink-soft: #546980;
            --paper: #f4f3ee;
            --panel: rgba(255, 255, 255, 0.86);
            --line: rgba(23, 42, 62, 0.14);
            --accent: #db5d16;
            --accent-2: #0e7f7c;
            --accent-3: #1b5dbf;
            --ok-bg: #d7f5ec;
            --ok-fg: #0f7757;
            --warn-bg: #fff3d6;
            --warn-fg: #8a5800;
            --err-bg: #fde2de;
            --err-fg: #8d2e23;
            --focus: #0a7a75;
            --shadow: 0 14px 30px rgba(18, 38, 62, 0.14);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 14% 0%, rgba(219, 93, 22, 0.17), transparent 36%),
                radial-gradient(circle at 88% 100%, rgba(14, 127, 124, 0.2), transparent 40%),
                linear-gradient(145deg, #fbfaf6 0%, #f0efe8 42%, #e8efea 100%);
            min-height: 100vh;
            padding: 18px;
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}

        .hero {{
            border: 1px solid var(--line);
            border-radius: 22px;
            background: var(--panel);
            backdrop-filter: blur(8px);
            box-shadow: var(--shadow);
            padding: 22px;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            justify-content: space-between;
            align-items: flex-start;
            animation: rise 520ms ease-out both;
        }}

        .hero h1 {{
            margin: 0 0 8px;
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
            letter-spacing: -0.015em;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
        }}

        .hero p {{
            margin: 0;
            color: var(--ink-soft);
            max-width: 66ch;
        }}

        .hero-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }}

        button, .link-btn {{
            border: 1px solid rgba(27, 93, 191, 0.3);
            background: rgba(27, 93, 191, 0.1);
            color: #164a97;
            border-radius: 10px;
            padding: 9px 14px;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }}

        button:hover, .link-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 14px rgba(17, 56, 119, 0.18);
        }}

        button:focus-visible, .link-btn:focus-visible, input:focus-visible {{
            outline: 3px solid var(--focus);
            outline-offset: 2px;
        }}

        .switch {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: var(--ink-soft);
            font-size: 0.92rem;
        }}

        .stamp {{
            margin-top: 8px;
            color: var(--ink-soft);
            font-size: 0.85rem;
        }}

        .stats-grid {{
            margin-top: 14px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
            animation: rise 620ms ease-out both;
        }}

        .stat-card {{
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--panel);
            padding: 14px 16px;
            box-shadow: var(--shadow);
        }}

        .stat-card h2 {{
            margin: 0;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--ink-soft);
        }}

        .stat-card .value {{
            margin-top: 8px;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        .stat-card .meta {{
            margin-top: 6px;
            color: var(--ink-soft);
            font-size: 0.86rem;
        }}

        .layout {{
            margin-top: 14px;
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 12px;
        }}

        .panel {{
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--panel);
            box-shadow: var(--shadow);
            padding: 14px;
            animation: rise 700ms ease-out both;
        }}

        .panel h3 {{
            margin: 0 0 10px;
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
            letter-spacing: -0.01em;
        }}

        .stack {{
            display: grid;
            gap: 10px;
        }}

        .task-card, .session-card {{
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.82);
        }}

        .task-card header, .session-card header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}

        code {{
            font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
            font-size: 0.83rem;
            padding: 3px 7px;
            border-radius: 8px;
            background: rgba(27, 93, 191, 0.08);
            color: #1f4378;
        }}

        .task-message {{
            margin: 0 0 8px;
            color: #263f5b;
            line-height: 1.4;
            word-break: break-word;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 6px 12px;
            color: var(--ink-soft);
            font-size: 0.84rem;
        }}

        .meta-grid strong {{
            color: #203955;
            font-weight: 650;
            margin-left: 4px;
        }}

        .chip {{
            border-radius: 999px;
            padding: 3px 9px;
            font-size: 0.73rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border: 1px solid transparent;
        }}

        .chip-ok {{
            background: var(--ok-bg);
            color: var(--ok-fg);
            border-color: rgba(15, 119, 87, 0.24);
        }}

        .chip-alert {{
            background: var(--err-bg);
            color: var(--err-fg);
            border-color: rgba(141, 46, 35, 0.2);
        }}

        .chip-session {{
            background: rgba(27, 93, 191, 0.1);
            color: #1f4f91;
            border-color: rgba(27, 93, 191, 0.28);
        }}

        .chip-pending {{
            background: var(--warn-bg);
            color: var(--warn-fg);
            border-color: rgba(138, 88, 0, 0.24);
        }}

        .chip-running {{
            background: rgba(214, 244, 255, 0.84);
            color: #0f5b7e;
            border-color: rgba(15, 91, 126, 0.2);
        }}

        .chip-completed {{
            background: var(--ok-bg);
            color: var(--ok-fg);
            border-color: rgba(15, 119, 87, 0.24);
        }}

        .chip-failed {{
            background: var(--err-bg);
            color: var(--err-fg);
            border-color: rgba(141, 46, 35, 0.2);
        }}

        .progress-track {{
            height: 7px;
            width: 100%;
            border-radius: 999px;
            background: rgba(27, 93, 191, 0.12);
            overflow: hidden;
            margin: 8px 0 10px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #db5d16, #1b5dbf);
            border-radius: 999px;
            transition: width 220ms ease;
        }}

        .empty-state {{
            color: var(--ink-soft);
            margin: 4px 0;
        }}

        @keyframes rise {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 980px) {{
            .layout {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <main class="container">
        <section class="hero">
            <div>
                <h1>Backend Operations Dashboard</h1>
                <p>Monitor model availability, streaming sessions, and queued jobs in one place.</p>
                <div class="stamp">Last updated: <span id="lastUpdate">{now_str}</span></div>
            </div>
            <div>
                <div class="hero-actions">
                    <button type="button" onclick="location.reload()">Refresh</button>
                    <a class="link-btn" href="/">Home</a>
                    <a class="link-btn" href="/status">Status JSON</a>
                    <label class="switch">
                        <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()">
                        Auto refresh (5s)
                    </label>
                </div>
            </div>
        </section>

        <section class="stats-grid">
            <article class="stat-card">
                <h2>Model</h2>
                <div class="value"><span class="chip {model_class}">{model_state}</span></div>
                <div class="meta">Device: {device}</div>
                <div class="meta">Load time: {load_time_str}</div>
                <div class="meta">Uptime: {uptime_str}</div>
            </article>
            <article class="stat-card">
                <h2>Sessions</h2>
                <div class="value">{len(sessions)}</div>
                <div class="meta">Active streaming sessions tracked in memory</div>
            </article>
            <article class="stat-card">
                <h2>Active Tasks</h2>
                <div class="value">{len(active_tasks)}</div>
                <div class="meta">Queued + running inference jobs</div>
            </article>
            <article class="stat-card">
                <h2>Completed Tasks</h2>
                <div class="value">{len(completed_tasks)}</div>
                <div class="meta">Finished jobs retained in task history</div>
            </article>
            <article class="stat-card">
                <h2>Model Directory</h2>
                <div class="meta" style="word-break: break-all;">{model_dir}</div>
            </article>
        </section>

        <section class="layout">
            <article class="panel">
                <h3>Streaming Sessions</h3>
                <div class="stack">
                    {session_cards_html}
                </div>
            </article>

            <div class="stack">
                <article class="panel">
                    <h3>Active Tasks</h3>
                    <div class="stack">
                        {active_tasks_html}
                    </div>
                </article>
                <article class="panel">
                    <h3>Recent Completed Tasks</h3>
                    <div class="stack">
                        {completed_tasks_html}
                    </div>
                </article>
            </div>
        </section>
    </main>

    <script>
        let autoRefreshInterval = null;

        function toggleAutoRefresh() {{
            const checkbox = document.getElementById("autoRefresh");
            if (checkbox.checked) {{
                autoRefreshInterval = setInterval(() => {{
                    location.reload();
                }}, 5000);
            }} else if (autoRefreshInterval) {{
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }}
        }}

        setInterval(() => {{
            const now = new Date();
            const stamp = document.getElementById("lastUpdate");
            if (stamp) {{
                stamp.textContent = now.toLocaleString();
            }}
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

    @_app.post("/v1/sessions", response_model=SessionCreateResponse)
    async def create_session(request: SessionCreateRequest):
        """Create a streaming session for edge frame uploads."""
        if _session_root is None:
            raise HTTPException(status_code=500, detail="Session storage is not initialized")
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        session_dir = os.path.join(_session_root, session_id)
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        config = dict(request.config or {})
        chunk_size = _safe_positive_int(
            config.get("chunk_size"), STREAM_DEFAULT_CHUNK_SIZE, minimum=1, maximum=256
        )
        max_inflight_chunks = _safe_positive_int(
            config.get("max_inflight_chunks"), STREAM_DEFAULT_MAX_INFLIGHT, minimum=1, maximum=8
        )
        auto_flush = bool(config.get("auto_flush", False))
        created_at = time.time()
        with _stream_lock:
            _stream_sessions[session_id] = {
                "session_id": session_id,
                "robot_id": request.robot_id,
                "camera": request.camera,
                "config": config,
                "created_at": created_at,
                "session_dir": session_dir,
                "frames_dir": frames_dir,
                "frames": [],
                "next_upload_index": 0,
                "frame_count": 0,
                "next_frame_index": 0,
                "next_chunk_id": 0,
                "inflight_task_ids": [],
                "completed_chunks": 0,
                "failed_chunks": 0,
                "chunk_size": chunk_size,
                "max_inflight_chunks": max_inflight_chunks,
                "auto_flush": auto_flush,
                "flush_requested": False,
                "latest_frame_path": None,
                "latest_chunk": None,
                "map_pointer": None,
                "last_inference_at": None,
                "worker_state": "idle",
                "events": [],
            }
            _append_session_event(
                _stream_sessions[session_id],
                "session_created",
                {
                    "robot_id": request.robot_id,
                    "chunk_size": chunk_size,
                    "max_inflight_chunks": max_inflight_chunks,
                    "auto_flush": auto_flush,
                },
            )

        return SessionCreateResponse(
            session_id=session_id,
            upload_url=f"/v1/sessions/{session_id}/frames",
            events_url=f"/v1/sessions/{session_id}/events",
            map_url=f"/v1/sessions/{session_id}/map/latest",
            created_at=created_at,
        )

    @_app.get("/v1/sessions")
    async def list_sessions():
        """List active streaming sessions."""
        with _stream_lock:
            sessions = []
            for s in _stream_sessions.values():
                sessions.append(
                    {
                        "session_id": s["session_id"],
                        "robot_id": s["robot_id"],
                        "created_at": s["created_at"],
                        "frame_count": s["frame_count"],
                        "pending_frames": _pending_stream_frames(s),
                        "inflight_chunks": len(s["inflight_task_ids"]),
                        "completed_chunks": s["completed_chunks"],
                        "failed_chunks": s["failed_chunks"],
                        "worker_state": s["worker_state"],
                    }
                )
        return {"total": len(sessions), "sessions": sessions}

    @_app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session metadata."""
        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return {
                "session_id": session["session_id"],
                "robot_id": session["robot_id"],
                "created_at": session["created_at"],
                "camera": session["camera"],
                "config": session["config"],
                "frame_count": session["frame_count"],
                "pending_frames": _pending_stream_frames(session),
                "chunk_size": session["chunk_size"],
                "max_inflight_chunks": session["max_inflight_chunks"],
                "auto_flush": session["auto_flush"],
                "inflight_chunks": len(session["inflight_task_ids"]),
                "completed_chunks": session["completed_chunks"],
                "failed_chunks": session["failed_chunks"],
                "latest_frame_path": session["latest_frame_path"],
                "latest_chunk": session["latest_chunk"],
                "map_pointer": session["map_pointer"],
                "last_inference_at": session["last_inference_at"],
                "worker_state": session["worker_state"],
            }

    @_app.post("/v1/sessions/{session_id}/frames", response_model=FrameUploadResponse)
    async def upload_frame(
        session_id: str,
        frame: UploadFile = File(...),
        frame_id: Optional[str] = Form(None),
        timestamp_ns: Optional[int] = Form(None),
        prior_pose_json: Optional[str] = Form(None),
    ):
        """
        Upload a single frame to a session.

        This endpoint is intentionally lightweight for MVP ingestion:
        it stores frame bytes and records metadata for downstream workers.
        """
        frame_bytes = await frame.read()
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="Empty frame payload")

        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            frame_index = int(session.get("next_upload_index", session["frame_count"]))
            session["next_upload_index"] = frame_index + 1
            frames_dir = session["frames_dir"]

        safe_frame_id = _safe_name(frame_id, f"{frame_index:06d}")
        ext = os.path.splitext(frame.filename or "")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"

        file_name = f"{frame_index:06d}_{safe_frame_id}{ext}"
        frame_path = os.path.join(frames_dir, file_name)
        with open(frame_path, "wb") as f:
            f.write(frame_bytes)

        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            session["frame_count"] = len(session["frames"]) + 1
            session["latest_frame_path"] = frame_path
            session["frames"].append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "frame_path": frame_path,
                    "has_prior_pose": bool(prior_pose_json),
                }
            )
            _append_session_event(
                session,
                "frame_uploaded",
                {
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "frame_path": frame_path,
                    "has_prior_pose": bool(prior_pose_json),
                },
            )
            auto_flush = bool(session["auto_flush"])
            total_frames = int(session["frame_count"])

        queued_tasks = _schedule_stream_inference(session_id, force=auto_flush)
        with _stream_lock:
            session = _stream_sessions.get(session_id)
            pending_frames = _pending_stream_frames(session) if session is not None else 0

        return FrameUploadResponse(
            success=True,
            session_id=session_id,
            frame_index=frame_index,
            frame_path=frame_path,
            total_frames=total_frames,
            queued_tasks=queued_tasks,
            pending_frames=pending_frames,
        )

    @_app.get("/v1/sessions/{session_id}/events")
    async def get_session_events(session_id: str, limit: int = 100):
        """Get recent session events (MVP polling endpoint)."""
        limit = max(1, min(limit, 1000))
        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            events = session["events"][-limit:]
        return {"session_id": session_id, "events": events, "count": len(events)}

    @_app.post("/v1/sessions/{session_id}/flush")
    async def flush_session(session_id: str):
        """
        Force scheduling of pending frames, even when chunk is incomplete.

        Useful for low-FPS streams and explicit end-of-segment flushes.
        """
        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            pending_before = _pending_stream_frames(session)
            inflight_before = len(session["inflight_task_ids"])
            session["flush_requested"] = True

        queued_tasks = _schedule_stream_inference(session_id, force=True)

        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            pending_after = _pending_stream_frames(session)
            inflight_after = len(session["inflight_task_ids"])
            if pending_after <= 0 and inflight_after <= 0:
                session["flush_requested"] = False
            _append_session_event(
                session,
                "session_flushed",
                {
                    "queued_tasks": queued_tasks,
                    "pending_before": pending_before,
                    "pending_after": pending_after,
                    "inflight_before": inflight_before,
                    "inflight_after": inflight_after,
                },
            )

        return {
            "session_id": session_id,
            "queued_tasks": queued_tasks,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "inflight_before": inflight_before,
            "inflight_after": inflight_after,
        }

    @_app.get("/v1/sessions/{session_id}/map/latest")
    async def get_latest_map(session_id: str):
        """Return latest map snapshot metadata for the session pipeline."""
        with _stream_lock:
            session = _stream_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            pending = _pending_stream_frames(session)
            inflight = len(session["inflight_task_ids"])
            if session["map_pointer"] is not None:
                status = "ready"
            elif pending > 0 or inflight > 0:
                status = "processing"
            else:
                status = "idle"
            return {
                "session_id": session_id,
                "status": status,
                "frame_count": session["frame_count"],
                "pending_frames": pending,
                "inflight_chunks": inflight,
                "completed_chunks": session["completed_chunks"],
                "failed_chunks": session["failed_chunks"],
                "latest_frame_path": session["latest_frame_path"],
                "latest_chunk": session["latest_chunk"],
                "last_inference_at": session["last_inference_at"],
                "map_pointer": session["map_pointer"],
            }

    @_app.post("/inference", response_model=InferenceResponse)
    async def run_inference(request: InferenceRequest):
        """Submit inference task and return task ID."""
        if _backend is None:
            raise HTTPException(status_code=500, detail="Backend not initialized")

        task_id = _enqueue_inference_task(request=request, task_kind="manual")

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
