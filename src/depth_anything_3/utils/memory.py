# Copyright (c) 2025 Delanoe Pirard and/or its affiliates
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
GPU memory utility helpers.

Shared cleanup and memory checking logic used by both the backend API and
the Gradio UI to keep memory-management behavior consistent.
"""
from __future__ import annotations

import gc

from typing import Any, Dict, Optional

import torch


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Return a snapshot of current GPU memory usage or None if CUDA not available.

    Keys in returned dict: total_gb, allocated_gb, reserved_gb, free_gb, utilization
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory

        return {
            "total_gb": total_memory / 1024 ** 3,
            "allocated_gb": allocated_memory / 1024 ** 3,
            "reserved_gb": reserved_memory / 1024 ** 3,
            "free_gb": free_memory / 1024 ** 3,
            "utilization": (reserved_memory / total_memory) * 100,
        }
    except Exception:
        return None


def cleanup_cuda_memory() -> None:
    """Perform a robust GPU cleanup sequence.

    This includes synchronizing, emptying caches, collecting IPC handles and
    running the Python garbage collector. Use this instead of a raw
    ``torch.cuda.empty_cache()`` where you need reliable freeing of GPU memory
    between model loads or in error handling paths.
    """
    try:
        if torch.cuda.is_available():
            mem_before = get_gpu_memory_info()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Collect cross-process cuda resources
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # Older PyTorch versions or non-cuda devices may not support
                # ipc_collect (no-op if not available)
                pass
            gc.collect()

            mem_after = get_gpu_memory_info()
            if mem_before and mem_after:
                freed = mem_before["reserved_gb"] - mem_after["reserved_gb"]
                print(
                    f"CUDA cleanup: freed {freed:.2f}GB, "
                    f"available: {mem_after['free_gb']:.2f}GB/{mem_after['total_gb']:.2f}GB"
                )
            else:
                print("CUDA memory cleanup completed")
    except Exception as e:
        print(f"Warning: CUDA cleanup failed: {e}")


def check_memory_availability(required_gb: float = 2.0) -> tuple[bool, str]:
    """Return whether at least ``required_gb`` seems available on the current GPU.

    The returned tuple is (is_available, message) with a human-friendly message.
    """
    try:
        if not torch.cuda.is_available():
            return False, "CUDA is not available"

        mem_info = get_gpu_memory_info()
        if mem_info is None:
            return True, "Cannot check memory, proceeding anyway"

        if mem_info["free_gb"] < required_gb:
            return (
                False,
                (
                    f"Insufficient GPU memory: {mem_info['free_gb']:.2f}GB available, "
                    f"{required_gb:.2f}GB required. Total: {mem_info['total_gb']:.2f}GB, "
                    f"Used: {mem_info['reserved_gb']:.2f}GB ({mem_info['utilization']:.1f}%)"
                ),
            )

        return (
            True,
            (
                f"Memory check passed: {mem_info['free_gb']:.2f}GB available, "
                f"{required_gb:.2f}GB required"
            ),
        )
    except Exception as e:
        return True, f"Memory check failed: {e}, proceeding anyway"
def estimate_memory_requirement(num_images: int, process_res: int) -> float:
    """Heuristic estimate for memory usage (GB) based on image count and resolution.

    This mirrors the simple policy used by the backend service so other code
    (e.g., Gradio UI) can make consistent decisions when checking available
    memory before loading a model or running inference.

    Args:
        num_images: Number of images to process.
        process_res: Processing resolution.

    Returns:
        Estimated memory requirement in GB.
    """
    base_memory = 2.0
    per_image_memory = (process_res / 504) ** 2 * 0.5
    total_memory = base_memory + (num_images * per_image_memory * 0.1)
    return total_memory


# ===========================
# Proactive Memory Management
# ===========================


def cleanup_mps_memory() -> None:
    """
    Perform proactive MPS memory cleanup.

    MPS (Apple Silicon) has unified memory architecture where CPU and GPU
    share the same memory pool. Proactive cleanup prevents fragmentation.
    """
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()
            print("MPS memory cache cleared")
    except Exception as e:
        print(f"Warning: MPS cleanup failed: {e}")


def cleanup_all_device_memory() -> None:
    """
    Clean up memory for all available devices (CUDA, MPS, CPU).

    Call this between batch processing or after large allocations
    to prevent memory fragmentation and OOM errors.

    Example:
        >>> from depth_anything_3.utils.memory import cleanup_all_device_memory
        >>> # Process batch 1
        >>> model.inference(images_batch1)
        >>> cleanup_all_device_memory()  # Clean between batches
        >>> # Process batch 2
        >>> model.inference(images_batch2)
    """
    cleanup_cuda_memory()
    cleanup_mps_memory()
    gc.collect()


def clear_cache_if_low_memory(threshold_gb: float = 2.0) -> bool:
    """
    Conditionally clear cache if available memory is below threshold.

    Args:
        threshold_gb: Memory threshold in GB (default: 2.0)

    Returns:
        True if cache was cleared, False otherwise

    Example:
        >>> from depth_anything_3.utils.memory import clear_cache_if_low_memory
        >>> # Before large allocation
        >>> if clear_cache_if_low_memory(threshold_gb=3.0):
        ...     print("Low memory detected, cache cleared")
    """
    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        if mem_info and mem_info["free_gb"] < threshold_gb:
            print(f"Low memory detected ({mem_info['free_gb']:.2f} GB < {threshold_gb:.2f} GB)")
            cleanup_cuda_memory()
            return True

    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        # MPS doesn't expose free memory easily, always clear if requested
        cleanup_mps_memory()
        return True

    return False


def log_memory_summary() -> None:
    """
    Log current memory usage summary for all devices.

    Useful for debugging memory issues or understanding memory patterns.
    """
    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        if mem_info:
            print(
                f"[CUDA Memory] Allocated: {mem_info['allocated_gb']:.2f} GB, "
                f"Reserved: {mem_info['reserved_gb']:.2f} GB, "
                f"Free: {mem_info['free_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
                f"({mem_info['utilization']:.1f}% used)"
            )

    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            driver_allocated = torch.mps.driver_allocated_memory() / (1024**3)
            print(
                f"[MPS Memory] Allocated: {allocated:.2f} GB, "
                f"Driver Allocated: {driver_allocated:.2f} GB"
            )
        except Exception as e:
            print(f"[MPS Memory] Stats unavailable: {e}")

    else:
        print("[CPU Memory] Stats not available via PyTorch")
