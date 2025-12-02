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
Zero-copy utilities for efficient tensor operations.

Provides utilities to minimize memory copies between NumPy and PyTorch,
especially for CPU→GPU transfers.
"""

from __future__ import annotations

import numpy as np
import torch


def numpy_to_torch_zerocopy(arr: np.ndarray, dtype: torch.dtype | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor with zero-copy when possible.

    Zero-copy is possible when:
    1. Array is C-contiguous
    2. Target device is CPU
    3. dtype is compatible

    For GPU transfers, this still saves one copy (CPU→pinned→GPU vs CPU→CPU→GPU).

    Args:
        arr: Input NumPy array
        dtype: Target PyTorch dtype (if None, infer from numpy dtype)
        device: Target device ('cpu', 'cuda', 'mps')

    Returns:
        PyTorch tensor on specified device

    Example:
        >>> arr = np.random.rand(1000, 1000)
        >>> tensor = numpy_to_torch_zerocopy(arr, device='cuda')
        >>> # No intermediate copy on CPU if arr is C-contiguous
    """
    # Check if zero-copy is possible
    is_contiguous = arr.flags['C_CONTIGUOUS']

    if not is_contiguous:
        # Need to make contiguous copy anyway
        arr = np.ascontiguousarray(arr)

    # Create tensor with zero-copy (shares memory on CPU)
    tensor = torch.from_numpy(arr)

    # Apply dtype conversion if needed
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    # Move to target device
    if str(device) != "cpu":
        # Use non_blocking for async transfer
        tensor = tensor.to(device, non_blocking=True)

    return tensor


def ensure_pinned_memory(arr: np.ndarray) -> np.ndarray:
    """
    Ensure NumPy array uses pinned (page-locked) memory for faster GPU transfers.

    Pinned memory allows DMA (Direct Memory Access) for faster CPU→GPU transfers.
    Only beneficial for repeated transfers of the same data.

    Args:
        arr: Input NumPy array

    Returns:
        Array in pinned memory

    Note:
        Pinned memory is a limited resource. Only use for frequently transferred data.
        For CUDA devices only (no effect on MPS/CPU).
    """
    if not torch.cuda.is_available():
        return arr

    # Convert to torch tensor with pinned memory
    tensor = torch.from_numpy(arr).pin_memory()

    # Convert back to numpy (shares pinned memory)
    # Note: This creates a new numpy array view over pinned memory
    return tensor.numpy()


def stack_arrays_zerocopy(arrays: list[np.ndarray], dtype: np.dtype | None = None) -> np.ndarray:
    """
    Stack list of arrays with minimal copying.

    Args:
        arrays: List of NumPy arrays to stack
        dtype: Target dtype (if None, use arrays[0].dtype)

    Returns:
        Stacked array

    Note:
        If all arrays already have compatible dtype and layout,
        np.stack uses optimized C-level stacking.
    """
    if not arrays:
        raise ValueError("Cannot stack empty list")

    # Check if all arrays have compatible dtype
    if dtype is None:
        dtype = arrays[0].dtype

    # Ensure all arrays are C-contiguous with same dtype
    # This may create copies, but better done once than repeatedly
    arrays_contig = []
    for arr in arrays:
        if arr.dtype != dtype or not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr, dtype=dtype)
        arrays_contig.append(arr)

    # Stack (single memory allocation + copy)
    return np.stack(arrays_contig, axis=0)


def batch_to_device(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    device: str | torch.device,
    non_blocking: bool = True
) -> list[torch.Tensor]:
    """
    Move multiple tensors to device with optimal settings.

    Args:
        tensors: List/tuple of tensors to move
        device: Target device
        non_blocking: Use async transfer (default: True)

    Returns:
        List of tensors on target device

    Example:
        >>> tensors = [torch.rand(100), torch.rand(200)]
        >>> gpu_tensors = batch_to_device(tensors, 'cuda')
    """
    return [t.to(device, non_blocking=non_blocking) if t is not None else None for t in tensors]


def get_optimal_pin_memory() -> bool:
    """
    Determine if pin_memory should be used for DataLoader.

    Returns:
        True if CUDA is available and pinned memory is beneficial

    Usage:
        >>> DataLoader(dataset, pin_memory=get_optimal_pin_memory())
    """
    return torch.cuda.is_available()