"""
Model caching utilities for Depth Anything 3.

Provides model caching functionality to avoid reloading model weights on every instantiation.
This significantly reduces latency for repeated model creation (2-5s gain).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from depth_anything_3.utils.logger import logger


class ModelCache:
    """
    Thread-safe singleton cache for Depth Anything 3 models.

    Caches loaded model weights to avoid reloading from disk on every instantiation.
    Each unique combination of (model_name, device) is cached separately.

    Usage:
        cache = ModelCache()
        model = cache.get(model_name, device, loader_fn)
        # loader_fn is only called if cache miss

    Thread Safety:
        Uses threading.Lock to ensure thread-safe access to cache.

    Memory Management:
        - Models are kept in cache until explicitly cleared
        - Use clear() to free memory when needed
        - Use clear_device() to clear specific device models
    """

    _instance: Optional["ModelCache"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure single cache instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize cache storage."""
        if self._initialized:
            return

        self._cache: Dict[Tuple[str, str], nn.Module] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True
        logger.info("ModelCache initialized")

    def get(
        self,
        model_name: str,
        device: torch.device | str,
        loader_fn: callable,
    ) -> nn.Module:
        """
        Get cached model or load if not in cache.

        Args:
            model_name: Name of the model (e.g., "da3-large")
            device: Target device (cuda, mps, cpu)
            loader_fn: Function to load model if cache miss
                      Should return nn.Module

        Returns:
            Cached or freshly loaded model on specified device

        Example:
            >>> cache = ModelCache()
            >>> model = cache.get(
            ...     "da3-large",
            ...     "cuda",
            ...     lambda: create_model()
            ... )
        """
        device_str = str(device)
        cache_key = (model_name, device_str)

        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug(f"Model cache HIT: {model_name} on {device_str}")
                return self._cache[cache_key]

            logger.info(f"Model cache MISS: {model_name} on {device_str}. Loading...")
            model = loader_fn()
            self._cache[cache_key] = model
            logger.info(f"Model cached: {model_name} on {device_str}")

            return model

    def clear(self) -> None:
        """
        Clear entire cache and free memory.

        Removes all cached models and forces garbage collection.
        Useful when switching between many different models.
        """
        with self._cache_lock:
            num_cached = len(self._cache)
            self._cache.clear()

            # Force garbage collection to free GPU memory
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logger.info(f"Model cache cleared ({num_cached} models removed)")

    def clear_device(self, device: torch.device | str) -> None:
        """
        Clear all models on specific device.

        Args:
            device: Device to clear (e.g., "cuda", "mps", "cpu")

        Example:
            >>> cache = ModelCache()
            >>> cache.clear_device("cuda")  # Clear all CUDA models
        """
        device_str = str(device)

        with self._cache_lock:
            keys_to_remove = [key for key in self._cache if key[1] == device_str]
            for key in keys_to_remove:
                del self._cache[key]

            # Free device memory
            if "cuda" in device_str and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif "mps" in device_str and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logger.info(f"Model cache cleared for device {device_str} ({len(keys_to_remove)} models removed)")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache info:
                - total: Total number of cached models
                - by_device: Number of models per device
        """
        with self._cache_lock:
            info = {
                "total": len(self._cache),
                "by_device": {},
            }

            for model_name, device_str in self._cache.keys():
                if device_str not in info["by_device"]:
                    info["by_device"][device_str] = 0
                info["by_device"][device_str] += 1

            return info


# Global singleton instance
_global_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """
    Get global model cache instance.

    Returns:
        Singleton ModelCache instance

    Example:
        >>> from depth_anything_3.cache import get_model_cache
        >>> cache = get_model_cache()
        >>> cache.clear()
    """
    return _global_cache