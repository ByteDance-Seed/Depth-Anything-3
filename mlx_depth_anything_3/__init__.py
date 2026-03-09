# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""MLX implementation of Depth Anything 3 for Apple Silicon."""

from mlx_depth_anything_3.model import DepthAnything3Net
from mlx_depth_anything_3.inference import DepthAnything3

__all__ = ["DepthAnything3Net", "DepthAnything3"]
