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

"""ONNX runtime inference for Depth Anything 3.

Public entry points:

* :class:`DepthAnything3Onnx` — single-net ONNX inference (depth + camera).
* :class:`DepthAnything3OnnxNested` — main + metric ONNX with NumPy alignment.

The ONNX export tool (``depth_anything_3.onnx.export``) is intentionally NOT
imported here, because it requires ``torch``. Use ``import
depth_anything_3.onnx.export`` (or the ``da3 onnx-export`` CLI command) when
you need it.
"""

from depth_anything_3.onnx.api import DepthAnything3Onnx, DepthAnything3OnnxNested

__all__ = ["DepthAnything3Onnx", "DepthAnything3OnnxNested"]
