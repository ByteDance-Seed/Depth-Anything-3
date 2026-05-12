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

"""Services module for Depth Anything 3.

`backend` is lazy-imported because it requires torch (it runs the PyTorch
`DepthAnything3` API). Loading `depth_anything_3.services` should be cheap
and torch-free so the ONNX CLI commands work without `[torch]` installed.
"""


def __getattr__(name):
    if name in ("create_app", "start_server"):
        # Lazy-import: requires torch via .backend -> ..api -> ..model -> torch
        from depth_anything_3.services.backend import create_app, start_server

        return {"create_app": create_app, "start_server": start_server}[name]
    raise AttributeError(f"module 'depth_anything_3.services' has no attribute {name!r}")


__all__ = ["create_app", "start_server"]
