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
Modules package for Depth Anything 3 Gradio app.

This package contains all the modular components for the Gradio application.
"""

from depth_anything_3.app.modules.event_handlers import EventHandlers
from depth_anything_3.app.modules.file_handlers import FileHandler
from depth_anything_3.app.modules.model_inference import ModelInference
from depth_anything_3.app.modules.ui_components import UIComponents
from depth_anything_3.app.modules.utils import (
    create_depth_visualization,
    get_logo_base64,
    get_scene_info,
    save_to_gallery_func,
)
from depth_anything_3.app.modules.visualization import VisualizationHandler

__all__ = [
    "ModelInference",
    "FileHandler",
    "VisualizationHandler",
    "EventHandlers",
    "UIComponents",
    "create_depth_visualization",
    "save_to_gallery_func",
    "get_scene_info",
    "get_logo_base64",
]
