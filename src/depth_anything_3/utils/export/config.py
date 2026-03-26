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

"""Configuration export utilities for experiment tracking."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from depth_anything_3.specs import Prediction


def export_to_config(
    prediction: Prediction,
    export_dir: str,
    model_name: Optional[str] = None,
    input_info: Optional[Dict[str, Any]] = None,
    processing_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Export configuration and metadata to JSON file for experiment tracking.
    
    Args:
        prediction: Model prediction containing inference results
        export_dir: Output directory where config.json will be written
        model_name: Name of the model used (e.g., "DA3-LARGE")
        input_info: Dictionary containing input information (e.g., video properties, image list)
        processing_params: Dictionary containing processing parameters (e.g., process_res, fps)
    
    Returns:
        Path to the exported config.json file
    """
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model_name or "unknown",
        },
    }
    
    # Add input information
    if input_info:
        config["input"] = input_info
    
    # Add processing parameters
    if processing_params:
        config["processing"] = processing_params
    
    # Add prediction results information
    results = {}
    if prediction.depth is not None:
        results["depth_shape"] = list(prediction.depth.shape)
    if prediction.conf is not None:
        results["confidence_shape"] = list(prediction.conf.shape)
    if prediction.extrinsics is not None:
        results["extrinsics_shape"] = list(prediction.extrinsics.shape)
    if prediction.intrinsics is not None:
        results["intrinsics_shape"] = list(prediction.intrinsics.shape)
    if prediction.features is not None and len(prediction.features) > 0:
        results["feature_layers"] = len(prediction.features)
        results["feature_shapes"] = [list(f.shape) for f in prediction.features]
    
    config["results"] = results
    
    # List output files
    output_dir = Path(export_dir)
    output_files = []
    
    for file_pattern in ["scene.glb", "scene.jpg", "*.ply"]:
        for file_path in output_dir.glob(file_pattern):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024**2)
                output_files.append({
                    "file": file_path.name,
                    "size_mb": round(size_mb, 2)
                })
    
    for dir_name in ["depth_vis", "feat_vis", "gs_video", "input_images"]:
        dir_path = output_dir / dir_name
        if dir_path.is_dir():
            files = list(dir_path.iterdir())
            output_files.append({
                "directory": dir_name,
                "num_files": len(files)
            })
    
    if output_files:
        config["output_files"] = output_files
    
    # Write config file
    os.makedirs(export_dir, exist_ok=True)
    config_path = os.path.join(export_dir, "config.json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path
