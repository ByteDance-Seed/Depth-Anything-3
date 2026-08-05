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

import os
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.parallel_utils import async_call


@async_call
def export_to_mini_npy(
    prediction: Prediction,
    export_dir: str,
):
    output_dir = os.path.join(export_dir, "exports", "mini_npy")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "depth.npy"), np.round(prediction.depth, 8))
    if prediction.conf is not None:
        np.save(os.path.join(output_dir, "conf.npy"), np.round(prediction.conf, 2))
    if prediction.extrinsics is not None:
        np.save(os.path.join(output_dir, "extrinsics.npy"), prediction.extrinsics)
    if prediction.intrinsics is not None:
        np.save(os.path.join(output_dir, "intrinsics.npy"), prediction.intrinsics)


@async_call
def export_to_npy(
    prediction: Prediction,
    export_dir: str,
):
    output_dir = os.path.join(export_dir, "exports", "npy")
    os.makedirs(output_dir, exist_ok=True)

    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    image = prediction.processed_images  # (N,H,W,3) uint8

    np.save(os.path.join(output_dir, "image.npy"), image)
    np.save(os.path.join(output_dir, "depth.npy"), np.round(prediction.depth, 8))

    if prediction.conf is not None:
        np.save(os.path.join(output_dir, "conf.npy"), np.round(prediction.conf, 2))
    if prediction.extrinsics is not None:
        np.save(os.path.join(output_dir, "extrinsics.npy"), prediction.extrinsics)
    if prediction.intrinsics is not None:
        np.save(os.path.join(output_dir, "intrinsics.npy"), prediction.intrinsics)