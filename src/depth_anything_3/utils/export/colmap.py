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
import pycolmap
import cv2 as cv
import numpy as np

from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
    voxel_size: float = 0.05,  # Merge points within 5cm. Increase for fewer points.
) -> None:
    """
    Exports ML-based depth and poses to COLMAP format with voxel downsampling
    to ensure 3D points have tracks across multiple frames.
    """
    # 1. Data preparation (Filtered by confidence)
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)

    # Get 3D points and colors (using your existing internal helper)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )

    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]

    # Create pixel coordinates corresponding to each point
    all_xyf = _create_xyf(num_frames, h, w)
    points_xyf = all_xyf[prediction.conf >= conf_thresh]

    # 2. Voxel Downsampling & Merging (This creates the Tracks)
    # Group points by their grid location in 3D space
    v_keys = (points / voxel_size).astype(int)
    unique_v_keys, inverse_indices = np.unique(v_keys, axis=0, return_inverse=True)
    num_voxels = len(unique_v_keys)

    logger.info(
        f"Merging {len(points)} pixels into {num_voxels} unique 3D points (voxel_size={voxel_size})"
    )

    # Compute mean 3D position and color for each voxel group
    voxel_points = np.zeros((num_voxels, 3))
    voxel_colors = np.zeros((num_voxels, 3))
    counts = np.zeros(num_voxels)
    np.add.at(voxel_points, inverse_indices, points)
    np.add.at(voxel_colors, inverse_indices, colors)
    np.add.at(counts, inverse_indices, 1)

    voxel_points /= counts[:, None]
    voxel_colors = (voxel_colors / counts[:, None]).astype(np.uint8)

    # 3. Initialize Reconstruction
    reconstruction = pycolmap.Reconstruction()
    point3d_ids = []
    for i in range(num_voxels):
        # Create a new 3D point in the model. Initially it has an empty track.
        pid = reconstruction.add_point3D(voxel_points[i], pycolmap.Track(), voxel_colors[i])
        point3d_ids.append(pid)

    # 4. Loop through frames to add Cameras and link 2D observations to 3D points
    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        # Scaling intrinsics to match original image size
        intrinsic = prediction.intrinsics[fidx].copy()
        if process_res_method.endswith("resize"):
            intrinsic[0, 0] *= orig_w / w
            intrinsic[1, 1] *= orig_h / h
            intrinsic[0, 2] *= orig_w / w
            intrinsic[1, 2] *= orig_h / h

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # Link 2D pixels to the voxelized 3D points
        point2d_list = []
        in_frame_indices = np.where(points_xyf[:, 2] == fidx)[0]

        # Prevent adding the same 3D point twice to a single image
        seen_voxels_in_this_image = set()

        for idx in in_frame_indices:
            voxel_idx = inverse_indices[idx]

            if voxel_idx in seen_voxels_in_this_image:
                continue
            seen_voxels_in_this_image.add(voxel_idx)

            point2d = points_xyf[idx][:2].astype(float).copy()
            point2d[0] *= orig_w / w
            point2d[1] *= orig_h / h

            pt3d_id = point3d_ids[voxel_idx]
            point2d_list.append(pycolmap.Point2D(point2d, pt3d_id))

            # This is the crucial step: add this observation to the track.
            # point3D.track now contains multiple images that see the same physical point.
            reconstruction.point3D(pt3d_id).track.add_element(
                image.image_id, len(point2d_list) - 1
            )

        # set and add image
        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    # 5. Export to disk
    os.makedirs(export_dir, exist_ok=True)
    reconstruction.write(export_dir)
    reconstruction.write_text(export_dir)
    logger.info(f"Successfully exported COLMAP model to {export_dir}")


def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
