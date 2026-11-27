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
Utility functions for benchmark evaluation.

Contains:
- Pose evaluation metrics (AUC) and helper functions
- 3D reconstruction evaluation metrics (Acc/Comp/F-score)
- Geometry utilities (quaternion conversion, etc.)
"""

from typing import Dict as TDict, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from addict import Dict
from scipy.spatial import KDTree

from depth_anything_3.utils.geometry import mat_to_quat


# =============================================================================
# Geometry Utilities
# =============================================================================


def quat2rotmat(qvec: list) -> np.ndarray:
    """
    Convert quaternion (WXYZ order) to rotation matrix.

    Args:
        qvec: Quaternion as [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    rotmat = np.array(
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ]
    )
    rotmat = rotmat.reshape(3, 3)
    return rotmat


# =============================================================================
# 3D Reconstruction Evaluation
# =============================================================================


def nn_correspondance(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:
    """
    Compute nearest neighbor distances from verts2 to verts1 using KDTree.

    Args:
        verts1: Reference point cloud [N, 3]
        verts2: Query point cloud [M, 3]

    Returns:
        Distance array [M,] - distance from each point in verts2 to nearest in verts1
    """
    if len(verts1) == 0 or len(verts2) == 0:
        return np.array([])

    kdtree = KDTree(verts1)
    distances, _ = kdtree.query(verts2)
    return distances.reshape(-1)


def evaluate_3d_reconstruction(
    pcd_pred: Union[o3d.geometry.PointCloud, np.ndarray],
    pcd_trgt: Union[o3d.geometry.PointCloud, np.ndarray],
    threshold: float = 0.05,
    down_sample: Optional[float] = None,
) -> TDict[str, float]:
    """
    Evaluate 3D reconstruction quality using standard metrics.

    This function computes:
    - Accuracy: Mean distance from predicted points to GT surface
    - Completeness: Mean distance from GT points to predicted surface
    - Overall: Average of accuracy and completeness
    - Precision: Fraction of predicted points within threshold of GT
    - Recall: Fraction of GT points within threshold of prediction
    - F-score: Harmonic mean of precision and recall

    Args:
        pcd_pred: Predicted point cloud (Open3D or numpy array)
        pcd_trgt: Ground truth point cloud (Open3D or numpy array)
        threshold: Distance threshold for precision/recall (meters)
        down_sample: Voxel size for downsampling (None to skip)

    Returns:
        Dict with metrics: acc, comp, overall, precision, recall, fscore
    """
    # Convert to Open3D if needed
    if isinstance(pcd_pred, np.ndarray):
        pcd_pred_o3d = o3d.geometry.PointCloud()
        pcd_pred_o3d.points = o3d.utility.Vector3dVector(pcd_pred)
        pcd_pred = pcd_pred_o3d
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt_o3d = o3d.geometry.PointCloud()
        pcd_trgt_o3d.points = o3d.utility.Vector3dVector(pcd_trgt)
        pcd_trgt = pcd_trgt_o3d

    # Downsample if requested
    if down_sample is not None and down_sample > 0:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    # Handle empty point clouds
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {
            "acc": float("inf"),
            "comp": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }

    # Compute distances
    dist_pred_to_gt = nn_correspondance(verts_trgt, verts_pred)  # Accuracy
    dist_gt_to_pred = nn_correspondance(verts_pred, verts_trgt)  # Completeness

    # Compute metrics
    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    overall = (accuracy + completeness) / 2

    precision = float(np.mean((dist_pred_to_gt < threshold).astype(float)))
    recall = float(np.mean((dist_gt_to_pred < threshold).astype(float)))

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "acc": accuracy,
        "comp": completeness,
        "overall": overall,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def create_tsdf_volume(
    voxel_length: float = 4.0 / 512.0,
    sdf_trunc: float = 0.04,
    color_type: str = "RGB8",
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """
    Create a scalable TSDF volume for depth fusion.

    Args:
        voxel_length: Size of each voxel
        sdf_trunc: Truncation distance for SDF
        color_type: Color integration type ("RGB8" or "Gray32")

    Returns:
        Initialized ScalableTSDFVolume
    """
    if color_type == "RGB8":
        color_enum = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    else:
        color_enum = o3d.pipelines.integration.TSDFVolumeColorType.Gray32

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_enum,
    )
    return volume


def fuse_depth_to_tsdf(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    depths: np.ndarray,
    images: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    max_depth: float = 10.0,
) -> o3d.geometry.TriangleMesh:
    """
    Fuse multiple depth maps into TSDF volume and extract mesh.

    Args:
        volume: TSDF volume to integrate into
        depths: Depth maps [N, H, W]
        images: RGB images [N, H, W, 3]
        intrinsics: Camera intrinsics [N, 3, 3]
        extrinsics: Camera extrinsics (world-to-camera) [N, 4, 4]
        max_depth: Maximum depth for truncation

    Returns:
        Extracted triangle mesh
    """
    for i in range(len(depths)):
        depth = depths[i]
        image = images[i]
        ixt = intrinsics[i]
        ext = extrinsics[i]

        h, w = depth.shape[:2]

        # Create RGBD image
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(image.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        # Create camera intrinsics
        ixt_o3d = o3d.camera.PinholeCameraIntrinsic(
            w, h, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        )

        # Integrate into volume
        volume.integrate(rgbd, ixt_o3d, ext)

    # Extract mesh
    mesh = volume.extract_triangle_mesh()
    return mesh


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 1000000,
) -> o3d.geometry.PointCloud:
    """
    Uniformly sample points from a triangle mesh.

    Args:
        mesh: Input triangle mesh
        num_points: Number of points to sample

    Returns:
        Sampled point cloud
    """
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        # Clamp colors to valid range [0, 1] for Open3D PLY export
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            colors = np.clip(colors, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        # Fallback: create random points if mesh is invalid (with fixed seed for reproducibility)
        rng = np.random.default_rng(seed=42)
        points = rng.uniform(-1, 1, size=(num_points, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# =============================================================================
# Pose Evaluation
# =============================================================================


def build_pair_index(N: int, B: int = 1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = ((i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_])
    return i1, i2


def compute_pose(pred_se3: torch.Tensor, gt_se3: torch.Tensor) -> Dict:
    """
    Compute pose estimation metrics between predicted and ground truth trajectories.

    Args:
        pred_se3: Predicted SE(3) transformations [N, 4, 4]
        gt_se3: Ground truth SE(3) transformations [N, 4, 4]

    Returns:
        Dict with AUC metrics at different thresholds (auc30, auc15, auc05, auc03)
    """
    pred_se3 = align_to_first_camera(pred_se3)
    gt_se3 = align_to_first_camera(gt_se3)

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, len(pred_se3))
    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()

    output = Dict()
    output.auc30, _ = calculate_auc_np(rError, tError, max_threshold=30)
    output.auc15, _ = calculate_auc_np(rError, tError, max_threshold=15)
    output.auc05, _ = calculate_auc_np(rError, tError, max_threshold=5)
    output.auc03, _ = calculate_auc_np(rError, tError, max_threshold=3)
    return output


def align_to_first_camera(camera_poses: torch.Tensor) -> torch.Tensor:
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Camera poses as SE3 transformations [N, 4, 4]

    Returns:
        Aligned camera poses [N, 4, 4]
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses


def rotation_angle(
    rot_gt: torch.Tensor, rot_pred: torch.Tensor, batch_size: int = None, eps: float = 1e-15
) -> torch.Tensor:
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(
    tvec_gt: torch.Tensor,
    tvec_pred: torch.Tensor,
    batch_size: int = None,
    ambiguity: bool = True,
) -> torch.Tensor:
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(
    t_gt: torch.Tensor, t: torch.Tensor, eps: float = 1e-15, default_err: float = 1e6
) -> torch.Tensor:
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(
    r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30
) -> tuple:
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    Args:
        r_error: Rotation error values in degrees
        t_error: Translation error values in degrees
        max_threshold: Maximum threshold value for binning

    Returns:
        Tuple of (AUC value, normalized histogram)
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(
    pred_se3: torch.Tensor, gt_se3: torch.Tensor, num_frames: int
) -> tuple:
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Tuple of (rotation angle errors, translation angle errors) in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses (cam_i1 -> cam_i2) for w2c SE(3) matrices
    relative_pose_gt = gt_se3[pair_idx_i2].bmm(closed_form_inverse_se3(gt_se3[pair_idx_i1]))
    relative_pose_pred = pred_se3[pair_idx_i2].bmm(closed_form_inverse_se3(pred_se3[pair_idx_i1]))

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def closed_form_inverse_se3(
    se3: torch.Tensor, R: torch.Tensor = None, T: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    Uses closed-form solution instead of torch.inverse() for numerical stability.

    Args:
        se3: Nx4x4 or Nx3x4 tensor of SE3 matrices
        R: Optional Nx3x3 rotation matrices
        T: Optional Nx3x1 translation vectors

    Returns:
        Inverted SE3 matrices with same shape as input
    """
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

