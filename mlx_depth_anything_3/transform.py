# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Quaternion math and camera pose encoding/decoding utilities for MLX."""

import mlx.core as mx


def extri_intri_to_pose_encoding(
    extrinsics: mx.array,
    intrinsics: mx.array,
    image_size_hw: tuple[int, int],
) -> mx.array:
    """Convert camera extrinsics (c2w) and intrinsics to a compact 9-D pose encoding.

    Args:
        extrinsics: (B, S, 4, 4) camera-to-world matrices.
        intrinsics: (B, S, 3, 3) camera intrinsic matrices.
        image_size_hw: (H, W) of the images.

    Returns:
        (B, S, 9) pose encoding: [T_x, T_y, T_z, q_x, q_y, q_z, q_w, fov_h, fov_w].
    """
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2.0 * mx.arctan(mx.array(H / 2.0) / intrinsics[:, :, 1, 1])
    fov_w = 2.0 * mx.arctan(mx.array(W / 2.0) / intrinsics[:, :, 0, 0])
    pose_encoding = mx.concatenate(
        [T, quat, mx.expand_dims(fov_h, -1), mx.expand_dims(fov_w, -1)], axis=-1
    ).astype(mx.float32)
    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding: mx.array,
    image_size_hw: tuple[int, int],
) -> tuple[mx.array, mx.array]:
    """Convert 9-D pose encoding back to extrinsics and intrinsics.

    Returns:
        extrinsics: (B, S, 3, 4)
        intrinsics: (B, S, 3, 3)
    """
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    R = quat_to_mat(quat)
    extrinsics = mx.concatenate([R, mx.expand_dims(T, -1)], axis=-1)
    H, W = image_size_hw
    fy = (H / 2.0) / mx.clip(mx.tan(fov_h / 2.0), a_min=1e-6, a_max=None)
    fx = (W / 2.0) / mx.clip(mx.tan(fov_w / 2.0), a_min=1e-6, a_max=None)
    B_dim, S_dim = pose_encoding.shape[:2]
    intrinsics = mx.zeros((*pose_encoding.shape[:2], 3, 3))
    # Build intrinsics row by row
    row0 = mx.stack([fx, mx.zeros_like(fx), mx.full_like(fx, W / 2.0)], axis=-1)
    row1 = mx.stack([mx.zeros_like(fy), fy, mx.full_like(fy, H / 2.0)], axis=-1)
    row2 = mx.stack(
        [mx.zeros_like(fx), mx.zeros_like(fx), mx.ones_like(fx)], axis=-1
    )
    intrinsics = mx.stack([row0, row1, row2], axis=-2)
    return extrinsics, intrinsics


def quat_to_mat(quaternions: mx.array) -> mx.array:
    """Quaternion (XYZW, scalar-last) to rotation matrix.

    Args:
        quaternions: (..., 4) in [i, j, k, r] order.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    i = quaternions[..., 0]
    j = quaternions[..., 1]
    k = quaternions[..., 2]
    r = quaternions[..., 3]
    two_s = 2.0 / mx.sum(quaternions * quaternions, axis=-1)
    o = mx.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return mx.reshape(o, (*quaternions.shape[:-1], 3, 3))


def mat_to_quat(matrix: mx.array) -> mx.array:
    """Rotation matrix to quaternion (XYZW, scalar-last).

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in [i, j, k, r] order.
    """
    batch_dim = matrix.shape[:-2]
    flat = mx.reshape(matrix, (*batch_dim, 9))
    m00 = flat[..., 0]
    m01 = flat[..., 1]
    m02 = flat[..., 2]
    m10 = flat[..., 3]
    m11 = flat[..., 4]
    m12 = flat[..., 5]
    m20 = flat[..., 6]
    m21 = flat[..., 7]
    m22 = flat[..., 8]

    q_abs = _sqrt_positive_part(
        mx.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    quat_by_rijk = mx.stack(
        [
            mx.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            mx.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            mx.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            mx.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    flr = mx.array(0.1)
    denom = 2.0 * mx.maximum(mx.expand_dims(q_abs, -1), flr)
    quat_candidates = quat_by_rijk / denom

    # Select best-conditioned quaternion
    best_idx = mx.argmax(q_abs, axis=-1)
    # Manual one-hot: compare each position with best_idx
    indices = mx.arange(4)
    # Broadcast: best_idx (...,1) vs indices (4,)
    one_hot = (mx.expand_dims(best_idx, -1) == indices).astype(q_abs.dtype)
    mask = one_hot > 0.5
    # Gather the selected quaternion for each batch element
    out = mx.sum(quat_candidates * mx.expand_dims(mask, -1), axis=-2)

    # Convert from [r, i, j, k] to [i, j, k, r]
    out = mx.concatenate([out[..., 1:2], out[..., 2:3], out[..., 3:4], out[..., 0:1]], axis=-1)
    out = standardize_quaternion(out)
    return out


def _sqrt_positive_part(x: mx.array) -> mx.array:
    """sqrt(max(0, x)) with zero subgradient at x=0."""
    return mx.sqrt(mx.maximum(x, 0.0))


def standardize_quaternion(quaternions: mx.array) -> mx.array:
    """Ensure the real (last) component of the quaternion is non-negative."""
    return mx.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def affine_inverse(A: mx.array) -> mx.array:
    """Efficient inverse of an affine (rigid-body) transformation matrix.

    Args:
        A: (..., 4, 4) affine matrix.

    Returns:
        (..., 4, 4) inverse matrix.
    """
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    Rt = mx.transpose(R, axes=list(range(R.ndim - 2)) + [-1, -2])
    return mx.concatenate(
        [mx.concatenate([Rt, -(Rt @ T)], axis=-1), P], axis=-2
    )
