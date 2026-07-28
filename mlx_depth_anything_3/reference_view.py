# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Reference view selection strategies for multi-view depth estimation (MLX)."""

import mlx.core as mx


THRESH_FOR_REF_SELECTION = 3


def select_reference_view(
    x: mx.array,
    strategy: str = "saddle_balanced",
) -> mx.array:
    """Select a reference view from multiple views.

    Args:
        x: (B, S, N, C) feature tensor.
        strategy: "first", "middle", "saddle_balanced", or "saddle_sim_range".

    Returns:
        (B,) tensor of selected view indices.
    """
    B, S, N, C = x.shape
    if S <= 1:
        return mx.zeros((B,), dtype=mx.int32)
    if strategy == "first":
        return mx.zeros((B,), dtype=mx.int32)
    if strategy == "middle":
        return mx.full((B,), S // 2, dtype=mx.int32)

    # Feature-based strategies: normalized class tokens
    cls_feat = x[:, :, 0]  # (B, S, C)
    cls_norm = mx.sqrt(mx.sum(cls_feat * cls_feat, axis=-1, keepdims=True) + 1e-12)
    img_class_feat = cls_feat / cls_norm  # (B, S, C)

    if strategy == "saddle_balanced":
        sim = img_class_feat @ mx.transpose(img_class_feat, axes=(0, 2, 1))  # (B, S, S)
        eye = mx.broadcast_to(mx.expand_dims(mx.eye(S), 0), sim.shape)
        sim_no_diag = sim - eye
        sim_score = mx.sum(sim_no_diag, axis=-1) / (S - 1)

        feat_norm = mx.sqrt(mx.sum(cls_feat * cls_feat, axis=-1) + 1e-12)
        feat_var = mx.var(img_class_feat, axis=-1)

        def normalize_metric(m):
            mn = mx.min(m, axis=1, keepdims=True)
            mx_val = mx.max(m, axis=1, keepdims=True)
            return (m - mn) / (mx_val - mn + 1e-8)

        sim_n = normalize_metric(sim_score)
        norm_n = normalize_metric(feat_norm)
        var_n = normalize_metric(feat_var)

        balance = mx.abs(sim_n - 0.5) + mx.abs(norm_n - 0.5) + mx.abs(var_n - 0.5)
        return mx.argmin(balance, axis=1)

    elif strategy == "saddle_sim_range":
        sim = img_class_feat @ mx.transpose(img_class_feat, axes=(0, 2, 1))
        eye = mx.broadcast_to(mx.expand_dims(mx.eye(S), 0), sim.shape)
        sim_no_diag = sim - eye
        sim_max = mx.max(sim_no_diag, axis=-1)
        sim_min = mx.min(sim_no_diag, axis=-1)
        sim_range = sim_max - sim_min
        return mx.argmax(sim_range, axis=1)

    raise ValueError(f"Unknown strategy: {strategy}")


def reorder_by_reference(x: mx.array, b_idx: mx.array) -> mx.array:
    """Reorder views so that the reference view is at index 0.

    Args:
        x: (B, S, ...) tensor.
        b_idx: (B,) reference view indices.

    Returns:
        Reordered tensor.
    """
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x

    positions = mx.broadcast_to(mx.arange(S).reshape(1, S), (B, S))
    b_idx_expanded = mx.expand_dims(b_idx, 1)

    reorder_indices = mx.where(
        (positions > 0) & (positions <= b_idx_expanded),
        positions - 1,
        positions,
    )
    # Set position 0 to ref_idx
    col0 = mx.expand_dims(b_idx, 1)
    reorder_indices = mx.concatenate([col0, reorder_indices[:, 1:]], axis=1)
    # Fix: for positions > 0 and <= b_idx, we already set positions-1, which is correct
    # For position 0, we set b_idx, which is correct

    # Gather using take_along_axis
    # Expand indices to match x's extra dimensions
    idx_shape = list(reorder_indices.shape) + [1] * (x.ndim - 2)
    idx_expanded = mx.reshape(reorder_indices, idx_shape)
    idx_expanded = mx.broadcast_to(idx_expanded, list(reorder_indices.shape) + list(x.shape[2:]))
    return mx.take_along_axis(x, idx_expanded, axis=1)


def restore_original_order(x: mx.array, b_idx: mx.array) -> mx.array:
    """Restore original view order after reference-first reordering.

    Args:
        x: (B, S, ...) reordered tensor.
        b_idx: (B,) original reference view indices.

    Returns:
        Tensor with original order restored.
    """
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x

    positions = mx.broadcast_to(mx.arange(S).reshape(1, S), (B, S))
    b_idx_expanded = mx.expand_dims(b_idx, 1)

    restore_indices = mx.where(
        positions < b_idx_expanded,
        positions + 1,
        positions,
    )
    # Position = ref_idx comes from position 0
    # Use a mask to set the right position
    ref_mask = positions == b_idx_expanded
    restore_indices = mx.where(ref_mask, mx.zeros_like(restore_indices), restore_indices)

    # Gather using take_along_axis
    idx_shape = list(restore_indices.shape) + [1] * (x.ndim - 2)
    idx_expanded = mx.reshape(restore_indices, idx_shape)
    idx_expanded = mx.broadcast_to(idx_expanded, list(restore_indices.shape) + list(x.shape[2:]))
    return mx.take_along_axis(x, idx_expanded, axis=1)
