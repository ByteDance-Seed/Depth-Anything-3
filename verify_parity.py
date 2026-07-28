#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Verify parity between PyTorch and MLX Depth Anything 3 models.

This script:
1. Creates both PyTorch and MLX models with matching configurations.
2. Generates identical random inputs (images, extrinsics, intrinsics).
3. Runs forward passes on both models.
4. Compares depth outputs and reports max/mean absolute differences.

Usage:
    python verify_parity.py --model da3-small [--weights pytorch_weights.safetensors]

If --weights is provided, it loads the PyTorch weights, converts them to MLX,
and runs both models with real weights.  Without --weights, it initialises
both models from scratch with identical random weights (useful for checking
the architecture mapping).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ---- helpers ---------------------------------------------------------------

def make_random_input(
    num_views: int = 2,
    height: int = 504,
    width: int = 504,
    seed: int = 42,
    with_cameras: bool = True,
):
    """Generate deterministic random inputs for both frameworks."""
    rng = np.random.RandomState(seed)
    images = rng.randn(1, num_views, height, width, 3).astype(np.float32)

    extrinsics = None
    intrinsics = None
    if with_cameras:
        # Simple extrinsics: identity + small perturbation
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (1, num_views, 1, 1))
        for i in range(num_views):
            extrinsics[0, i, :3, 3] = rng.randn(3) * 0.1
        # Simple intrinsics
        fx = fy = 500.0
        cx, cy = width / 2, height / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics = np.tile(K, (1, num_views, 1, 1))
    return images, extrinsics, intrinsics


# ---- PyTorch forward -------------------------------------------------------

def run_pytorch(
    images_np: np.ndarray,
    extrinsics_np: np.ndarray | None,
    intrinsics_np: np.ndarray | None,
    config_name: str,
    weights_path: str | None,
) -> np.ndarray:
    """Run PyTorch model and return depth as numpy.

    images_np: (1, N, H, W, 3) NHWC float32.
    Returns: (N, H, W) depth.
    """
    import torch
    from omegaconf import OmegaConf

    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.geometry import affine_inverse

    cfg_path = f"src/depth_anything_3/configs/{config_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    model = create_object(cfg)
    model.eval()

    if weights_path is not None:
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        
        # Strip 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"[PyTorch] Loaded weights from {weights_path}")

    # Convert NHWC -> NCHW for PyTorch
    # images_np: (1, N, H, W, 3) -> torch (1, N, 3, H, W)
    images_pt = torch.from_numpy(images_np).permute(0, 1, 4, 2, 3).float()

    ext_pt = None
    int_pt = None
    if extrinsics_np is not None:
        ext_pt = torch.from_numpy(extrinsics_np).float()
    if intrinsics_np is not None:
        int_pt = torch.from_numpy(intrinsics_np).float()

    with torch.no_grad():
        output = model(images_pt, extrinsics=ext_pt, intrinsics=int_pt,
                       export_feat_layers=[], infer_gs=False, use_ray_pose=False)

    depth = output["depth"].numpy()  # (1, N, H, W)
    return depth[0]  # (N, H, W)


# ---- MLX forward -----------------------------------------------------------

def run_mlx(
    images_np: np.ndarray,
    extrinsics_np: np.ndarray | None,
    intrinsics_np: np.ndarray | None,
    config_name: str,
    weights_path: str | None,
) -> np.ndarray:
    """Run MLX model and return depth as numpy.

    images_np: (1, N, H, W, 3) NHWC float32.
    Returns: (N, H, W) depth.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_depth_anything_3.model import DepthAnything3Net

    model = DepthAnything3Net(config_name)

    if weights_path is not None:
        from convert_to_mlx import convert_weights, load_pytorch_weights
        pt_weights = load_pytorch_weights(weights_path)
        mlx_weights = convert_weights(pt_weights)
        
        def flatten_params(params, prefix=""):
            flat = {}
            for k, v in params.items():
                name = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flat.update(flatten_params(v, name))
                else:
                    flat[name] = v
            return flat
            
        model_params = flatten_params(model.parameters())
        filtered_weights = {k: v for k, v in mlx_weights.items() if k in model_params}
        
        # Load the filtered weights
        model.load_weights(list(filtered_weights.items()), strict=False)
        mx.eval(model.parameters())
        print(f"[MLX] Loaded and converted {len(filtered_weights)} weights from {weights_path}")

    x = mx.array(images_np)
    ext_mx = mx.array(extrinsics_np) if extrinsics_np is not None else None
    int_mx = mx.array(intrinsics_np) if intrinsics_np is not None else None

    output = model(x, extrinsics=ext_mx, intrinsics=int_mx)
    mx.eval(output["depth"])

    depth = np.array(output["depth"][0])  # (N, H, W)
    return depth


# ---- Comparison ------------------------------------------------------------

def compare(pt_depth: np.ndarray, mlx_depth: np.ndarray, tol: float = 1e-3):
    """Compare PyTorch and MLX depth outputs."""
    assert pt_depth.shape == mlx_depth.shape, (
        f"Shape mismatch: PyTorch {pt_depth.shape} vs MLX {mlx_depth.shape}"
    )
    diff = np.abs(pt_depth - mlx_depth)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_diff = diff / (np.abs(pt_depth) + 1e-8)
    max_rel = rel_diff.max()
    mean_rel = rel_diff.mean()

    print(f"\n{'='*60}")
    print(f"Parity Verification Results")
    print(f"{'='*60}")
    print(f"  Output shape:    {pt_depth.shape}")
    print(f"  Absolute error:")
    print(f"    Max:           {max_diff:.6e}")
    print(f"    Mean:          {mean_diff:.6e}")
    print(f"  Relative error:")
    print(f"    Max:           {max_rel:.6e}")
    print(f"    Mean:          {mean_rel:.6e}")
    print(f"  PyTorch range:   [{pt_depth.min():.4f}, {pt_depth.max():.4f}]")
    print(f"  MLX range:       [{mlx_depth.min():.4f}, {mlx_depth.max():.4f}]")

    passed = max_diff < tol
    print(f"\n  Status:          {'PASS ✓' if passed else 'FAIL ✗'} (tol={tol})")
    print(f"{'='*60}\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify DA3 PyTorch/MLX parity")
    parser.add_argument("--model", type=str, default="da3-small",
                        help="Model config name")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to PyTorch weights file")
    parser.add_argument("--num-views", type=int, default=2,
                        help="Number of input views")
    parser.add_argument("--height", type=int, default=504,
                        help="Image height")
    parser.add_argument("--width", type=int, default=504,
                        help="Image width")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Disable camera conditioning")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Tolerance for parity check")
    args = parser.parse_args()

    print(f"Generating random inputs: {args.num_views} views, "
          f"{args.height}x{args.width}, cameras={'off' if args.no_cameras else 'on'}")
    images, extrinsics, intrinsics = make_random_input(
        num_views=args.num_views,
        height=args.height,
        width=args.width,
        with_cameras=not args.no_cameras,
    )

    print(f"\nRunning PyTorch model ({args.model})...")
    pt_depth = run_pytorch(images, extrinsics, intrinsics, args.model, args.weights)

    print(f"Running MLX model ({args.model})...")
    mlx_depth = run_mlx(images, extrinsics, intrinsics, args.model, args.weights)

    passed = compare(pt_depth, mlx_depth, tol=args.tol)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
