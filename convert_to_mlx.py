#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Convert PyTorch Depth Anything 3 weights to MLX format.

Usage:
    python convert_to_mlx.py --model da3-small --input weights.safetensors --output mlx_weights.safetensors

This script:
1. Loads PyTorch weights (safetensors or .pth).
2. Maps parameter names from PyTorch model structure to MLX model structure.
3. Transposes Conv2d weights from PyTorch (OIHW) to MLX (OHWI) format.
4. Saves the result as a safetensors file loadable by mlx.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_pytorch_weights(path: str) -> dict[str, np.ndarray]:
    """Load weights from a PyTorch safetensors or .pth file."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        weights = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights
    else:
        import torch
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        return {k: v.numpy() for k, v in state_dict.items()}


def map_key(pt_key: str) -> str | None:
    """Map a PyTorch state_dict key to the corresponding MLX model key.

    Returns None if the key should be skipped (e.g., GS-related weights).
    """
    k = pt_key

    # Skip GS-related weights
    if "gs_head" in k or "gs_adapter" in k:
        return None

    # ---- Backbone (DinoV2) ----
    # backbone.pretrained.X -> backbone.pretrained.X
    k = k.replace("backbone.pretrained.", "backbone.pretrained.")

    # blocks.N.attn -> blocks.N.attn
    # blocks.N.mlp -> blocks.N.mlp
    # blocks.N.norm1 -> blocks.N.norm1
    # blocks.N.ls1.gamma -> blocks.N.ls1.gamma
    # blocks.N.ls2.gamma -> blocks.N.ls2.gamma

    # ---- Head (DualDPT) ----
    # head.norm -> head.norm
    # head.projects.N -> head.projects.N
    # head.resize_layers.0 -> head.resize_0
    # head.resize_layers.1 -> head.resize_1
    # head.resize_layers.3 -> head.resize_3
    k = re.sub(r"head\.resize_layers\.0\.", "head.resize_0.", k)
    k = re.sub(r"head\.resize_layers\.1\.", "head.resize_1.", k)
    k = re.sub(r"head\.resize_layers\.3\.", "head.resize_3.", k)
    # Skip resize_layers.2 (Identity)
    if "head.resize_layers.2" in k:
        return None

    # head.scratch.layer1_rn -> head.layer1_rn
    k = k.replace("head.scratch.", "head.")

    # head.refinenet1.resConfUnit1.conv1 -> head.refinenet1.resConfUnit1.conv1
    # head.output_conv1 -> head.output_conv1

    # head.output_conv2.0 -> head.output_conv2_a
    # head.output_conv2.2 -> head.output_conv2_b
    k = re.sub(r"head\.output_conv2\.0\.", "head.output_conv2_a.", k)
    k = re.sub(r"head\.output_conv2\.2\.", "head.output_conv2_b.", k)
    # Skip output_conv2.1 (ReLU, no params)
    if re.search(r"head\.output_conv2\.\d+\.", k) and "output_conv2_a" not in k and "output_conv2_b" not in k:
        return None

    # Aux head: output_conv1_aux.N.M -> output_conv1_aux.N.M (list of lists)
    # head.output_conv1_aux.0.0 -> head.output_conv1_aux.0.layers.0
    k = re.sub(
        r"head\.output_conv1_aux\.(\d+)\.(\d+)\.",
        r"head.output_conv1_aux.\1.\2.",
        k,
    )

    # head.output_conv2_aux.N.0 -> head.output_conv2_aux.N.conv1
    # head.output_conv2_aux.N.1 -> (Permute, skip)
    # head.output_conv2_aux.N.2 -> head.output_conv2_aux.N.ln
    # head.output_conv2_aux.N.3 -> (ReLU, skip)
    # head.output_conv2_aux.N.4 -> head.output_conv2_aux.N.conv2
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.0\.(.*)", k)
    if m:
        k = f"head.output_conv2_aux.{m.group(1)}.conv1.{m.group(2)}"
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.2\.(.*)", k)
    if m:
        k = f"head.output_conv2_aux.{m.group(1)}.ln.{m.group(2)}"
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.4\.(.*)", k)
    if m:
        k = f"head.output_conv2_aux.{m.group(1)}.conv2.{m.group(2)}"
    # Skip Permute (indices 1, 3) and ReLU
    if re.match(r"head\.output_conv2_aux\.\d+\.[13]\.", k):
        return None

    # ---- Camera Encoder ----
    # cam_enc.trunk.N -> cam_enc.trunk.N
    # cam_enc.pose_branch.fc1 -> cam_enc.pose_branch.fc1
    # cam_enc.token_norm -> cam_enc.token_norm
    # cam_enc.trunk_norm -> cam_enc.trunk_norm

    # ---- Camera Decoder ----
    # cam_dec.backbone.0 -> cam_dec.backbone_fc1
    # cam_dec.backbone.2 -> cam_dec.backbone_fc2
    k = re.sub(r"cam_dec\.backbone\.0\.", "cam_dec.backbone_fc1.", k)
    k = re.sub(r"cam_dec\.backbone\.2\.", "cam_dec.backbone_fc2.", k)
    # Skip indices 1,3 (ReLU, no params)
    if re.match(r"cam_dec\.backbone\.\d+\.", k):
        return None

    # cam_dec.fc_fov.0 -> cam_dec.fc_fov_linear
    k = re.sub(r"cam_dec\.fc_fov\.0\.", "cam_dec.fc_fov_linear.", k)
    # Skip fc_fov.1 (ReLU)
    if re.match(r"cam_dec\.fc_fov\.\d+\.", k):
        return None

    # Handle skip_add (FloatFunctional) - no parameters
    if "skip_add" in k:
        return None

    return k


def is_conv_weight(key: str, shape: tuple) -> bool:
    """Check if a weight tensor is a Conv2d/ConvTranspose2d weight that needs transposing.

    PyTorch conv weights: (out_ch, in_ch, kH, kW) -> 4D with spatial dims
    MLX conv weights: (out_ch, kH, kW, in_ch)
    """
    if len(shape) != 4:
        return False
    # Check if it's a weight (not bias) in a conv-like layer
    if key.endswith(".weight"):
        # Identify conv layers by name patterns
        conv_patterns = [
            "proj.weight",  # patch embed
            "conv1.weight", "conv2.weight",  # residual conv units
            "out_conv.weight",  # fusion out conv
            "layer1_rn.weight", "layer2_rn.weight",
            "layer3_rn.weight", "layer4_rn.weight",
            "output_conv",  # output convolutions
            "resize_0.weight", "resize_1.weight", "resize_3.weight",
            "projects.",  # per-stage projections
        ]
        return any(p in key for p in conv_patterns)
    return False


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose conv weight from PyTorch (OIHW) to MLX (OHWI)."""
    return np.ascontiguousarray(weight.transpose(0, 2, 3, 1))


def is_conv_transpose_weight(key: str) -> bool:
    """Check if this is a ConvTranspose2d weight."""
    return "resize_0.weight" in key or "resize_1.weight" in key


def transpose_conv_transpose_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose ConvTranspose2d weight from PyTorch (I,O,kH,kW) to MLX (O,kH,kW,I)."""
    return np.ascontiguousarray(weight.transpose(1, 2, 3, 0))


def convert_weights(
    pt_weights: dict[str, np.ndarray],
) -> dict[str, mx.array]:
    """Convert PyTorch weights to MLX format.

    Returns:
        dict mapping MLX key paths to mx.array values.
    """
    mlx_weights = {}
    skipped = []
    for pt_key, value in pt_weights.items():
        mlx_key = map_key(pt_key)
        if mlx_key is None:
            skipped.append(pt_key)
            continue

        arr = value
        if is_conv_transpose_weight(mlx_key):
            arr = transpose_conv_transpose_weight(arr)
        elif is_conv_weight(mlx_key, arr.shape):
            arr = transpose_conv_weight(arr)

        mlx_weights[mlx_key] = mx.array(arr)

    if skipped:
        print(f"Skipped {len(skipped)} keys (GS/unused):")
        for s in skipped[:10]:
            print(f"  {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(description="Convert DA3 PyTorch weights to MLX")
    parser.add_argument("--model", type=str, default="da3-small",
                        choices=["da3-small", "da3-base", "da3-large", "da3-giant"],
                        help="Model configuration name")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to PyTorch weights (.safetensors or .pth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for MLX weights (.safetensors or .npz)")
    args = parser.parse_args()

    print(f"Loading PyTorch weights from {args.input}...")
    pt_weights = load_pytorch_weights(args.input)
    print(f"  Loaded {len(pt_weights)} parameters")

    print("Converting weights...")
    mlx_weights = convert_weights(pt_weights)
    print(f"  Converted {len(mlx_weights)} parameters")

    print(f"Saving MLX weights to {args.output}...")
    if args.output.endswith(".safetensors"):
        mx.save_safetensors(args.output, mlx_weights)
    elif args.output.endswith(".npz"):
        mx.savez(args.output, **mlx_weights)
    else:
        raise ValueError(f"Unsupported output format: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
