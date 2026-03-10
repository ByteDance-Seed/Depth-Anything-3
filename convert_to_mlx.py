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
                new_key = key
                if key.startswith("model."):
                    new_key = key[6:]
                weights[new_key] = f.get_tensor(key)
        return weights
    else:
        import torch
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Also handle cases where keys in state_dict["model"] still have "model." prefix
        new_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[6:]
            new_dict[new_key] = v.numpy() if hasattr(v, 'numpy') else v
        return new_dict


def map_key(pt_key: str) -> str | None:
    """Map a PyTorch state_dict key to the corresponding MLX model key.

    Returns None if the key should be skipped (e.g., GS-related weights).
    """
    k = pt_key

    # Skip GS-related and Aux weights (not used in MLX forward)
    if any(p in k for p in ["gs_head", "gs_adapter", "aux", "refinenet", "layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn", "output_conv"]):
        # Wait, some of these might be used in the main head!
        # refinenet1..4 are used. refinenetX_aux are NOT.
        # output_conv1, output_conv2_a/b are used. output_convX_aux are NOT.
        if "_aux" in k:
            return None
        if "gs_head" in k or "gs_adapter" in k:
            return None

    # ---- Backbone (DinoV2) ----
    # backbone.pretrained.X -> backbone.X
    k = k.replace("backbone.pretrained.", "backbone.")

    # Bulk rename for Swift compatibility (snake_case -> camelCase)
    for snake, camel in [
        ("cls_token", "clsToken"), ("pos_embed", "posEmbed"),
        ("patch_embed", "patchEmbed"), ("camera_token", "cameraToken"),
        ("layer1_rn", "layer1Rn"), ("layer2_rn", "layer2Rn"),
        ("layer3_rn", "layer3Rn"), ("layer4_rn", "layer4Rn"),
        ("output_conv1", "outputConv1"), ("output_conv2_a", "outputConv2a"),
        ("output_conv2_b", "outputConv2b"),
        # New additions for missing MLX Swift layers:
        ("out_conv", "outConv"),
        (".mlp.", ".mlpLayer."),
        (".swiglu.", ".swiGluLayer."),
        ("q_norm", "qNorm"), 
        ("k_norm", "kNorm"),
        ("cam_enc", "camEnc"), 
        ("pose_branch", "poseBranch"),
        ("token_norm", "tokenNorm"), 
        ("trunk_norm", "trunkNorm"),
        ("cam_dec", "camDec"),
        ("backbone_fc", "backboneFc"),
        ("fc_t", "fcT"),
        ("fc_qvec", "fcQvec"),
        ("fc_fov_linear", "fcFovLinear"),
    ]:
        k = k.replace(snake, camel)

    # ---- Head (DualDPT) ----
    k = re.sub(r"head\.resize_layers\.0\.", "head.resize0.", k)
    k = re.sub(r"head\.resize_layers\.1\.", "head.resize1.", k)
    k = re.sub(r"head\.resize_layers\.3\.", "head.resize3.", k)
    
    # head.scratch.output_conv -> head.output_conv
    k = k.replace("head.scratch.", "head.")

    # head.refinenet1.resConfUnit1.conv1 -> head.refinenet1.resConfUnit1.conv1
    # head.output_conv1 -> head.output_conv1

    # head.output_conv2.0 -> head.outputConv2a
    # head.output_conv2.2 -> head.outputConv2b
    k = re.sub(r"head\.output_conv2\.0\.", "head.outputConv2a.", k)
    k = re.sub(r"head\.output_conv2\.2\.", "head.outputConv2b.", k)
    # Skip output_conv2.1 (ReLU, no params)
    if re.search(r"head\.output_conv2\.\d+\.", k) and "outputConv2a" not in k and "outputConv2b" not in k:
        return None

    # Aux head: output_conv1_aux.N.M -> head.output_conv1_aux.N.layers.M
    k = re.sub(
        r"head\.output_conv1_aux\.(\d+)\.(\d+)\.",
        r"head.output_conv1_aux.\1.layers.\2.",
        k,
    )

    # head.output_conv2_aux.N.0 -> head.output_conv2_aux.N.conv1
    # head.output_conv2_aux.N.1 -> (Permute, skip)
    # head.output_conv2_aux.N.2 -> head.output_conv2_aux.N.ln
    # head.output_conv2_aux.N.3 -> (ReLU, skip)
    # head.output_conv2_aux.N.4 -> (Identity, skip)
    # head.output_conv2_aux.N.5 -> head.output_conv2_aux.N.conv2
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.0\.(.*)", k)
    if m:
        k = f"head.output_conv2_aux.{m.group(1)}.conv1.{m.group(2)}"
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.2\.(.*)", k)
    if m:
        k = f"head.output_conv2_aux.{m.group(1)}.ln.{m.group(2)}"
    m = re.match(r"head\.output_conv2_aux\.(\d+)\.[45]\.(.*)", k)
    if m:
        # Both 4 and 5 in PyTorch often map to the final conv in different configs
        k = f"head.output_conv2_aux.{m.group(1)}.conv2.{m.group(2)}"
    
    # Skip Permute, ReLU, and optional Identity (indices 1, 3, 4)
    if re.match(r"head\.output_conv2_aux\.\d+\.[134]\.", k) and "conv2" not in k:
        return None

    # ---- Camera Encoder ----
    # cam_enc.trunk.N -> cam_enc.trunk.N
    # cam_enc.pose_branch.fc1 -> cam_enc.pose_branch.fc1
    # cam_enc.token_norm -> cam_enc.token_norm
    # cam_enc.trunk_norm -> cam_enc.trunk_norm

    # ---- Camera Decoder ----
    # camDec.backbone.0 -> camDec.backboneFc1
    # camDec.backbone.2 -> camDec.backboneFc2
    k = re.sub(r"camDec\.backbone\.0\.", "camDec.backboneFc1.", k)
    k = re.sub(r"camDec\.backbone\.2\.", "camDec.backboneFc2.", k)
    # Skip indices 1,3 (ReLU, no params)
    if re.match(r"camDec\.backbone\.\d+\.", k):
        return None
    
    # camDec.fc_fov.0 -> camDec.fcFovLinear
    k = re.sub(r"camDec\.fc_fov\.0\.", "camDec.fcFovLinear.", k)
    # Skip fc_fov.1 (ReLU)
    if re.match(r"camDec\.fc_fov\.\d+\.", k):
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
        # Identify conv layers by name patterns (using Swift mapped names)
        conv_patterns = [
            "patchEmbed.proj.weight",
            "conv1.weight", "conv2.weight", 
            "outConv.weight",
            "layer1Rn.weight", "layer2Rn.weight",
            "layer3Rn.weight", "layer4Rn.weight",
            "outputConv",
            "resize0.weight", "resize1.weight", "resize3.weight",
            "projects.",
        ]
        return any(p in key for p in conv_patterns)
    return False


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose conv weight from PyTorch (OIHW) to MLX (OHWI)."""
    return np.ascontiguousarray(weight.transpose(0, 2, 3, 1))


def is_conv_transpose_weight(key: str) -> bool:
    """Check if this is a ConvTranspose2d weight."""
    return "resize0.weight" in key or "resize1.weight" in key


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
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Output data type for MLX weights")
    args = parser.parse_args()

    print(f"Loading PyTorch weights from {args.input}...")
    pt_weights = load_pytorch_weights(args.input)
    print(f"  Loaded {len(pt_weights)} parameters")

    print(f"Converting weights to {args.dtype}...")
    mlx_weights = convert_weights(pt_weights)
    
    # Cast weights to requested dtype
    dtype = getattr(mx, args.dtype)
    mlx_weights = {k: v.astype(dtype) for k, v in mlx_weights.items()}
    
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
