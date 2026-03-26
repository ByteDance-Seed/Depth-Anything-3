#!/usr/bin/env python3
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from depth_anything_3.api import DepthAnything3
print("Import successful!")

print("Creating model...")
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
print("Model loaded!")
print("Done!")
