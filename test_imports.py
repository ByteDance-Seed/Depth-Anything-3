#!/usr/bin/env python3
"""Simple test to verify all imports work correctly."""

print("Testing imports...")
print("-" * 60)

try:
    import os
    print("✓ os")
except Exception as e:
    print(f"✗ os: {e}")

try:
    import numpy as np
    print(f"✓ numpy (version: {np.__version__})")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"✓ matplotlib (version: {matplotlib.__version__})")
except Exception as e:
    print(f"✗ matplotlib: {e}")
    import traceback
    traceback.print_exc()

try:
    from PIL import Image
    print(f"✓ PIL (Pillow)")
except Exception as e:
    print(f"✗ PIL: {e}")

try:
    import torch
    print(f"✓ torch (version: {torch.__version__})")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    from depth_anything_3.api import DepthAnything3
    print("✓ depth_anything_3.api.DepthAnything3")
except Exception as e:
    print(f"✗ depth_anything_3.api: {e}")
    import traceback
    traceback.print_exc()

try:
    from depth_anything_3.utils.visualize import visualize_depth
    print("✓ depth_anything_3.utils.visualize.visualize_depth")
except Exception as e:
    print(f"✗ depth_anything_3.utils.visualize: {e}")

print("-" * 60)
print("All import tests completed!")
