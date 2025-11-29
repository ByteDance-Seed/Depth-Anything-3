#!/usr/bin/env python3
"""
Performance Tuning Example

This example demonstrates how to optimize performance for different platforms and use cases.

Usage:
    python examples/performance_tuning.py  # Auto-detect and benchmark
    python examples/performance_tuning.py --mixed-precision fp16
    python examples/performance_tuning.py --batch-size 2 --benchmark
    python examples/performance_tuning.py --profile  # Show all configurations
"""

import argparse
import time
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


def detect_platform():
    """Detect the current platform and return optimal settings."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "recommended_precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            "recommended_compile": True,
            "recommended_batch_size": 8,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "device": "mps",
            "device_name": "Apple Silicon (MPS)",
            "memory_gb": None,  # Unified memory
            "recommended_precision": "float32",  # More stable on MPS
            "recommended_compile": False,  # Slower on MPS
            "recommended_batch_size": 4,
        }
    else:
        return {
            "device": "cpu",
            "device_name": "CPU",
            "memory_gb": None,
            "recommended_precision": "float16",
            "recommended_compile": False,
            "recommended_batch_size": 2,
        }


def benchmark_config(model, device, num_images=10):
    """Benchmark a specific model configuration."""
    import glob

    # Use example images
    image_paths = sorted(glob.glob("assets/examples/SOH/*.png"))[:num_images]

    if not image_paths:
        print("Warning: No example images found, using dummy data")
        # Create dummy images
        import numpy as np

        image_paths = []
        for i in range(num_images):
            dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            from PIL import Image

            path = f"/tmp/dummy_{i}.png"
            Image.fromarray(dummy).save(path)
            image_paths.append(path)

    # Warmup
    _ = model.inference(image_paths[:2])

    # Benchmark
    start = time.time()
    prediction = model.inference(image_paths)
    elapsed = time.time() - start

    return {
        "num_images": len(image_paths),
        "total_time": elapsed,
        "images_per_sec": len(image_paths) / elapsed,
        "time_per_image": elapsed / len(image_paths),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Performance tuning example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["auto", "fp16", "fp32", "bf16"],
        help="Override mixed precision mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Force enable torch.compile()",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Force disable torch.compile()",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show all configuration options",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="da3-small",
        choices=["da3-small", "da3-base", "da3-large"],
        help="Model size",
    )
    args = parser.parse_args()

    # Detect platform
    platform = detect_platform()

    print("=" * 60)
    print("PERFORMANCE TUNING GUIDE")
    print("=" * 60)
    print(f"\nDetected Platform:")
    print(f"  Device: {platform['device']}")
    print(f"  Name: {platform['device_name']}")
    if platform["memory_gb"]:
        print(f"  Memory: {platform['memory_gb']:.1f} GB")

    print(f"\nRecommended Settings:")
    print(f"  Mixed Precision: {platform['recommended_precision']}")
    print(f"  torch.compile(): {platform['recommended_compile']}")
    print(f"  Batch Size: {platform['recommended_batch_size']}")

    if args.profile:
        print("\n" + "=" * 60)
        print("CONFIGURATION OPTIONS")
        print("=" * 60)

        print("\n1. MIXED PRECISION")
        print("   - auto: Platform-aware (recommended)")
        print("   - fp32: Maximum accuracy, slower")
        print("   - fp16: 2x faster, half memory")
        print("   - bf16: Better than fp16 (CUDA Ampere+)")

        print("\n2. BATCH SIZE")
        print("   - Higher: Faster throughput")
        print("   - Lower: Less memory usage")
        print("   Platform recommendations:")
        print("     • CUDA (16GB+): 8-16")
        print("     • CUDA (8GB): 4-8")
        print("     • MPS (8GB): 2-4")
        print("     • CPU: 1-2")

        print("\n3. torch.compile()")
        print("   - CUDA: Enable (30-50% faster)")
        print("   - MPS: Disable (2x slower if enabled)")
        print("   - CPU: Disable (minimal benefit)")

        print("\n4. MODEL SIZE")
        print("   - da3-small (80M): Fastest, good quality")
        print("   - da3-base (120M): Balanced")
        print("   - da3-large (350M): Best quality, slower")

        return

    # Determine configuration
    mixed_precision = args.mixed_precision or platform["recommended_precision"]
    batch_size = args.batch_size or platform["recommended_batch_size"]

    if args.enable_compile:
        enable_compile = True
    elif args.disable_compile:
        enable_compile = False
    else:
        enable_compile = platform["recommended_compile"]

    print(f"\nUsing Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  Batch Size: {batch_size}")
    print(f"  torch.compile(): {enable_compile}")

    # Load model
    print(f"\nLoading model...")
    model_name_map = {
        "da3-small": "depth-anything/DA3-SMALL",
        "da3-base": "depth-anything/DA3-BASE",
        "da3-large": "depth-anything/DA3-LARGE",
    }

    model = DepthAnything3.from_pretrained(
        model_name_map[args.model],
        batch_size=batch_size,
        mixed_precision=None if mixed_precision == "auto" else mixed_precision,
        enable_compile=enable_compile,
    )
    model = model.to(platform["device"])
    print("Model loaded!")

    if args.benchmark:
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK")
        print("=" * 60)

        results = benchmark_config(model, platform["device"], num_images=20)

        print(f"\nResults:")
        print(f"  Images processed: {results['num_images']}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Speed: {results['images_per_sec']:.2f} images/sec")
        print(f"  Per image: {results['time_per_image']*1000:.0f}ms")

        # Performance rating
        ips = results["images_per_sec"]
        if platform["device"] == "cuda":
            if ips > 50:
                rating = "🚀 Excellent"
            elif ips > 30:
                rating = "✅ Good"
            elif ips > 15:
                rating = "⚠️  Fair"
            else:
                rating = "❌ Slow"
        elif platform["device"] == "mps":
            if ips > 25:
                rating = "🚀 Excellent"
            elif ips > 15:
                rating = "✅ Good"
            elif ips > 8:
                rating = "⚠️  Fair"
            else:
                rating = "❌ Slow"
        else:  # CPU
            if ips > 5:
                rating = "✅ Good (for CPU)"
            elif ips > 2:
                rating = "⚠️  Fair"
            else:
                rating = "❌ Slow"

        print(f"\nPerformance: {rating}")

        # Suggestions
        if ips < 10 and platform["device"] == "cuda":
            print("\n💡 Suggestions to improve performance:")
            print("   • Try smaller model: --model da3-small")
            print("   • Enable compilation: --enable-compile")
            print("   • Use fp16: --mixed-precision fp16")
            print("   • Increase batch size: --batch-size 8")

        elif ips < 15 and platform["device"] == "mps":
            print("\n💡 Suggestions to improve performance:")
            print("   • Ensure compile is disabled (default)")
            print("   • Try fp16: --mixed-precision fp16")
            print("   • Use smaller model: --model da3-small")

    else:
        print("\n✅ Configuration loaded successfully!")
        print("   Run with --benchmark to test performance")
        print("   Run with --profile to see all options")


if __name__ == "__main__":
    main()
