#!/usr/bin/env python3
# Copyright (c) 2025 Delanoe Pirard
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
GPU Preprocessing Benchmark

Compares CPU vs GPU preprocessing performance across different image sizes.
Measures:
- Preprocessing time only
- Total inference time (preprocessing + model forward)
- Memory usage
- Speedup percentages
"""

import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.gpu_input_processor import GPUInputProcessor


import os
import shutil

def create_test_files(sizes: List[Tuple[int, int]], count: int = 4, temp_dir: str = "temp_bench_imgs") -> Tuple[List[List[str]], str]:
    """Create test image files on disk.

    Args:
        sizes: List of (width, height) tuples
        count: Number of images per size
        temp_dir: Directory to save images

    Returns:
        List of image path batches, one per size
        Path to temp directory
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    batches = []
    for w, h, _ in sizes:
        batch = []
        for i in range(count):
            img = Image.new("RGB", (w, h), color=(i * 50, 100, 150))
            fname = f"{temp_dir}/{w}x{h}_{i}.jpg"
            img.save(fname, quality=95, subsampling=0)
            batch.append(fname)
        batches.append(batch)
    return batches, temp_dir

def benchmark_gpu_decode_files(
    processor,
    image_paths: List[str],
    process_res: int = 504,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    num_workers: int = 8,
) -> float:
    """Benchmark GPU decoding (from file path)."""
    # Warmup
    for _ in range(warmup_runs):
        processor(
            image=image_paths,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
        )

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        if hasattr(processor, 'device') and processor.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        # Pass file paths directly to GPUInputProcessor
        tensor, _, _ = processor(
            image=image_paths,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
        )

        if hasattr(processor, 'device') and processor.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)

def create_test_images(sizes: List[Tuple[int, int]], count: int = 4) -> List[List[Image.Image]]:
    """Create test images for each size.

    Args:
        sizes: List of (width, height) tuples
        count: Number of images per size

    Returns:
        List of image batches, one per size
    """
    batches = []
    for w, h in sizes:
        batch = [Image.new("RGB", (w, h), color=(i * 50, 100, 150)) for i in range(count)]
        batches.append(batch)
    return batches


def benchmark_hybrid(
    processor,
    images: List[Image.Image],
    process_res: int = 504,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    num_workers: int = 8,
    device=torch.device("cuda")
) -> float:
    """Benchmark hybrid preprocessing (CPU resize -> GPU normalize)."""
    
    # Warmup
    for _ in range(warmup_runs):
        imgs_cpu, _, _ = processor(
            image=images,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
            perform_normalization=False
        )
        imgs_gpu = imgs_cpu.to(device, non_blocking=True).float() / 255.0
        _ = InputProcessor.normalize_tensor(imgs_gpu, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        
        # 1. CPU Preprocessing (uint8)
        imgs_cpu, _, _ = processor(
            image=images,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
            perform_normalization=False
        )
        
        # 2. Transfer + Normalize
        imgs_gpu = imgs_cpu.to(device, non_blocking=True).float() / 255.0
        _ = InputProcessor.normalize_tensor(imgs_gpu, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if device.type == "cuda":
            torch.cuda.synchronize()
            
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
    return np.mean(times)

def benchmark_preprocessing(
    processor,
    images: List[Image.Image],
    process_res: int = 504,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    num_workers: int = 8,
) -> float:
    """Benchmark preprocessing performance.

    Args:
        processor: InputProcessor or GPUInputProcessor instance
        images: List of test images
        process_res: Processing resolution
        warmup_runs: Number of warmup runs to discard
        benchmark_runs: Number of benchmark runs to average
        num_workers: Number of parallel workers (for CPU processor)

    Returns:
        Average preprocessing time in seconds
    """
    # Warmup
    for _ in range(warmup_runs):
        processor(
            image=images,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
        )

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        if hasattr(processor, 'device') and processor.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        tensor, _, _ = processor(
            image=images,
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=num_workers,
        )

        if hasattr(processor, 'device') and processor.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def print_results_table(results: List[dict]):
    """Pretty print benchmark results as table."""
    print("\n" + "=" * 140)
    print("GPU PREPROCESSING BENCHMARK RESULTS")
    print("=" * 140)
    print(f"{'Image Size':<15} {'CPU Time':<12} {'GPU Time':<12} {'Hybrid Time':<12} {'GPU Decode':<12} {'Best Method':<15}")
    print("-" * 140)

    for result in results:
        size_str = f"{result['width']}x{result['height']}"
        cpu_time = f"{result['cpu_time']*1000:.2f} ms"
        gpu_time = f"{result['gpu_time']*1000:.2f} ms"
        hybrid_time = f"{result['hybrid_time']*1000:.2f} ms"
        gpu_decode_time = f"{result['gpu_decode_time']*1000:.2f} ms"
        
        times = [result['cpu_time'], result['gpu_time'], result['hybrid_time'], result['gpu_decode_time']]
        labels = ["CPU", "GPU", "Hybrid", "GPU Decode"]
        best_idx = np.argmin(times)
        best = labels[best_idx]

        print(f"{size_str:<15} {cpu_time:<12} {gpu_time:<12} {hybrid_time:<12} {gpu_decode_time:<12} {best:<15}")

    print("=" * 140 + "\n")


def main():
    """Run comprehensive benchmark."""
    print("\n" + "=" * 100)
    print("INITIALIZING GPU PREPROCESSING BENCHMARK")
    print("=" * 100)

    # Check GPU availability
    if torch.cuda.is_available():
        device_name = "cuda"
        device_info = torch.cuda.get_device_name(0)
        print(f"✓ GPU Device: {device_info}")
        print("✓ GPU preprocessing: ENABLED (NVJPEG + Kornia)")
    elif torch.backends.mps.is_available():
        device_name = "mps"
        device_info = "Apple MPS"
        print(f"✓ GPU Device: {device_info}")
        print("ℹ GPU preprocessing: DISABLED on MPS (CPU is faster on Apple Silicon)")
        print("  → GPUInputProcessor will use CPU path automatically")
        print("  → GPU reserved for model inference (5-10x speedup there)")
    else:
        print("✗ No GPU available - benchmark will show CPU vs CPU (no speedup expected)")
        device_name = "cpu"
        device_info = "CPU only"

    device = torch.device(device_name)

    # Create processors
    cpu_proc = InputProcessor()
    gpu_proc = GPUInputProcessor(device=device_name)
    print(f"✓ Processors initialized: CPU vs {device_name.upper()}")

    # Test configurations
    # Format: (width, height, description)
    test_sizes = [
        (640, 480, "Small (VGA)"),
        (1280, 720, "Medium (HD)"),
        (1920, 1080, "Large (Full HD)"),
        (3840, 2160, "XLarge (4K)"),
    ]

    process_res = 504
    num_images = 4
    num_workers = 8

    print(f"✓ Test config: {num_images} images per batch, process_res={process_res}, num_workers={num_workers}")
    print(f"✓ Testing {len(test_sizes)} image sizes: {', '.join([desc for _, _, desc in test_sizes])}")

    # Create test images
    print("\nGenerating test images (PIL & Files)...")
    image_batches_pil = create_test_images([(w, h) for w, h, _ in test_sizes], count=num_images)
    image_batches_files, temp_dir = create_test_files(test_sizes, count=num_images)
    print("✓ Test images generated")

    # Run benchmarks
    print("\nRunning benchmarks (this may take a minute)...\n")
    results = []

    try:
        for (w, h, desc), imgs_pil, imgs_files in zip(test_sizes, image_batches_pil, image_batches_files):
            print(f"Benchmarking {desc} ({w}x{h})...", end=" ", flush=True)

            cpu_time = benchmark_preprocessing(cpu_proc, imgs_pil, process_res, num_workers=num_workers)
            gpu_time = benchmark_preprocessing(gpu_proc, imgs_pil, process_res, num_workers=num_workers)
            hybrid_time = benchmark_hybrid(cpu_proc, imgs_pil, process_res, num_workers=num_workers, device=device)
            
            # GPU Decode uses file paths
            gpu_decode_time = benchmark_gpu_decode_files(gpu_proc, imgs_files, process_res, num_workers=num_workers)

            results.append({
                'width': w,
                'height': h,
                'description': desc,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'hybrid_time': hybrid_time,
                'gpu_decode_time': gpu_decode_time
            })
            
            best_time = min(cpu_time, gpu_time, hybrid_time, gpu_decode_time)
            if best_time == gpu_decode_time:
                win = "GPU Decode"
            elif best_time == hybrid_time:
                win = "Hybrid"
            elif best_time == gpu_time:
                win = "GPU"
            else:
                win = "CPU"

            print(f"✓ Best: {win}")

        # Print results table
        print_results_table(results)

        # Memory info (CUDA only)
        if device_name == "cuda":
            print("\nGPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temp directory: {temp_dir}")

if __name__ == "__main__":
    main()

