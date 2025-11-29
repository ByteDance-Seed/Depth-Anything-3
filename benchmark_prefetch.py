#!/usr/bin/env python3
"""
Benchmark prefetch pipeline vs vanilla sequential processing.

Compares:
- Vanilla: Load batch → Transfer → Compute → Repeat (sequential)
- Prefetch: Overlap I/O (load+transfer) with compute

Tests on MPS, CPU (and CUDA if available).
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import statistics
from typing import List, Iterator

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_anything_3.utils.prefetch_pipeline import create_pipeline


class DummyModel(nn.Module):
    """Dummy model to simulate inference workload."""

    def __init__(self, input_size=512, hidden_size=2048, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, input_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Simulate compute-heavy operation
        return self.net(x)


def create_batch_loader(num_batches: int, batch_size: int, input_size: int) -> Iterator[torch.Tensor]:
    """
    Generator that yields batches (simulates data loading from disk).

    Adds artificial I/O delay to simulate realistic file loading.
    """
    for _ in range(num_batches):
        # Simulate I/O delay (realistic for image loading + decode)
        time.sleep(0.01)  # 10ms per batch (typical JPEG decode time)

        # Generate random batch (simulates loaded data)
        batch = torch.randn(batch_size, input_size)
        yield batch


def benchmark_vanilla(
    model: nn.Module,
    device: torch.device,
    num_batches: int,
    batch_size: int,
    input_size: int,
    num_runs: int = 3,
) -> dict:
    """
    Benchmark vanilla sequential pipeline.

    Sequential: Load → Transfer → Compute → Repeat
    """
    print(f"\nBenchmarking VANILLA (sequential) on {device.type.upper()}...")

    times = []
    for run in range(num_runs):
        results = []
        batch_loader = create_batch_loader(num_batches, batch_size, input_size)

        start = time.perf_counter()
        for batch in batch_loader:
            # Transfer to device
            batch_device = batch.to(device)

            # Compute
            with torch.no_grad():
                output = model(batch_device)
                results.append(output.cpu())

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s ({len(results)} batches)")

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    throughput = num_batches / mean_time

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "throughput": throughput,
    }


def benchmark_prefetch(
    model: nn.Module,
    device: torch.device,
    num_batches: int,
    batch_size: int,
    input_size: int,
    prefetch_factor: int = 2,
    num_runs: int = 3,
) -> dict:
    """
    Benchmark prefetch pipeline.

    Overlaps: Load batch N+1 while computing batch N
    """
    print(f"\nBenchmarking PREFETCH (overlap) on {device.type.upper()}...")

    times = []
    all_metrics = []

    for run in range(num_runs):
        batch_loader = create_batch_loader(num_batches, batch_size, input_size)

        pipeline = create_pipeline(model, device, prefetch_factor=prefetch_factor)

        start = time.perf_counter()
        results = pipeline.run_inference(batch_loader)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        metrics = pipeline.get_metrics()
        all_metrics.append(metrics)

        print(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s ({len(results)} batches)")
        print(f"    Overlap efficiency: {metrics['overlap_efficiency']:.1%}")

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    throughput = num_batches / mean_time
    avg_overlap = statistics.mean([m['overlap_efficiency'] for m in all_metrics])

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "throughput": throughput,
        "overlap_efficiency": avg_overlap,
    }


def main():
    """Run benchmarks on all available devices."""
    print("="*80)
    print("PREFETCH PIPELINE BENCHMARK: Vanilla vs Prefetch")
    print("="*80)

    # Detect devices
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    devices.append(torch.device("cpu"))

    print(f"\nTesting on: {', '.join(d.type.upper() for d in devices)}")

    # Benchmark config
    num_batches = 50
    batch_size = 32
    input_size = 512
    num_runs = 3

    print(f"\nConfig: {num_batches} batches, batch_size={batch_size}, input_size={input_size}")
    print(f"Model: 4-layer MLP (512 → 2048 → 2048 → 2048 → 512)")
    print("I/O simulation: 10ms delay per batch (realistic JPEG decode)\n")

    all_results = {}

    for device in devices:
        print("\n" + "="*80)
        print(f"TESTING ON {device.type.upper()}")
        print("="*80)

        # Create model
        model = DummyModel(input_size=input_size, hidden_size=2048, num_layers=4)
        model = model.to(device)
        model.eval()

        # Warmup
        print("\nWarming up...")
        warmup_loader = create_batch_loader(5, batch_size, input_size)
        for batch in warmup_loader:
            with torch.no_grad():
                _ = model(batch.to(device))

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark vanilla
        vanilla_results = benchmark_vanilla(
            model, device, num_batches, batch_size, input_size, num_runs
        )

        # Benchmark prefetch
        prefetch_results = benchmark_prefetch(
            model, device, num_batches, batch_size, input_size,
            prefetch_factor=2, num_runs=num_runs
        )

        # Compute speedup
        speedup = vanilla_results["mean_time"] / prefetch_results["mean_time"]
        improvement_pct = ((vanilla_results["mean_time"] - prefetch_results["mean_time"])
                          / vanilla_results["mean_time"]) * 100

        all_results[device.type] = {
            "vanilla": vanilla_results,
            "prefetch": prefetch_results,
            "speedup": speedup,
            "improvement_pct": improvement_pct,
        }

        # Print summary
        print("\n" + "-"*80)
        print("RESULTS")
        print("-"*80)
        print(f"Vanilla:   {vanilla_results['mean_time']:.3f} ± {vanilla_results['std_time']:.3f}s")
        print(f"           {vanilla_results['throughput']:.1f} batches/s")
        print(f"\nPrefetch:  {prefetch_results['mean_time']:.3f} ± {prefetch_results['std_time']:.3f}s")
        print(f"           {prefetch_results['throughput']:.1f} batches/s")
        print(f"           Overlap efficiency: {prefetch_results['overlap_efficiency']:.1%}")
        print(f"\nSpeedup:   {speedup:.2f}x ({improvement_pct:+.1f}%)")
        print("-"*80)

    # Final summary
    print("\n\n" + "="*80)
    print("SUMMARY BY DEVICE")
    print("="*80)
    print(f"\n{'Device':<10} {'Vanilla (s)':<15} {'Prefetch (s)':<15} {'Speedup':<12} {'Gain':<10} {'Expected'}")
    print("-"*80)

    expected_gains = {
        "cuda": "15-25%",
        "mps": "10-15%",
        "cpu": "3-8%",
    }

    for device_type, results in all_results.items():
        vanilla = results["vanilla"]
        prefetch = results["prefetch"]
        speedup = results["speedup"]
        gain_pct = results["improvement_pct"]
        expected = expected_gains.get(device_type, "N/A")

        print(f"{device_type.upper():<10} "
              f"{vanilla['mean_time']:>6.3f} ± {vanilla['std_time']:<5.3f}  "
              f"{prefetch['mean_time']:>6.3f} ± {prefetch['std_time']:<5.3f}  "
              f"{speedup:>4.2f}x       "
              f"{gain_pct:>+6.1f}%   "
              f"{expected}")

    print("="*80)

    # Validation
    print("\nVALIDATION:")
    for device_type, results in all_results.items():
        gain = results["improvement_pct"]
        expected_range = expected_gains.get(device_type, "")

        if device_type == "cuda" and 15 <= gain <= 25:
            status = "✓ PASS"
        elif device_type == "mps" and 10 <= gain <= 15:
            status = "✓ PASS"
        elif device_type == "cpu" and 3 <= gain <= 8:
            status = "✓ PASS"
        else:
            status = "⚠ Outside expected range"

        print(f"  {device_type.upper()}: {gain:+.1f}% (expected {expected_range}) - {status}")


if __name__ == "__main__":
    main()
