#!/usr/bin/env python3
"""
Benchmark script to compare optimized attention implementation vs upstream.

Compares:
- Optimized version (awesome-depth-anything-3) with fused softmax
- Upstream version (depth-anything-3-upstream) without optimizations

Tests on MPS (Apple Silicon) primarily, but works on CUDA/CPU too.
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
import statistics

# Add both repos to path
OPTIMIZED_PATH = Path(__file__).parent / "src"
UPSTREAM_PATH = Path(__file__).parent.parent / "depth-anything-3-upstream" / "src"

sys.path.insert(0, str(OPTIMIZED_PATH))
sys.path.insert(1, str(UPSTREAM_PATH))


def benchmark_attention(device_type="mps", batch_size=2, seq_len=1024, num_heads=8, head_dim=64, num_runs=50, warmup=10):
    """
    Benchmark attention implementations.

    Args:
        device_type: 'mps', 'cuda', or 'cpu'
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    """
    device = torch.device(device_type)
    print(f"\n{'='*80}")
    print(f"Benchmarking on {device_type.upper()}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
    print(f"Runs: {num_runs} (warmup: {warmup})")
    print(f"{'='*80}\n")

    # Create test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    scale = head_dim ** -0.5

    # Import optimized version
    from depth_anything_3.model.optimized_attention import (
        scaled_dot_product_attention_optimized as optimized_attn
    )

    # Import upstream version (manual implementation from block.py or fallback)
    def upstream_attention(q, k, v, scale):
        """Upstream manual attention (non-optimized)."""
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        _ = optimized_attn(q, k, v, scale=scale)
        _ = upstream_attention(q, k, v, scale)

    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()

    # Benchmark optimized version
    print("Benchmarking OPTIMIZED version (fused softmax)...")
    optimized_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out_opt = optimized_attn(q, k, v, scale=scale)
        if device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        optimized_times.append(time.perf_counter() - start)

    # Benchmark upstream version
    print("Benchmarking UPSTREAM version (non-fused)...")
    upstream_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out_up = upstream_attention(q, k, v, scale)
        if device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        upstream_times.append(time.perf_counter() - start)

    # Verify outputs are close
    max_diff = torch.max(torch.abs(out_opt - out_up)).item()
    print(f"\nMax output difference: {max_diff:.2e}")
    if max_diff > 1e-4:
        print("⚠️  WARNING: Outputs differ significantly!")
    else:
        print("✓ Outputs match (within tolerance)")

    # Statistics
    opt_mean = statistics.mean(optimized_times) * 1000  # ms
    opt_std = statistics.stdev(optimized_times) * 1000
    up_mean = statistics.mean(upstream_times) * 1000
    up_std = statistics.stdev(upstream_times) * 1000

    speedup = up_mean / opt_mean
    improvement_pct = ((up_mean - opt_mean) / up_mean) * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Optimized:  {opt_mean:.3f} ± {opt_std:.3f} ms")
    print(f"Upstream:   {up_mean:.3f} ± {up_std:.3f} ms")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Improvement: {improvement_pct:.1f}%")
    print(f"{'='*80}\n")

    return {
        "device": device_type,
        "optimized_ms": opt_mean,
        "upstream_ms": up_mean,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
    }


def main():
    """Run benchmarks on available devices."""
    # Detect all available devices
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")  # Always available

    print(f"Will benchmark on: {', '.join(d.upper() for d in devices)}\n")

    # Test different sizes
    configs = [
        {"seq_len": 256, "batch_size": 4, "num_heads": 8},
        {"seq_len": 1024, "batch_size": 2, "num_heads": 8},
        {"seq_len": 2048, "batch_size": 1, "num_heads": 8},
    ]

    all_results = {}

    for device_type in devices:
        device_results = []
        for config in configs:
            result = benchmark_attention(
                device_type=device_type,
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                num_heads=config["num_heads"],
                head_dim=64,
                num_runs=50,
                warmup=10,
            )
            device_results.append(result)
        all_results[device_type] = device_results

    # Summary per device
    print("\n" + "="*80)
    print("SUMMARY BY DEVICE")
    print("="*80)
    for device_type, results in all_results.items():
        avg_speedup = statistics.mean([r["speedup"] for r in results])
        avg_improvement = statistics.mean([r["improvement_pct"] for r in results])
        print(f"\n{device_type.upper()}:")
        print(f"  Average speedup:     {avg_speedup:.2f}x")
        print(f"  Average improvement: {avg_improvement:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
