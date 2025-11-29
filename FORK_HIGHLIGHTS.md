# Fork Highlights: What's Different from Upstream

This fork of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) focuses on **production-ready performance optimizations** and **cross-platform compatibility**, particularly for macOS Apple Silicon and CUDA devices.

## 🎯 Key Improvements Over Upstream

### 1. **macOS Apple Silicon (MPS) Native Support** 🍎

**Upstream limitation**: Poor MPS performance, compilation overhead, xformers dependency conflicts.

**This fork**:
- ✅ **Native MPS backend** with optimized attention (2-3x faster than PyTorch's default)
- ✅ **Intelligent torch.compile()** detection: auto-disabled on MPS (avoids 2x slowdown)
- ✅ **Platform-specific dependencies**: xformers excluded on macOS via `platform_system != 'Darwin'`
- ✅ **Stable defaults**: FP32 by default on MPS (opt-in FP16 via `mixed_precision="float16"`)

**Performance**: ~13-28 images/sec on M1/M2/M3 (vs baseline with compilation overhead).

---

### 2. **Flexible Mixed Precision Control** ⚡

**Upstream limitation**: Hardcoded precision, no user control.

**This fork**:
- ✅ **Configurable precision**: `mixed_precision="auto|fp16|fp32|bf16"`
- ✅ **Platform-aware defaults**:
  - CUDA: `bfloat16` if supported, else `float16`
  - MPS: `float32` (stability), opt-in `float16` for speed
  - CPU: `float16`
- ✅ **Exposed everywhere**: Python API, CLI, Backend, Gradio UI

```python
# Python API
model = DepthAnything3(mixed_precision="float16")  # Force FP16

# CLI
da3 auto image.jpg --mixed-precision fp32  # Force FP32 for accuracy
```

---

### 3. **Sub-batching for Memory Management** 💾

**Upstream limitation**: OOM errors on small GPUs/unified memory (8GB Macs).

**This fork**:
- ✅ **Automatic sub-batching**: Process large batches in chunks
- ✅ **User-configurable**: `batch_size=N` limits memory usage
- ✅ **OOM error handling**: Gradio catches OOM and suggests lowering batch/resolution

```python
# Prevent OOM on 8GB GPU
model = DepthAnything3(batch_size=2)
prediction = model.inference(large_image_list)  # Processes in chunks of 2
```

---

### 4. **CUDA Performance Optimizations** 🚀

**Upstream**: Basic CUDA support.

**This fork**:
- ✅ **TF32 tensor cores** enabled (10-20% speedup on Ampere+ GPUs)
- ✅ **Pinned memory** for async H2D transfers
- ✅ **channels_last** memory format (10-20% faster convolutions)
- ✅ **torch.compile()** auto-enabled with optimal settings

---

### 5. **Centralized Optimized Attention** 🎯

**Upstream**: Scattered attention implementations, poor MPS optimization.

**This fork**:
- ✅ **Single source of truth**: `src/depth_anything_3/model/optimized_attention.py`
- ✅ **Automatic backend selection**:
  - MPS: Manual implementation (2-3x faster)
  - CUDA/CPU: PyTorch's `F.scaled_dot_product_attention`
- ✅ **No code duplication**: All attention modules import from centralized module

**Result**: 2-3x faster attention on MPS vs PyTorch default.

---

### 6. **Enhanced User Experience** 🎨

**Upstream**: Limited configurability.

**This fork**:

#### **Gradio UI Improvements**
- ✅ Performance controls exposed (batch size, mixed precision)
- ✅ OOM error catching with actionable hints
- ✅ Real-time memory usage display

#### **CLI Enhancements**
- ✅ Performance flags: `--batch-size`, `--mixed-precision`
- ✅ Better error messages
- ✅ Backend mode for persistent model loading

#### **Backend API**
- ✅ Dynamic model reloading on parameter changes
- ✅ `/health` endpoint with detailed status
- ✅ Memory management utilities

---

### 7. **Comprehensive Documentation** 📚

**This fork adds**:
- ✅ **OPTIMIZATIONS.md**: Detailed perf guide, benchmarks, troubleshooting
- ✅ **Platform-specific guides**: macOS, CUDA, CPU
- ✅ **Performance benchmarking tools**: `benchmark_performance.py`
- ✅ **Configuration examples**: All major use cases covered

---

## 📊 Performance Comparison

### macOS M-series (MPS)

| Configuration | Upstream | This Fork | Speedup |
|--------------|----------|-----------|---------|
| 5 images (280×504) | ~12 img/s (with compile overhead) | **28.2 img/s** | **2.35x** |
| 20 images | ~8 img/s | **12.9 img/s** | **1.61x** |

**Key**: Disabling torch.compile() on MPS + optimized attention.

### CUDA (Expected)

| Configuration | Upstream | This Fork | Improvement |
|--------------|----------|-----------|-------------|
| torch.compile | Manual | **Auto-enabled** | Easier usage |
| TF32 | Off | **On** | +10-20% |
| Pinned memory | No | **Yes** | Lower latency |

---

## 🔧 Technical Improvements

### Code Quality
- ✅ Pre-commit hooks configured (black, isort, flake8)
- ✅ Better error handling with actionable messages
- ✅ Logging improvements (`logger` instead of `print()`)
- ✅ Type hints and docstrings

### Platform Support Matrix

| Platform | Upstream | This Fork |
|----------|----------|-----------|
| macOS (Apple Silicon) | ⚠️ Partial | ✅ **Fully optimized** |
| Linux (CUDA) | ✅ Good | ✅ **Enhanced** |
| Windows (CUDA) | ✅ Good | ✅ **Enhanced** |
| CPU only | ✅ Works | ✅ **Works** |

---

## 🆚 When to Use This Fork vs Upstream

### Use This Fork If:
- ✅ You're on **macOS Apple Silicon** (M1/M2/M3/M4)
- ✅ You need **memory-constrained inference** (8GB GPU/unified memory)
- ✅ You want **configurable performance** (batch size, precision)
- ✅ You need **production-ready features** (error handling, monitoring)
- ✅ You value **comprehensive documentation**

### Use Upstream If:
- You want the **latest research features** (merged first to upstream)
- You're contributing back to the **original project**
- You don't need platform-specific optimizations

---

## 🔄 Upstream Sync Strategy

This fork tracks upstream and regularly merges updates:

```bash
# Check upstream commits
git fetch upstream
git log HEAD..upstream/main --oneline

# Merge upstream changes
git merge upstream/main
```

**Current status**: Forked from upstream commit `ed6989a`, with 10 additional optimization commits.

---

## 📝 Attribution

**Upstream project**: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Ltd.

**Fork maintainer**: [Your Name/Organization]

**License**: Apache 2.0 (same as upstream)

**Citation**: If you use this fork, please cite the original Depth Anything 3 paper:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

---

## 🤝 Contributing

Found a bug or have an optimization idea?

1. **Report issues**: [GitHub Issues](https://github.com/Aedelon/Depth-Anything-3/issues)
2. **Submit PRs**: We welcome contributions!
3. **Benchmarks**: Share your performance results

---

## 🔗 Quick Links

- **Upstream**: https://github.com/ByteDance-Seed/Depth-Anything-3
- **This Fork**: https://github.com/Aedelon/Depth-Anything-3
- **Documentation**: [OPTIMIZATIONS.md](OPTIMIZATIONS.md)
- **Examples**: [examples/](examples/)