# Changelog

All notable changes to this fork will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-24

### Added

#### Platform Support
- **macOS Apple Silicon (MPS) native support** with optimized performance
  - Auto-detection of MPS backend
  - Intelligent torch.compile() toggling (auto-disabled on MPS for 2x speedup)
  - Optimized attention implementation (2-3x faster than PyTorch default on MPS)
  - Platform-specific dependencies: xformers automatically excluded on macOS via `platform_system != 'Darwin'`

#### Performance Optimizations
- **Centralized optimized attention module** (`src/depth_anything_3/model/optimized_attention.py`)
  - Automatic backend selection (manual on MPS, PyTorch native on CUDA/CPU)
  - Single source of truth, no code duplication
  - 2-3x speedup on MPS devices

- **Configurable mixed precision inference**
  - New parameter: `mixed_precision="auto|fp16|fp32|bf16"`
  - Platform-aware defaults:
    - MPS: fp32 (stability), opt-in fp16 for speed
    - CUDA: bfloat16 if supported, else float16
    - CPU: float16
  - Exposed in Python API, CLI, Backend, and Gradio UI

- **Sub-batching for memory management**
  - New parameter: `batch_size=N` to limit memory usage
  - Prevents OOM errors on small GPUs and unified memory (8GB Macs)
  - Processes large batches in chunks automatically
  - Exposed in all interfaces (API, CLI, Backend, Gradio)

- **CUDA optimizations**
  - TF32 tensor cores enabled (10-20% speedup on Ampere+ GPUs)
  - Pinned memory for async H2D transfers
  - channels_last memory format (10-20% faster convolutions)
  - torch.compile() auto-enabled with optimal settings

- **torch.compile() enhancements**
  - Scalar output capture to reduce graph breaks
  - Automatic dynamic shapes support
  - Intelligent platform detection

#### User Interface Improvements
- **Gradio UI enhancements**
  - Performance controls exposed (batch size, mixed precision)
  - OOM error catching with actionable hints
  - Better error messages for invalid configurations

- **CLI enhancements**
  - New flags: `--batch-size`, `--mixed-precision`
  - Improved error messages
  - Backend mode for persistent model loading

- **Backend API improvements**
  - Dynamic model reloading when parameters change
  - `/health` endpoint with detailed status
  - Memory management utilities integration

#### Documentation
- **OPTIMIZATIONS.md**: Comprehensive performance guide
  - Detailed optimization explanations
  - Benchmarking tools and results
  - Platform-specific guides (macOS, CUDA, CPU)
  - Troubleshooting section

- **FORK_HIGHLIGHTS.md**: Comparison with upstream
  - Feature-by-feature comparison
  - Performance benchmarks
  - When to use this fork vs upstream

- **Improved code documentation**
  - Enhanced docstrings in `api.py`
  - Inline comments for optimization decisions

#### Testing & Benchmarking
- `benchmark_performance.py`: Comprehensive performance testing tool
- `test_optimizations.py`: Validation of optimization features
- `test_attention.py`: Attention module correctness tests

### Changed

- **torch.compile() behavior**
  - Now auto-disabled on MPS/CPU (better performance)
  - Auto-enabled on CUDA with optimal settings
  - User can override with `enable_compile` parameter

- **Default precision on MPS**
  - Changed from auto fp16 to fp32 for stability
  - Users can opt-in to fp16 via `mixed_precision="float16"`

- **Backend model loading**
  - Now reloads model when performance parameters change
  - Tracks `batch_size` and `mixed_precision` to detect mismatches

- **Error handling**
  - Better error messages for common issues
  - OOM errors caught in Gradio with suggestions
  - Invalid mixed_precision values rejected with clear message

### Fixed

- **OOM errors on small GPUs**
  - Sub-batching prevents memory exhaustion
  - Gradio catches OOM and suggests lowering batch/resolution

- **Logger warnings**
  - Fixed mixed_precision parsing warnings
  - Proper handling of "auto" value

- **Gradio event handlers**
  - Fixed signature after adding performance controls
  - Proper parameter passing for batch_size and mixed_precision

- **Visualization handler**
  - Fixed signature after performance controls update

### Performance

#### macOS M-series (MPS)
| Configuration | Baseline | This Fork | Speedup |
|--------------|----------|-----------|---------|
| 5 images (280×504) | ~12 img/s | **28.2 img/s** | **2.35x** |
| 20 images | ~8 img/s | **12.9 img/s** | **1.61x** |

#### CUDA (Expected)
- +10-20% from TF32 tensor cores
- +10-20% from channels_last memory format
- Lower latency from pinned memory transfers

### Technical Debt

- TODO: Support all types of iterable in `parallel_utils.py`
- TODO: Prune sky region in `gsply_helpers.py`
- FIXME: Review drop_path2 in `block.py`

### Attribution

This fork is based on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Ltd.

Original paper: Lin et al., "Depth Anything 3: Recovering the visual space from any views", arXiv:2511.10647, 2025

All models and paper remain unchanged from upstream. This fork only adds performance optimizations.

---

## Upstream Sync

Forked from upstream commit: `ed6989a` (contiguous before reshape)

To sync with upstream:
```bash
git fetch upstream
git merge upstream/main
```

---

[0.1.0]: https://github.com/Aedelon/Depth-Anything-3/releases/tag/v0.1.0
