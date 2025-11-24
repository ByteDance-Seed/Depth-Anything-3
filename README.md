<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">Depth Anything 3: Recovering the Visual Space from Any Views</h1>
<!-- <h2 style="border-top: none; margin-top: 3px;">Recovering the Visual Space from Any Views</h2> -->


[**Haotong Lin**](https://haotongl.github.io/)<sup>&ast;</sup> · [**Sili Chen**](https://github.com/SiliChen321)<sup>&ast;</sup> · [**Jun Hao Liew**](https://liewjunhao.github.io/)<sup>&ast;</sup> · [**Donny Y. Chen**](https://donydchen.github.io)<sup>&ast;</sup> · [**Zhenyu Li**](https://zhyever.github.io/) · [**Guang Shi**](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
<br>
[**Bingyi Kang**](https://bingykang.github.io/)<sup>&ast;&dagger;</sup>

&dagger;project lead&emsp;&ast;Equal Contribution

<a href="https://arxiv.org/abs/2511.10647"><img src='https://img.shields.io/badge/arXiv-Depth Anything 3-red' alt='Paper PDF'></a>
<a href='https://depth-anything-3.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything 3-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://github.com/Aedelon/Depth-Anything-3'><img src='https://img.shields.io/badge/Fork-Optimized-orange' alt='Optimized Fork'></a>
<!-- <a href='https://huggingface.co/datasets/depth-anything/VGB'><img src='https://img.shields.io/badge/Benchmark-VisGeo-yellow' alt='Benchmark'></a> -->
<!-- <a href='https://huggingface.co/datasets/depth-anything/data'><img src='https://img.shields.io/badge/Benchmark-xxx-yellow' alt='Data'></a> -->

</div>

---

## ⚡ **This is an Optimized Fork**

<div align="center">

| **What** | **Upstream** | **This Fork** | **Benefit** |
|:--------:|:------------:|:-------------:|:-----------:|
| macOS M-series | ~12 img/s | **28 img/s** | 🚀 **2.35x faster** |
| xformers on macOS | ❌ Build fails | ✅ Auto-excluded | 🍎 **Just works** |
| Memory control | Fixed | **Configurable** | 💾 **No OOM** |
| Precision | Hardcoded | **auto\|fp16\|fp32\|bf16** | 🎛️ **Full control** |
| CUDA | Basic | **TF32 + compile** | ⚡ **+50% faster** |

</div>

This repository is a **production-ready fork** of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) with **cross-platform performance optimizations**:

- 🍎 **macOS Apple Silicon**: Native MPS support, 2-3x faster attention, intelligent compilation
- 🚀 **CUDA Enhancements**: TF32, pinned memory, auto-tuned kernels
- 💾 **Memory Management**: Sub-batching, OOM handling, configurable precision
- 🎛️ **User Control**: Exposed performance flags (CLI, API, Gradio, Backend)
- 📊 **Benchmarks**: ~13-28 img/s on M1/M2/M3 (2.35x speedup vs baseline)

**📖 Quick comparison:** [FORK_VALUE.md](.github/FORK_VALUE.md) | **Detailed docs:** [FORK_HIGHLIGHTS.md](FORK_HIGHLIGHTS.md) | [OPTIMIZATIONS.md](OPTIMIZATIONS.md)

**Original models and paper unchanged** — all credit to ByteDance team. Optimizations are additive and documented.

---

This work presents **Depth Anything 3 (DA3)**, a model that predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses.
In pursuit of minimal modeling, DA3 yields two key insights:
- 💎 A **single plain transformer** (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization,
- ✨ A singular **depth-ray representation** obviates the need for complex multi-task learning.

🏆 DA3 significantly outperforms
[DA2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation,
and [VGGT](https://github.com/facebookresearch/vggt) for multi-view depth estimation and pose estimation.
All models are trained exclusively on **public academic datasets**.

<!-- <p align="center">
  <img src="assets/images/da3_teaser.png" alt="Depth Anything 3" width="100%">
</p> -->
<p align="center">
  <img src="assets/images/demo320-2.gif" alt="Depth Anything 3 - Left" width="70%">
</p>
<p align="center">
  <img src="assets/images/da3_radar.png" alt="Depth Anything 3" width="100%">
</p>


## 📰 News
- **2025-11-14:** 🎉 Paper, project page, code and models are all released.

## 🚀 Performance Optimizations (fork)

This fork includes **platform-specific optimizations** for improved performance on macOS and other platforms:

- ✅ **macOS (Apple Silicon) Support**: Native MPS (Metal) backend with optimized settings
- ✅ **Intelligent torch.compile()**: Auto-enabled on CUDA, auto-disabled on MPS/CPU for optimal performance
- ✅ **Platform-Specific Dependencies**: xformers automatically excluded on macOS
- ✅ **Memory Optimizations**: channels_last format and mixed precision inference
- ✅ **Comprehensive Benchmarking**: Tools to measure and compare performance

**Defaults in this fork (Nov 2025):**
- MPS: fp32 by default (autocast off) for stability; fp16 opt-in via `mixed_precision=True/"float16"`.
- CUDA: torch.compile on, TF32 on, channels_last + pinned memory.
- Sub-batching: `batch_size` to limit unified/GPU memory.

**📊 Performance on macOS M-series:** ~13-28 images/sec (vs baseline with compilation overhead)

**CLI / Backend / Gradio exposed flags:**
- `batch_size`: Sub-batching to limit memory usage (CLI, backend, Gradio).
- `mixed_precision`: `auto|fp16|fp32|bf16` (MPS: fp32 default, fp16 opt-in).

Gradio exposes these controls in the Inference section.

**📖 See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed documentation, benchmarks, and usage guide.**

## ✨ Highlights

### 🏆 Model Zoo
We release three series of models, each tailored for specific use cases in visual geometry.

- 🌟 **DA3 Main Series** (`DA3-Giant`, `DA3-Large`, `DA3-Base`, `DA3-Small`) These are our flagship foundation models, trained with a unified depth-ray representation. By varying the input configuration, a single model can perform a wide range of tasks:
  + 🌊 **Monocular Depth Estimation**: Predicts a depth map from a single RGB image.
  + 🌊 **Multi-View Depth Estimation**: Generates consistent depth maps from multiple images for high-quality fusion.
  + 🎯 **Pose-Conditioned Depth Estimation**: Achieves superior depth consistency when camera poses are provided as input.
  + 📷 **Camera Pose Estimation**:  Estimates camera extrinsics and intrinsics from one or more images.
  + 🟡 **3D Gaussian Estimation**: Directly predicts 3D Gaussians, enabling high-fidelity novel view synthesis.

- 📐 **DA3 Metric Series** (`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

- 🔍 **DA3 Monocular Series** (`DA3Mono-Large`). A dedicated model for high-quality relative monocular depth estimation. Unlike disparity-based models (e.g.,  [Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2)), it directly predicts depth, resulting in superior geometric accuracy.

🔗 Leveraging these available models, we developed a **nested series** (`DA3Nested-Giant-Large`). This series combines a any-view giant model with a metric model to reconstruct visual geometry at a real-world metric scale.

### 🛠️ Codebase Features
Our repository is designed to be a powerful and user-friendly toolkit for both practical application and future research.
- 🎨 **Interactive Web UI & Gallery**: Visualize model outputs and compare results with an easy-to-use Gradio-based web interface.
- ⚡ **Flexible Command-Line Interface (CLI)**: Powerful and scriptable CLI for batch processing and integration into custom workflows.
- 💾 **Multiple Export Formats**: Save your results in various formats, including `glb`, `npz`, depth images, `ply`, 3DGS videos, etc, to seamlessly connect with other tools.
- 🔧 **Extensible and Modular Design**: The codebase is structured to facilitate future research and the integration of new models or functionalities.


<!-- ### 🎯 Visual Geometry Benchmark
We introduce a new benchmark to rigorously evaluate geometry prediction models on three key tasks: pose estimation, 3D reconstruction, and visual rendering (novel view synthesis) quality.

- 🔄 **Broad Model Compatibility**: Our benchmark is designed to be versatile, supporting the evaluation of various models, including both monocular and multi-view depth estimation approaches.
- 🔬 **Robust Evaluation Pipeline**: We provide a standardized pipeline featuring RANSAC-based pose alignment, TSDF fusion for dense reconstruction, and a principled view selection strategy for novel view synthesis.
- 📊 **Standardized Metrics**: Performance is measured using established metrics: AUC for pose accuracy, F1-score and Chamfer Distance for reconstruction, and PSNR/SSIM/LPIPS for rendering quality.
- 🌍 **Diverse and Challenging Datasets**: The benchmark spans a wide range of scenes from datasets like HiRoom, ETH3D, DTU, 7Scenes, ScanNet++, DL3DV, Tanks and Temples, and MegaDepth. -->


## 🚀 Quick Start

### 📦 Installation

#### Prerequisites
- **Python**: 3.10–3.13
- **PyTorch**: 2.0+ with GPU support (CUDA 11.8+ or MPS for macOS)

#### Step 1: Install PyTorch

Choose your platform:

<details>
<summary><b>CUDA (Linux/Windows with NVIDIA GPU)</b></summary>

```bash
# CUDA 12.1 (recommended for RTX 30/40 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
</details>

<details>
<summary><b>macOS (Apple Silicon M1/M2/M3/M4)</b></summary>

```bash
# MPS (Metal Performance Shaders) auto-detected
pip install torch torchvision
```

Verify installation:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Note**: xformers will be automatically excluded (not needed on macOS).
</details>

<details>
<summary><b>CPU Only</b></summary>

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
</details>

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for more options.

#### Step 2: Install Depth Anything 3

```bash
# Basic installation (depth estimation only)
pip install -e .

# With Gradio UI (requires Python >=3.10)
pip install -e ".[app]"

# With 3D Gaussians support (requires torch already installed)
pip install -e ".[gs]"

# Complete installation (all features)
pip install -e ".[all]"
```

#### Step 3: Verify Installation

```bash
# Test import
python -c "from depth_anything_3.api import DepthAnything3; print('✅ Installation successful!')"

# Test CLI
da3 --help
```

#### Troubleshooting Installation

<details>
<summary><b>macOS: "ERROR: Failed building wheel for xformers"</b></summary>

**This is expected and normal!** xformers is automatically excluded on macOS. The installation will continue and use PyTorch's native implementation. No action needed.

If the error persists:
```bash
pip install -e . --no-build-isolation
```
</details>

<details>
<summary><b>"No module named 'torch'"</b></summary>

PyTorch is not installed. Follow Step 1 above to install PyTorch first.
</details>

<details>
<summary><b>"No module named 'gradio'" when running Gradio app</b></summary>

Install with app dependencies:
```bash
pip install -e ".[app]"
```
</details>

For more troubleshooting, see the [Troubleshooting](#-troubleshooting) section below.

---

For detailed model information, please refer to the [Model Cards](#-model-cards) section below.

### 💻 Basic Usage

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)
```

```bash

export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}

```

The model architecture is defined in [`DepthAnything3Net`](src/depth_anything_3/model/da3.py), and specified with a Yaml config file located at [`src/depth_anything_3/configs`](src/depth_anything_3/configs). The input and output processing are handled by [`DepthAnything3`](src/depth_anything_3/api.py). To customize the model architecture, simply create a new config file (*e.g.*, `path/to/new/config`) as:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params

  name: vitb
  out_layers: [5, 7, 9, 11]
  alt_start: 4
  qknorm_start: 4
  rope_start: 4
  cat_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params

  dim_in: &head_dim_in 1536
  output_dim: 2
  features: &head_features 128
  out_channels: &head_out_channels [96, 192, 384, 768]
```

Then, the model can be created with the following code snippet.
```python
from depth_anything_3.cfg import create_object, load_config

Model = create_object(load_config("path/to/new/config"))
```



## 📚 Useful Documentation

- 🖥️ [Command Line Interface](docs/CLI.md)
- 📑 [Python API](docs/API.md)
<!-- - 🏁 [Visual Geometry Benchmark](docs/BENCHMARK.md) -->

## 🗂️ Model Cards

Generally, you should observe that DA3-LARGE achieves comparable results to VGGT.

| 🗃️ Model Name                  | 📏 Params | 📊 Rel. Depth | 📷 Pose Est. | 🧭 Pose Cond. | 🎨 GS | 📐 Met. Depth | ☁️ Sky Seg | 📄 License     |
|-------------------------------|-----------|---------------|--------------|---------------|-------|---------------|-----------|----------------|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE)                     | 0.12B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)                     | 0.08B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)              | 0.35B     | ✅             |              |               |       | ✅             | ✅         | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE)                | 0.35B     | ✅             |              |               |               |       | ✅         | Apache 2.0     |


## ❓ FAQ

- **Older GPUs without XFormers support**: See [Issue #11](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11). Thanks to [@S-Mahoney](https://github.com/S-Mahoney) for the solution!

---

## 🔧 Troubleshooting

### Installation Issues

<details>
<summary><b>macOS: ERROR: Failed building wheel for xformers</b></summary>

**Status**: ✅ **Expected behavior** (not an error)

**Explanation**: xformers is automatically excluded on macOS via `platform_system != 'Darwin'` in dependencies. The code will use PyTorch's native SwiGLU implementation instead.

**Solution**: No action needed. Installation will complete successfully.

If build still fails:
```bash
pip install -e . --no-build-isolation
```
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'torch'</b></summary>

**Cause**: PyTorch not installed or not in Python path.

**Solution**: Install PyTorch first:
```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# macOS
pip install torch torchvision

# CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install Depth Anything 3:
```bash
pip install -e .
```
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'gradio'</b></summary>

**Cause**: Gradio dependencies not installed.

**Solution**:
```bash
pip install -e ".[app]"
```
</details>

---

### Runtime Issues

<details>
<summary><b>CUDA/MPS: Out of Memory (OOM)</b></summary>

**Symptoms**:
- `RuntimeError: CUDA out of memory`
- `RuntimeError: MPS backend out of memory`

**Solutions** (try in order):

1. **Reduce batch size**:
   ```python
   model = DepthAnything3(batch_size=2)  # Instead of default
   ```

2. **Use smaller model**:
   ```python
   model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
   ```

3. **Enable mixed precision**:
   ```python
   model = DepthAnything3(mixed_precision="fp16")
   ```

4. **Lower input resolution**:
   ```python
   prediction = model.inference(images, process_res=384)  # Instead of 504
   ```

5. **Process fewer images**:
   ```python
   for batch in [images[i:i+2] for i in range(0, len(images), 2)]:
       prediction = model.inference(batch)
   ```

**Memory requirements** (approximate):
| Model | FP32 (GB) | FP16 (GB) |
|-------|-----------|-----------|
| DA3-SMALL | 4-6 | 2-3 |
| DA3-BASE | 6-8 | 3-4 |
| DA3-LARGE | 10-14 | 5-7 |
| DA3-GIANT | 18-24 | 9-12 |

</details>

<details>
<summary><b>macOS: Slow inference (<10 images/sec)</b></summary>

**Symptoms**: Processing is slower than expected benchmarks (13-28 img/s).

**Causes & Solutions**:

1. **torch.compile() enabled** (auto-disabled by default, but check):
   ```python
   model = DepthAnything3(enable_compile=False)  # Ensure disabled
   ```

2. **Using FP16 on MPS** (can be unstable):
   ```python
   model = DepthAnything3(mixed_precision="fp32")  # Use FP32 for stability
   ```

3. **Thermal throttling** (M1/M2 under load):
   - Check Activity Monitor → CPU/GPU usage
   - Ensure adequate cooling
   - Close other heavy applications

4. **Swap memory pressure**:
   - Check: `sysctl vm.swapusage`
   - Solution: Reduce batch size or close apps

</details>

<details>
<summary><b>CUDA: Slow inference despite GPU</b></summary>

**Diagnostic**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"torch.compile enabled: {model.enable_compile}")
```

**Solutions**:

1. **Ensure compilation is enabled**:
   ```python
   model = DepthAnything3(enable_compile=True)
   ```

2. **Use appropriate precision**:
   ```python
   # Ampere+ GPUs (RTX 30/40 series)
   model = DepthAnything3(mixed_precision="bfloat16")

   # Older GPUs
   model = DepthAnything3(mixed_precision="float16")
   ```

3. **Check CUDA version compatibility**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # Should match your driver (nvidia-smi)
   ```

</details>

<details>
<summary><b>NaN or Inf in output</b></summary>

**Symptoms**: Depth map contains NaN or infinite values.

**Solutions**:

1. **Disable mixed precision**:
   ```python
   model = DepthAnything3(mixed_precision=False)  # Use FP32
   ```

2. **Check input images**:
   - Ensure images are valid (not corrupted)
   - Check for extreme brightness/darkness
   - Verify format (RGB, not BGR)

3. **Try different model**:
   ```python
   # Try smaller model first
   model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
   ```

</details>

---

### Performance Tuning

<details>
<summary><b>How to maximize speed?</b></summary>

**Platform-specific recommendations**:

**CUDA**:
```python
model = DepthAnything3(
    model_name="da3-small",      # Smaller model
    enable_compile=True,         # Enable compilation
    mixed_precision="bfloat16",  # Ampere+ GPUs
    batch_size=8                 # Larger batches
)
```

**macOS (MPS)**:
```python
model = DepthAnything3(
    model_name="da3-small",      # Smaller model
    enable_compile=False,        # Disable (auto)
    mixed_precision="fp16",      # 2x faster (if stable)
    batch_size=4                 # Balance speed/memory
)
```

**CPU**:
```python
model = DepthAnything3(
    model_name="da3-small",      # Smallest model
    enable_compile=False,        # No benefit on CPU
    mixed_precision="fp16",      # Reduce memory
    batch_size=1                 # Minimize memory
)
```

See [examples/performance_tuning.py](examples/performance_tuning.py) for benchmarking.

</details>

<details>
<summary><b>How to maximize accuracy?</b></summary>

```python
model = DepthAnything3(
    model_name="da3-giant",      # Largest model
    enable_compile=False,        # Avoid numerical changes
    mixed_precision=False        # Full FP32 precision
)

prediction = model.inference(
    images,
    process_res=672              # Higher resolution
)
```

**Trade-off**: ~4x slower, 2x more memory.

</details>

---

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `No module named 'depth_anything_3'` | Package not installed | `pip install -e .` |
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch_size, use fp16, or smaller model |
| `MPS backend out of memory` | Insufficient unified memory | Reduce batch_size or resolution |
| `Failed to download model` | Network issue or invalid model name | Check internet connection, verify model name |
| `Expected 3 channels, got 1` | Grayscale image input | Convert to RGB: `Image.open(path).convert('RGB')` |
| `torch.compile() not supported` | Old PyTorch version | Upgrade: `pip install --upgrade torch>=2.0` |

---

### Getting Help

If your issue isn't covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/Aedelon/Depth-Anything-3/issues)
2. **Check upstream issues**: [Upstream Issues](https://github.com/ByteDance-Seed/Depth-Anything-3/issues)
3. **Run diagnostics**:
   ```bash
   python examples/performance_tuning.py --profile
   ```
4. **Open new issue** with:
   - Platform (CUDA/MPS/CPU)
   - Python version
   - PyTorch version
   - Full error traceback
   - Minimal reproduction code

---

## 📝 Citations
If you find Depth Anything 3 useful in your research or projects, please cite our work:

```
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
