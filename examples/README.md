# Depth Anything 3 - Usage Examples

This directory contains practical examples demonstrating various use cases for Depth Anything 3.

## 📚 Examples Overview

| Example | Description | Difficulty | Use Case |
|---------|-------------|------------|----------|
| [single_image.py](#single_imagepy) | Basic depth estimation from one image | ⭐ Beginner | Quick start, testing |
| [batch_processing.py](#batch_processingpy) | Process multiple images efficiently | ⭐⭐ Intermediate | Dataset processing |
| [video_processing.py](#video_processingpy) | Extract depth from video files | ⭐⭐ Intermediate | Video analysis |
| [performance_tuning.py](#performance_tuningpy) | Optimize for your hardware | ⭐⭐⭐ Advanced | Production deployment |
| [export_formats.py](#export_formatspy) | Export results in different formats | ⭐⭐ Intermediate | Integration with other tools |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -e ".[all]"

# Run basic example
python examples/single_image.py

# Run with your own image
python examples/single_image.py --image path/to/your/image.jpg
```

## 📖 Detailed Examples

### single_image.py
**What it does:** Processes a single image and saves depth map + visualization.

**Key concepts:**
- Loading a pretrained model
- Basic inference
- Saving results

**Run:**
```bash
python examples/single_image.py
python examples/single_image.py --image custom.jpg --output-dir results/
```

---

### batch_processing.py
**What it does:** Efficiently processes multiple images with memory management.

**Key concepts:**
- Sub-batching for memory control
- Progress tracking
- Batch optimization

**Run:**
```bash
python examples/batch_processing.py --images-dir folder/ --batch-size 4
python examples/batch_processing.py --images-dir dataset/ --mixed-precision fp16
```

---

### video_processing.py
**What it does:** Extracts depth maps from video files frame-by-frame.

**Key concepts:**
- Video frame extraction
- Temporal processing
- Video export

**Run:**
```bash
python examples/video_processing.py --video input.mp4
python examples/video_processing.py --video input.mp4 --fps 15 --output-format glb
```

---

### performance_tuning.py
**What it does:** Demonstrates performance optimization for different platforms.

**Key concepts:**
- Platform detection
- Mixed precision
- Compilation strategies
- Benchmarking

**Run:**
```bash
# Auto-detect platform and optimize
python examples/performance_tuning.py

# Force specific configuration
python examples/performance_tuning.py --mixed-precision fp16 --batch-size 2
```

---

### export_formats.py
**What it does:** Shows how to export results in various formats (GLB, PLY, NPZ, etc.).

**Key concepts:**
- Format conversion
- 3D visualization
- Data serialization

**Run:**
```bash
python examples/export_formats.py --format glb
python examples/export_formats.py --format all  # Export all formats
```

## 🔧 Common Patterns

### Pattern 1: Maximum Accuracy
```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3(
    model_name="da3-large",
    enable_compile=False,     # Disable optimizations
    mixed_precision=False     # Use FP32
)
```

### Pattern 2: Maximum Speed
```python
model = DepthAnything3(
    model_name="da3-small",   # Smaller model
    enable_compile=True,      # Enable compilation (CUDA)
    mixed_precision="fp16"    # Half precision
)
```

### Pattern 3: Memory-Constrained (8GB GPU)
```python
model = DepthAnything3(
    model_name="da3-small",
    batch_size=2,             # Small batches
    mixed_precision="fp16"    # Reduce memory
)
```

### Pattern 4: macOS Apple Silicon
```python
model = DepthAnything3(
    model_name="da3-large",
    enable_compile=False,     # Auto-disabled, but explicit
    mixed_precision="fp32"    # Stable default for MPS
).to("mps")
```

## 🐛 Troubleshooting

### OOM (Out of Memory)
```python
# Reduce batch size
model = DepthAnything3(batch_size=1)

# Use smaller model
model = DepthAnything3(model_name="da3-small")

# Lower precision
model = DepthAnything3(mixed_precision="fp16")
```

### Slow Inference
```python
# Enable compilation (CUDA only)
model = DepthAnything3(enable_compile=True)

# Use half precision
model = DepthAnything3(mixed_precision="fp16")

# Use smaller resolution
prediction = model.inference(images, process_res=384)
```

### macOS xformers Error
This is expected! xformers is automatically excluded on macOS. The code will use PyTorch's native implementation.

## 📊 Performance Comparison

Run all examples with benchmarking:
```bash
# Benchmark each example
for ex in single_image batch_processing video_processing; do
    time python examples/$ex.py
done
```

## 🤝 Contributing Examples

Have a useful example? Please contribute!

1. Follow the existing structure
2. Add docstrings and comments
3. Include a `--help` argument
4. Update this README
5. Submit a PR

## 📖 Additional Resources

- [OPTIMIZATIONS.md](../OPTIMIZATIONS.md): Performance tuning guide
- [FORK_HIGHLIGHTS.md](../FORK_HIGHLIGHTS.md): Fork-specific features
- [docs/API.md](../docs/API.md): Complete API reference
- [docs/CLI.md](../docs/CLI.md): Command-line usage
