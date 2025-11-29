# 🚀 Depth Anything 3 Command Line Interface

## 📋 Table of Contents

- [📖 Overview](#overview)
- [⚡ Quick Start](#quick-start)
- [📚 Command Reference](#command-reference)
  - [🤖 auto - Auto Mode](#auto---auto-mode)
  - [🖼️ image - Single Image Processing](#image---single-image-processing)
  - [🗂️ images - Image Directory Processing](#images---image-directory-processing)
  - [🎬 video - Video Processing](#video---video-processing)
  - [📐 colmap - COLMAP Dataset Processing](#colmap---colmap-dataset-processing)
  - [🔧 backend - Backend Service](#backend---backend-service)
  - [🎨 gradio - Gradio Application](#gradio---gradio-application)
  - [🖼️ gallery - Gallery Server](#gallery---gallery-server)
- [⚙️ Parameter Details](#parameter-details)
- [💡 Usage Examples](#usage-examples)

## 📖 Overview

The Depth Anything 3 CLI provides a comprehensive command-line toolkit supporting image depth estimation, video processing, COLMAP dataset handling, and web applications.

The backend service enables cache model to GPU so that we do not need to reload model for each command.

## ⚡ Quick Start

The CLI can run fully offline or connect to the backend for cached weights and task scheduling:

```bash
# 🔧 Start backend service (optional, keeps model resident in GPU memory)
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# 🚀 Use auto mode to process input
da3 auto path/to/input --export-dir ./workspace/scene001

# ♻️ Reuse backend for next job
da3 auto path/to/video.mp4 \
    --export-dir ./workspace/scene002 \
    --use-backend \
    --backend-url http://localhost:8008
```

Each export directory contains `scene.glb`, `scene.jpg`, and optional extras such as `depth_vis/` or `gs_video/` depending on the requested format.

## 📚 Command Reference

### 🤖 auto - Auto Mode

Automatically detect input type and dispatch to the appropriate handler.

**Usage:**

```bash
da3 auto INPUT_PATH [OPTIONS]
```

**Input Type Detection:**
- 🖼️ Single image file (.jpg, .png, .jpeg, .webp, .bmp, .tiff, .tif)
- 📁 Image directory
- 🎬 Video file (.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v)
- 📐 COLMAP directory (containing `images/` and `sparse/` subdirectories)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `INPUT_PATH` | str | Required | Input path (image, directory, video, or COLMAP) |
| `--model-dir` | str | Default model | Model directory path |
| `--export-dir` | str | `debug` | Export directory |
| `--export-format` | str | `glb` | Export format (supports `mini_npz`, `glb`, `feat_vis`, etc., can be combined with hyphens) |
| `--device` | str | `cuda` | Device to use |
| `--use-backend` | bool | `False` | Use backend service for inference |
| `--backend-url` | str | `http://localhost:8008` | Backend service URL |
| `--process-res` | int | `504` | Processing resolution |
| `--process-res-method` | str | `upper_bound_resize` | Processing resolution method |
| `--export-feat` | str | `""` | Export features from specified layers, comma-separated (e.g., `"0,1,2"`) |
| `--auto-cleanup` | bool | `False` | Automatically clean export directory without confirmation |
| `--fps` | float | `1.0` | [Video] Frame sampling FPS |
| `--sparse-subdir` | str | `""` | [COLMAP] Sparse reconstruction subdirectory (e.g., `"0"` for `sparse/0/`) |
| `--align-to-input-ext-scale` | bool | `True` | [COLMAP] Align prediction to input extrinsics scale |
| `--conf-thresh-percentile` | float | `40.0` | [GLB] Lower percentile for adaptive confidence threshold |
| `--num-max-points` | int | `1000000` | [GLB] Maximum number of points in the point cloud |
| `--show-cameras` | bool | `True` | [GLB] Show camera wireframes in the exported scene |
| `--feat-vis-fps` | int | `15` | [FEAT_VIS] Frame rate for output video |

**Examples:**

```bash
# 🖼️ Auto-process an image
da3 auto path/to/image.jpg --export-dir ./output

# 🎬 Auto-process a video
da3 auto path/to/video.mp4 --fps 2.0 --export-dir ./output

# 🔧 Use backend service
da3 auto path/to/input \
    --export-format mini_npz-glb \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

---

### 🖼️ image - Single Image Processing

Process a single image for camera pose and depth estimation.

**Usage:**

```bash
da3 image IMAGE_PATH [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IMAGE_PATH` | str | Required | Input image file path |
| `--model-dir` | str | Default model | Model directory path |
| `--export-dir` | str | `debug` | Export directory |
| `--export-format` | str | `glb` | Export format |
| `--device` | str | `cuda` | Device to use |
| `--use-backend` | bool | `False` | Use backend service for inference |
| `--backend-url` | str | `http://localhost:8008` | Backend service URL |
| `--process-res` | int | `504` | Processing resolution |
| `--process-res-method` | str | `upper_bound_resize` | Processing resolution method |
| `--export-feat` | str | `""` | Export feature layer indices (comma-separated) |
| `--auto-cleanup` | bool | `False` | Automatically clean export directory |
| `--conf-thresh-percentile` | float | `40.0` | [GLB] Confidence threshold percentile |
| `--num-max-points` | int | `1000000` | [GLB] Maximum number of points |
| `--show-cameras` | bool | `True` | [GLB] Show cameras |
| `--feat-vis-fps` | int | `15` | [FEAT_VIS] Video frame rate |

**Examples:**

```bash
# ✨ Basic usage
da3 image path/to/image.png --export-dir ./output

# ⚡ With backend acceleration
da3 image path/to/image.png \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output

# 🔍 Export feature visualization
da3 image image.jpg \
    --export-format feat_vis \
    --export-feat "9,19,29,39" \
    --export-dir ./results
```

---

### 🗂️ images - Image Directory Processing

Process a directory of images for batch depth estimation.

**Usage:**

```bash
da3 images IMAGES_DIR [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IMAGES_DIR` | str | Required | Directory path containing images |
| `--image-extensions` | str | `png,jpg,jpeg` | Image file extensions to process (comma-separated) |
| `--model-dir` | str | Default model | Model directory path |
| `--export-dir` | str | `debug` | Export directory |
| `--export-format` | str | `glb` | Export format |
| `--device` | str | `cuda` | Device to use |
| `--use-backend` | bool | `False` | Use backend service for inference |
| `--backend-url` | str | `http://localhost:8008` | Backend service URL |
| `--process-res` | int | `504` | Processing resolution |
| `--process-res-method` | str | `upper_bound_resize` | Processing resolution method |
| `--export-feat` | str | `""` | Export feature layer indices |
| `--auto-cleanup` | bool | `False` | Automatically clean export directory |
| `--conf-thresh-percentile` | float | `40.0` | [GLB] Confidence threshold percentile |
| `--num-max-points` | int | `1000000` | [GLB] Maximum number of points |
| `--show-cameras` | bool | `True` | [GLB] Show cameras |
| `--feat-vis-fps` | int | `15` | [FEAT_VIS] Video frame rate |

**Examples:**

```bash
# 📁 Process directory (defaults to png/jpg/jpeg)
da3 images ./image_folder --export-dir ./output

# 🎯 Custom extensions
da3 images ./dataset --image-extensions "png,jpg,webp" --export-dir ./output

# 🔧 Use backend service
da3 images ./dataset \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

---

### 🎬 video - Video Processing

Process video by extracting frames for depth estimation.

**Usage:**

```bash
da3 video VIDEO_PATH [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VIDEO_PATH` | str | Required | Input video file path |
| `--fps` | float | `1.0` | Frame extraction sampling FPS |
| `--model-dir` | str | Default model | Model directory path |
| `--export-dir` | str | `debug` | Export directory |
| `--export-format` | str | `glb` | Export format |
| `--device` | str | `cuda` | Device to use |
| `--use-backend` | bool | `False` | Use backend service for inference |
| `--backend-url` | str | `http://localhost:8008` | Backend service URL |
| `--process-res` | int | `504` | Processing resolution |
| `--process-res-method` | str | `upper_bound_resize` | Processing resolution method |
| `--export-feat` | str | `""` | Export feature layer indices |
| `--auto-cleanup` | bool | `False` | Automatically clean export directory |
| `--conf-thresh-percentile` | float | `40.0` | [GLB] Confidence threshold percentile |
| `--num-max-points` | int | `1000000` | [GLB] Maximum number of points |
| `--show-cameras` | bool | `True` | [GLB] Show cameras |
| `--feat-vis-fps` | int | `15` | [FEAT_VIS] Video frame rate |

**Examples:**

```bash
# ✨ Basic video processing
da3 video path/to/video.mp4 --export-dir ./output

# ⚙️ Control frame sampling and resolution
da3 video path/to/video.mp4 \
    --fps 2.0 \
    --process-res 1024 \
    --export-dir ./output

# 🔧 Use backend service
da3 video path/to/video.mp4 \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

---

### 📐 colmap - COLMAP Dataset Processing

Run pose-conditioned depth estimation on COLMAP data.

**Usage:**

```bash
da3 colmap COLMAP_DIR [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `COLMAP_DIR` | str | Required | COLMAP directory containing `images/` and `sparse/` subdirectories |
| `--sparse-subdir` | str | `""` | Sparse reconstruction subdirectory (e.g., `"0"` for `sparse/0/`) |
| `--align-to-input-ext-scale` | bool | `True` | Align prediction to input extrinsics scale |
| `--model-dir` | str | Default model | Model directory path |
| `--export-dir` | str | `debug` | Export directory |
| `--export-format` | str | `glb` | Export format |
| `--device` | str | `cuda` | Device to use |
| `--use-backend` | bool | `False` | Use backend service for inference |
| `--backend-url` | str | `http://localhost:8008` | Backend service URL |
| `--process-res` | int | `504` | Processing resolution |
| `--process-res-method` | str | `upper_bound_resize` | Processing resolution method |
| `--export-feat` | str | `""` | Export feature layer indices |
| `--auto-cleanup` | bool | `False` | Automatically clean export directory |
| `--conf-thresh-percentile` | float | `40.0` | [GLB] Confidence threshold percentile |
| `--num-max-points` | int | `1000000` | [GLB] Maximum number of points |
| `--show-cameras` | bool | `True` | [GLB] Show cameras |
| `--feat-vis-fps` | int | `15` | [FEAT_VIS] Video frame rate |

**Examples:**

```bash
# 📐 Process COLMAP dataset
da3 colmap ./colmap_dataset --export-dir ./output

# 🎯 Use specific sparse subdirectory and align scale
da3 colmap ./colmap_dataset \
    --sparse-subdir 0 \
    --align-to-input-ext-scale \
    --export-dir ./output

# 🔧 Use backend service
da3 colmap ./colmap_dataset \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

---

### 🔧 backend - Backend Service

Start model backend service with integrated gallery.

**Usage:**

```bash
da3 backend [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-dir` | str | Default model | Model directory path |
| `--device` | str | `cuda` | Device to use |
| `--host` | str | `127.0.0.1` | Host address to bind to |
| `--port` | int | `8008` | Port number to bind to |
| `--gallery-dir` | str | Default gallery dir | Gallery directory path (optional) |

**Features:**
- 🎯 Keeps model resident in GPU memory
- 🔌 Provides REST inference API
- 📊 Integrated dashboard and status monitoring
- 🖼️ Optional gallery browser (if `--gallery-dir` is provided)

**Available Endpoints:**
- 🏠 `/` - Home page
- 📊 `/dashboard` - Dashboard
- ✅ `/status` - API status
- 🖼️ `/gallery/` - Gallery browser (if enabled)

**Examples:**

```bash
# 🚀 Basic backend service
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# 🖼️ Backend with gallery
da3 backend \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --device cuda \
    --host 0.0.0.0 \
    --port 8008 \
    --gallery-dir ./workspace

# 💻 Use CPU
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --device cpu
```

---

### 🎨 gradio - Gradio Application

Launch Depth Anything 3 Gradio interactive web application.

**Usage:**

```bash
da3 gradio [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-dir` | str | Required | Model directory path |
| `--workspace-dir` | str | Required | Workspace directory path |
| `--gallery-dir` | str | Required | Gallery directory path |
| `--host` | str | `127.0.0.1` | Host address to bind to |
| `--port` | int | `7860` | Port number to bind to |
| `--share` | bool | `False` | Create a public link |
| `--debug` | bool | `False` | Enable debug mode |
| `--cache-examples` | bool | `False` | Pre-cache all example scenes at startup |
| `--cache-gs-tag` | str | `""` | Tag to match scene names for high-res+3DGS caching |

**Examples:**

```bash
# 🎨 Basic Gradio application
da3 gradio \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --workspace-dir ./workspace \
    --gallery-dir ./gallery

# 🌐 Enable sharing and debug
da3 gradio \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --workspace-dir ./workspace \
    --gallery-dir ./gallery \
    --share \
    --debug

# ⚡ Pre-cache examples
da3 gradio \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --workspace-dir ./workspace \
    --gallery-dir ./gallery \
    --cache-examples \
    --cache-gs-tag "dl3dv"
```

---

### 🖼️ gallery - Gallery Server

Launch standalone Depth Anything 3 Gallery server.

**Usage:**

```bash
da3 gallery [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gallery-dir` | str | Default gallery dir | Gallery root directory |
| `--host` | str | `127.0.0.1` | Host address to bind to |
| `--port` | int | `8007` | Port number to bind to |
| `--open-browser` | bool | `False` | Open browser after launch |

**Note:**
The gallery expects each scene folder to contain at least `scene.glb` and `scene.jpg`, with optional subfolders such as `depth_vis/` or `gs_video/`.

**Examples:**

```bash
# 🖼️ Basic gallery server
da3 gallery --gallery-dir ./workspace

# 🌐 Custom host and port
da3 gallery \
    --gallery-dir ./workspace \
    --host 0.0.0.0 \
    --port 8007

# 🚀 Auto-open browser
da3 gallery --gallery-dir ./workspace --open-browser
```

---

## ⚙️ Parameter Details

### 🔧 Common Parameters

- **`--export-dir`**: Output directory, defaults to `debug`
- **`--export-format`**: Export format, supports combining multiple formats with hyphens:
  - 📦 `mini_npz`: Compressed NumPy format
  - 🎨 `glb`: glTF binary format (3D scene)
  - 🔍 `feat_vis`: Feature visualization
  - Example: `mini_npz-glb` exports both formats

- **`--process-res`** / **`--process-res-method`**: Control preprocessing resolution strategy
  - `process-res`: Target resolution (default 504)
  - `process-res-method`: Resize method (default `upper_bound_resize`)

- **`--auto-cleanup`**: Remove existing export directory without confirmation

- **`--use-backend`** / **`--backend-url`**: Reuse running backend service
  - ⚡ Reduces model loading time
  - 🌐 Supports distributed processing

- **`--export-feat`**: Layer indices for exporting intermediate features (comma-separated)
  - Example: `"9,19,29,39"`

### 🎨 GLB Export Parameters

- **`--conf-thresh-percentile`**: Lower percentile for adaptive confidence threshold (default 40.0)
  - Used to filter low-confidence points

- **`--num-max-points`**: Maximum number of points in point cloud (default 1,000,000)
  - Controls output file size and performance

- **`--show-cameras`**: Show camera wireframes in exported scene (default True)

### 🔍 Feature Visualization Parameters

- **`--feat-vis-fps`**: Frame rate for feature visualization output video (default 15)

### 🎬 Video-Specific Parameters

- **`--fps`**: Video frame extraction sampling rate (default 1.0 FPS)
  - Higher values extract more frames

### 📐 COLMAP-Specific Parameters

- **`--sparse-subdir`**: Sparse reconstruction subdirectory
  - Empty string uses `sparse/` directory
  - `"0"` uses `sparse/0/` directory

- **`--align-to-input-ext-scale`**: Align prediction to input extrinsics scale (default True)
  - Ensures depth estimation is consistent with COLMAP scale

---

## 💡 Usage Examples

### 1️⃣ Basic Workflow

```bash
# 🔧 Start backend service
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --host 0.0.0.0 --port 8008

# 🖼️ Process single image
da3 image image.jpg --export-dir ./output1 --use-backend

# 🎬 Process video
da3 video video.mp4 --fps 2.0 --export-dir ./output2 --use-backend

# 📐 Process COLMAP dataset
da3 colmap ./colmap_data --export-dir ./output3 --use-backend
```

### 2️⃣ Using Auto Mode

```bash
# 🤖 Auto-detect and process
da3 auto ./unknown_input --export-dir ./output

# ⚡ With backend acceleration
da3 auto ./unknown_input \
    --use-backend \
    --backend-url http://localhost:8008 \
    --export-dir ./output
```

### 3️⃣ Multi-Format Export

```bash
# 📦 Export both NPZ and GLB formats
da3 auto assets/examples/SOH \
    --export-format mini_npz-glb \
    --export-dir ./workspace/soh

# 🔍 Export feature visualization
da3 image image.jpg \
    --export-format feat_vis \
    --export-feat "9,19,29,39" \
    --export-dir ./results
```

### 4️⃣ Advanced Configuration

```bash
# ⚙️ Custom resolution and point cloud density
da3 image image.jpg \
    --process-res 1024 \
    --num-max-points 2000000 \
    --conf-thresh-percentile 30.0 \
    --export-dir ./output

# 📐 COLMAP advanced options
da3 colmap ./colmap_data \
    --sparse-subdir 0 \
    --align-to-input-ext-scale \
    --process-res 756 \
    --export-dir ./output
```

### 5️⃣ Batch Processing Workflow

```bash
# 🔧 Start backend
da3 backend \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --device cuda \
    --host 0.0.0.0 \
    --port 8008 \
    --gallery-dir ./workspace

# 🔄 Batch process multiple scenes
for scene in scene1 scene2 scene3; do
    da3 auto ./data/$scene \
        --export-dir ./workspace/$scene \
        --use-backend \
        --auto-cleanup
done

# 🖼️ Launch gallery to view results
da3 gallery --gallery-dir ./workspace --open-browser
```

### 6️⃣ Web Applications

```bash
# 🎨 Launch Gradio application
da3 gradio \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE \
    --workspace-dir workspace/gradio \
    --gallery-dir ./gallery \
    --host 0.0.0.0 \
    --port 7860 \
    --share
```

### 7️⃣ Transformer Feature Visualization

```bash
# 🔍 Export Transformer features
# 📦 Combined with numerical output
da3 auto video.mp4 \
    --export-format glb-feat_vis \
    --export-feat "11,21,31" \
    --export-dir ./debug \
    --use-backend
```

---

## 📝 Notes

1. **🔧 Backend Service**: Recommended for processing multiple tasks to improve efficiency
2. **💾 GPU Memory**: Be mindful of GPU memory usage when processing high-resolution inputs
3. **📁 Export Directory**: Use `--auto-cleanup` to avoid manual confirmation for deletion
4. **🔀 Format Combination**: Multiple export formats can be combined with hyphens (e.g., `mini_npz-glb-feat_vis`)
5. **📐 COLMAP Data**: Ensure COLMAP directory structure is correct (contains `images/` and `sparse/` subdirectories)

---

## ❓ Getting Help

View detailed help for any command:

```bash
# 📖 View main help
da3 --help

# 🔍 View specific command help
da3 auto --help
da3 image --help
da3 backend --help
```
