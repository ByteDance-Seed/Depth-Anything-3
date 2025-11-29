# 📚 DepthAnything3 API Documentation

## 📑 Table of Contents

1. [📖 Overview](#overview)
2. [💡 Usage Examples](#usage-examples)
3. [🔧 Core API](#core-api)
   - [DepthAnything3 Class](#depthanything3-class)
   - [inference() Method](#inference-method)
4. [⚙️ Parameters](#parameters)
   - [Input Parameters](#input-parameters)
   - [Pose Alignment Parameters](#pose-alignment-parameters)
   - [Feature Export Parameters](#feature-export-parameters)
   - [Rendering Parameters](#rendering-parameters)
   - [Processing Parameters](#processing-parameters)
   - [Export Parameters](#export-parameters)
5. [📤 Export Formats](#export-formats)
6. [↩️ Return Value](#return-value)

## 📖 Overview

This documentation provides comprehensive API reference for DepthAnything3, including usage examples, parameter specifications, export formats, and advanced features. It covers both basic pose and depth estimation workflows and advanced pose-conditioned processing with multiple export capabilities.

## 💡 Usage Examples

Here are quick examples to get you started:

### 🚀 Basic Depth Estimation
```python
from depth_anything_3.api import DepthAnything3

# Initialize and run inference
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
prediction = model.inference(["image1.jpg", "image2.jpg"])
```

### 📷 Pose-Conditioned Depth Estimation
```python
import numpy as np

# With camera parameters for better consistency
prediction = model.inference(
    image=["image1.jpg", "image2.jpg"],
    extrinsics=extrinsics_array,  # (N, 4, 4)
    intrinsics=intrinsics_array   # (N, 3, 3)
)
```

### 📤 Export Results
```python
# Export depth data and 3D visualization
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="mini_npz-glb"
)
```

### 🔍 Feature Extraction
```python
# Export intermediate features from specific layers
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="feat_vis",
    export_feat_layers=[0, 1, 2]  # Export features from layers 0, 1, 2
)
```

### ✨ Advanced Export with Gaussian Splatting
```python
# Export multiple formats including Gaussian Splatting
# Note: infer_gs=True requires da3-giant or da3nested-giant-large model
model = DepthAnything3(model_name="da3-giant").to("cuda")

prediction = model.inference(
    image=image_paths,
    extrinsics=extrinsics_array,
    intrinsics=intrinsics_array,
    export_dir="./output",
    export_format="npz-glb-gs_ply-gs_video",
    align_to_input_ext_scale=True,
    infer_gs=True,  # Required for gs_ply and gs_video exports
)
```

### 🎨 Advanced Export with Feature Visualization
```python
# Export with intermediate feature visualization
prediction = model.inference(
    image=image_paths,
    export_dir="./output",
    export_format="mini_npz-glb-depth_vis-feat_vis",
    export_feat_layers=[0, 5, 10, 15, 20],
    feat_vis_fps=30,
)
```

## 🔧 Core API

### 🔨 DepthAnything3 Class

The main API class that provides depth estimation capabilities with optional pose conditioning.

#### 🎯 Initialization

```python
from depth_anything_3 import DepthAnything3

# Initialize the model with a model name
model = DepthAnything3(model_name="da3-large")
model = model.to("cuda")  # Move to GPU
```

**Parameters:**
- `model_name` (str, default: "da3-large"): The name of the model preset to use.
  - **Available models:**
    - 🦾 `"da3-giant"` - 1.15B params, any-view model with GS support
    - ⭐ `"da3-large"` - 0.35B params, any-view model (recommended for most use cases)
    - 📦 `"da3-base"` - 0.12B params, any-view model
    - 🪶 `"da3-small"` - 0.08B params, any-view model
    - 👁️ `"da3mono-large"` - 0.35B params, monocular depth only
    - 📏 `"da3metric-large"` - 0.35B params, metric depth with sky segmentation
    - 🎯 `"da3nested-giant-large"` - 1.40B params, nested model with all features

### 🚀 inference() Method

The primary inference method that processes images and returns depth predictions.

```python
prediction = model.inference(
    image=image_list,
    extrinsics=extrinsics_array,      # Optional
    intrinsics=intrinsics_array,      # Optional
    align_to_input_ext_scale=True,   # Whether to align predicted poses to input scale
    infer_gs=True,                   # Enable Gaussian branch for gs exports
    render_exts=render_extrinsics,    # Optional renders for gs_video
    render_ixts=render_intrinsics,    # Optional renders for gs_video
    render_hw=(height, width),        # Optional renders for gs_video
    process_res=504,
    process_res_method="upper_bound_resize",
    export_dir="output_directory",    # Optional
    export_format="mini_npz",
    export_feat_layers=[],            # List of layer indices to export features from
    conf_thresh_percentile=40.0,      # Confidence threshold percentile for depth map in GLB export
    num_max_points=1_000_000,         # Maximum number of points to export in GLB export
    show_cameras=True,                # Whether to show cameras in GLB export
    feat_vis_fps=15,                  # Frames per second for feature visualization in feat_vis export
    export_kwargs={}                  # Optional, additional arguments to export functions. export_format:key:val, see 'Parameters/Export Parameters' for details
)
```

## ⚙️ Parameters

### 📸 Input Parameters

#### `image` (required)
- **Type**: `List[Union[np.ndarray, Image.Image, str]]`
- **Description**: List of input images. Can be numpy arrays, PIL Images, or file paths.
- **Example**:
  ```python
  # From file paths
  image = ["image1.jpg", "image2.jpg", "image3.jpg"]

  # From numpy arrays
  image = [np.array(img1), np.array(img2)]

  # From PIL Images
  image = [Image.open("image1.jpg"), Image.open("image2.jpg")]
  ```

#### `extrinsics` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(N, 4, 4)` where N is the number of input images
- **Description**: Camera extrinsic matrices (world-to-camera transformation). When provided, enables pose-conditioned depth estimation mode.
- **Note**: If not provided, the model operates in standard depth estimation mode.

#### `intrinsics` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(N, 3, 3)` where N is the number of input images
- **Description**: Camera intrinsic matrices containing focal length and principal point information. When provided, enables pose-conditioned depth estimation mode.

### 🎯 Pose Alignment Parameters

#### `align_to_input_ext_scale` (default: True)
- **Type**: `bool`
- **Description**: When True the predicted extrinsics are replaced with the input
  ones and the depth maps are rescaled to match their metric scale. When False the
  function returns the internally aligned poses computed via Umeyama alignment.

#### `infer_gs` (default: False)
- **Type**: `bool`
- **Description**: Enable Gaussian Splatting branch for gaussian splatting exports. Required when using `gs_ply` or `gs_video` export formats.

### 🔍 Feature Export Parameters

#### `export_feat_layers` (default: [])
- **Type**: `List[int]`
- **Description**: List of layer indices to export intermediate features from. Features are stored in the `aux` dictionary of the Prediction object with keys like `feat_layer_0`, `feat_layer_1`, etc.

### 🎥 Rendering Parameters

These arguments are only used when exporting Gaussian-splatting videos (include
`"gs_video"` in `export_format`). They describe an auxiliary camera trajectory
with ``M`` views.

#### `render_exts` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(M, 4, 4)`
- **Description**: Camera extrinsics for the synthesized trajectory. If omitted,
  the exporter falls back to the predicted poses.

#### `render_ixts` (optional)
- **Type**: `Optional[np.ndarray]`
- **Shape**: `(M, 3, 3)`
- **Description**: Camera intrinsics for each rendered frame. Leave `None` to
  reuse the input intrinsics.

#### `render_hw` (optional)
- **Type**: `Optional[Tuple[int, int]]`
- **Description**: Explicit output resolution `(height, width)` for the rendered
  frames. Defaults to the input resolution when not provided.

### ⚡ Processing Parameters

#### `process_res` (default: 504)
- **Type**: `int`
- **Description**: Base resolution for processing. The model will resize images to this resolution for inference.

#### `process_res_method` (default: "upper_bound_resize")
- **Type**: `str`
- **Description**: Method for resizing images to the target resolution.
- **Options**:
  - `"upper_bound_resize"`: Resize so that the specified dimension (504) becomes the longer side
  - `"lower_bound_resize"`: Resize so that the specified dimension (504) becomes the shorter side
- **Example**:
  - Input: 1200×1600 → Output: 378×504 (with `process_res=504`, `process_res_method="upper_bound_resize"`)
  - Input: 504×672 → Output: 504×672 (no change needed)

### 📦 Export Parameters

#### `export_dir` (optional)
- **Type**: `Optional[str]`
- **Description**: Directory path where exported files will be saved. If not provided, no files will be exported.

#### `export_format` (default: "mini_npz")
- **Type**: `str`
- **Description**: Format for exporting results. Supports multiple formats separated by `-`.
- **Example**: `"mini_npz-glb"` exports both mini_npz and glb formats.

#### 🌐 GLB Export Parameters

These parameters are passed directly to the `inference()` method and only apply when `export_format` includes `"glb"`.

##### `conf_thresh_percentile` (default: 40.0)
- **Type**: `float`
- **Description**: Lower percentile for adaptive confidence threshold. Points below this confidence percentile will be filtered out from the point cloud.

##### `num_max_points` (default: 1,000,000)
- **Type**: `int`
- **Description**: Maximum number of points in the exported point cloud. If the point cloud exceeds this limit, it will be downsampled.

##### `show_cameras` (default: True)
- **Type**: `bool`
- **Description**: Whether to include camera wireframes in the exported GLB file for visualization.

#### 🎨 Feature Visualization Parameters

These parameters are passed directly to the `inference()` method and only apply when `export_format` includes `"feat_vis"`.

##### `feat_vis_fps` (default: 15)
- **Type**: `int`
- **Description**: Frame rate for the output video when visualizing features across multiple images.

#### ✨🎥 3DGS and 3DGS Video Parameters

These parameters are passed directly to the `inference()` method and only apply when `export_format` includes `"gs_ply"` or `"gs_video"`.

##### `export_kwargs` (default: `{}`)
- Type: `dict[str, dict[str, Any]]`
- Description: Per-format extra arguments passed to export functions, mainly for `"gs_ply"` and `"gs_video"`.
  - Access pattern: `export_kwargs[export_format][key] = value`
  - Example:
    ```python
    {
        "gs_ply": {
            "gs_views_interval": 1,
        },
        "gs_video": {
            "trj_mode": "interpolate_smooth",
            "chunk_size": 1,
            "vis_depth": None,
        },
    }
    ```

## 📤 Export Formats

The API supports multiple export formats for different use cases:

### 📊 `mini_npz`
- **Description**: Minimal NPZ format containing essential data
- **Contents**: `depth`, `conf`, `exts`, `ixts`
- **Use case**: Lightweight storage for depth data with camera parameters

### 📦 `npz`
- **Description**: Full NPZ format with comprehensive data
- **Contents**: `depth`, `conf`, `exts`, `ixts`, `image`, etc.
- **Use case**: Complete data export for advanced processing

### 🌐 `glb`
- **Description**: 3D visualization format with point cloud and camera poses
- **Contents**:
  - Point cloud with colors from original images
  - Camera wireframes for visualization
  - Confidence-based filtering and downsampling
- **Use case**: 3D visualization, inspection, and analysis
- **Features**:
  - Automatic sky depth handling
  - Confidence threshold filtering
  - Background filtering (black/white)
  - Scene scale normalization
- **Parameters** (passed via `inference()` method directly):
  - `conf_thresh_percentile` (float, default: 40.0): Lower percentile for adaptive confidence threshold. Points below this confidence percentile will be filtered out.
  - `num_max_points` (int, default: 1,000,000): Maximum number of points in the exported point cloud. If exceeded, points will be downsampled.
  - `show_cameras` (bool, default: True): Whether to include camera wireframes in the exported GLB file for visualization.

### ✨ `gs_ply`
- **Description**: Gaussian Splatting point cloud format
- **Contents**: 3DGS data in PLY format. Compatible with standard 3DGS viewers such as [SuperSplat](https://superspl.at/editor) (recommended), [SPARK](https://sparkjs.dev/viewer/).
- **Use case**: Gaussian Splatting reconstruction
- **Requirements**: Must set `infer_gs=True` when calling `inference()`. Only supported by `da3-giant` and `da3nested-giant-large` models.
- **Additional configs**, provided via `export_kwargs` (see [Export Parameters](#export-parameters)):
  - `gs_views_interval`: Export to 3DGS every N views, default: `1`.

### 🎥 `gs_video`
- **Description**: Rasterized 3DGS to obtain videos
- **Contents**: A video of 3DGS-rasterized views using either provided viewpoints or a predefined camera trajectory.
- **Use case**: Video rendering for Gaussian Splatting
- **Requirements**: Must set `infer_gs=True` when calling `inference()`. Only supported by `da3-giant` and `da3nested-giant-large` models.
- **Note**: Can optionally use `render_exts`, `render_ixts`, and `render_hw` parameters in `inference()` method to specify novel viewpoints.
- **Additional configs**, provided via `export_kwargs` (see [Export Parameters](#export-parameters)):
  - `extrinsics`: Optional world-to-camera poses for novel views. Falls back to the predicted poses of input views if not provided. (Alternatively, use `render_exts` parameter in `inference()`)
  - `intrinsics`: Optional camera intrinsics for novel views. Falls back to the predicted intrinsics of input views if not provided. (Alternatively, use `render_ixts` parameter in `inference()`)
  - `out_image_hw`: Optional output resolution `H x W`. Falls back to input resolution if not provided. (Alternatively, use `render_hw` parameter in `inference()`)
  - `chunk_size`: Number of views rasterized per batch. Default: `8`.
  - `trj_mode`: Predefined camera trajectory for novel-view rendering.
  - `color_mode`: Same as `render_mode` in [gsplat](https://docs.gsplat.studio/main/apis/rasterization.html#gsplat.rasterization).
  - `vis_depth`: How depth is combined with RGB. Default: `hcat` (horizontal concatenation).
  - `enable_tqdm`: Whether to display a tqdm progress bar during rendering.
  - `output_name`: File name of the rendered video.
  - `video_quality`: Video quality to save. Default: `high`.
    - `high`: High quality video (default)
    - `medium`: Medium quality video (balance of storage space and quality)
    - `low`: Low quality video (fewer storage space)

### 🔍 `feat_vis`
- **Description**: Feature visualization format
- **Contents**: PCA-visualized intermediate features from specified layers
- **Use case**: Model interpretability and feature analysis
- **Note**: Requires `export_feat_layers` to be specified
- **Parameters** (passed via `inference()` method directly):
  - `feat_vis_fps` (int, default: 15): Frame rate for the output video when visualizing features across multiple images.

### 🎨 `depth_vis`
- **Description**: Depth visualization format
- **Contents**: Color-coded depth maps alongside original images
- **Use case**: Visual inspection of depth estimation quality

### 🔗 Multiple Format Export
You can export multiple formats simultaneously by separating them with `-`:

```python
# Export both mini_npz and glb formats
export_format = "mini_npz-glb"

# Export multiple formats
export_format = "npz-glb-gs_ply"
```

## ↩️ Return Value

The `inference()` method returns a `Prediction` object with the following attributes:

### 📊 Core Outputs

- **depth**: `np.ndarray` - Estimated depth maps with shape `(N, H, W)` where N is the number of images, H is height, and W is width.
- **conf**: `np.ndarray` - Confidence maps with shape `(N, H, W)` indicating prediction reliability (optional, depends on model).

### 📷 Camera Parameters

- **extrinsics**: `np.ndarray` - Camera extrinsic matrices with shape `(N, 3, 4)` representing world-to-camera transformations. Only present if camera poses were estimated or provided as input.
- **intrinsics**: `np.ndarray` - Camera intrinsic matrices with shape `(N, 3, 3)` containing focal length and principal point information. Only present if poses were estimated or provided as input.

### 🎁 Additional Outputs

- **processed_images**: `np.ndarray` - Preprocessed input images with shape `(N, H, W, 3)` in RGB format (0-255 uint8).
- **aux**: `dict` - Auxiliary outputs including:
  - `feat_layer_X`: Intermediate features from layer X (if `export_feat_layers` was specified)
  - `gaussians`: 3D Gaussian Splats data (if `infer_gs=True`)

### 💻 Usage Example

```python
prediction = model.inference(image=["img1.jpg", "img2.jpg"])

# Access depth maps
depth_maps = prediction.depth  # shape: (2, H, W)

# Access confidence
if hasattr(prediction, 'conf'):
    confidence = prediction.conf

# Access camera parameters (if available)
if hasattr(prediction, 'extrinsics'):
    camera_poses = prediction.extrinsics  # shape: (2, 4, 4)

if hasattr(prediction, 'intrinsics'):
    camera_intrinsics = prediction.intrinsics  # shape: (2, 3, 3)

# Access intermediate features (if export_feat_layers was set)
if hasattr(prediction, 'aux') and 'feat_layer_0' in prediction.aux:
    features = prediction.aux['feat_layer_0']
```
