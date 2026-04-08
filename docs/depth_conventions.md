# Depth Anything V2 vs V3: Depth Convention Reference

## Model Output Conventions

| Property | V2 (HuggingFace Transformers) | V3 (Official Repo) |
|---|---|---|
| **Output type** | Disparity (inverse depth) | Metric depth |
| **Convention** | Large value = near, small = far | Small value = near, large = far |
| **Head activation** | `ReLU` (non-negative disparity) | `exp()` (positive depth) |
| **Value range** | Arbitrary relative units (e.g. 41–599) | Metric-like scale (e.g. 0.7–1.2) |
| **Resolution** | Matches input (interpolated via bicubic) | Model resolution (e.g. 280×504) |
| **Raw format saved** | `.npy` float32 | `.npy` float32 |
| **Model name** | `depth-anything/Depth-Anything-V2-Large-hf` | `depth-anything/DA3-LARGE` |

## Inference Pipeline

| Step | V2 | V3 |
|---|---|---|
| **Load** | `AutoModelForDepthEstimation.from_pretrained(...)` | `DepthAnything3.from_pretrained(...)` |
| **Preprocess** | `AutoImageProcessor` (resize + normalize) | `InputProcessor` (resize + normalize + intrinsics) |
| **Forward** | `model(**inputs).predicted_depth` | `model.inference(image=[...]).depth` |
| **Postprocess** | `F.interpolate(..., mode="bicubic")` to original size | Already at model resolution; resize if needed |
| **Raw output** | `depth[i] = prediction.squeeze().numpy()` → **disparity** | `prediction.depth[i]` → **metric depth** |

## Converting to a Common Representation

To compare V2 and V3 on equal footing, convert V2 disparity to depth:

```
V2 disparity (large=near)  →  depth = 1 / disparity  →  depth (small=near)
V3 depth (small=near)      →  no conversion needed
```

| | V2 (raw) | V2 (converted) | V3 (raw) |
|---|---|---|---|
| **Representation** | Disparity | Depth | Depth |
| **Near objects** | Large values | Small values | Small values |
| **Far objects** | Small values | Large values | Large values |

## Visualization Pipeline

The official V3 `visualize_depth()` function works on **depth** (small=near) values:

```
depth → 1/depth (disparity) → percentile normalize [0,1] → flip (1-x) → Spectral colormap
```

| Step | Operation | Purpose |
|---|---|---|
| 1. Invert | `disp = 1 / depth` | Convert to disparity for better near-range resolution |
| 2. Normalize | `(disp - p2) / (p98 - p2)` | Robust [0,1] range using 2nd/98th percentiles |
| 3. Flip | `1 - normed` | Align so near→high colormap value (warm/red in Spectral) |
| 4. Colormap | `matplotlib Spectral(value)` | Near = red/warm, Far = blue/cool |

### Applying to Both Models

After converting V2 disparity → depth (step above), the **same** visualization function handles both:

```python
# V2: convert first
v2_depth = 1.0 / v2_disparity    # now small=near like V3

# Both use identical colormap path
v2_vis = depth_to_colormap(v2_depth, cmap="Spectral")
v3_vis = depth_to_colormap(v3_depth, cmap="Spectral")
```

## Common Pitfalls

| Mistake | Why it's wrong |
|---|---|
| Treating V2 output as depth | V2 outputs disparity — near/far meaning is reversed |
| Passing V2 disparity to V3's `visualize_depth()` | That function does `1/input` internally; applying it to disparity gives back depth, then flips it — resulting in **inverted** colors for V2 |
| Extracting grayscale from a Spectral-colored image | Spectral is non-monotonic in luminance (red→yellow→green→blue→purple); grayscale destroys the depth ordering |
| Comparing pre-rendered colormaps directly | V2 default uses INFERNO, V3 uses Spectral — different scales make visual comparison meaningless |

## Script Reference

| Script | Purpose |
|---|---|
| `process_video_dav2.py` | Run V2 inference on video frames, save raw `.npy` disparity + INFERNO vis |
| `test_v3_depth_output.py` | Run V3 inference on images, save raw `.npy` depth + official vis |
| `compare_v2_v3_depth.py` | **Correct** side-by-side comparison using raw `.npy` from both models |
