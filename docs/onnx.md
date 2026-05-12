# ONNX inference for Depth Anything 3

This page covers running inference with [`onnxruntime`][ort] instead of
PyTorch, so you don't need `torch` / `torchvision` (or `xformers`, `e3nn`, ...)
installed at deployment time.

The PyTorch path is unchanged — see the main [README](../README.md). All the
files involved here live under
[`src/depth_anything_3/onnx/`](../src/depth_anything_3/onnx).

[ort]: https://onnxruntime.ai

---

## 1. What's supported

| Variant | ONNX export | ONNX inference | Notes |
| --- | --- | --- | --- |
| `da3-small` / `-base` / `-large` / `-giant` | ✅ | ✅ | depth + camera (+ optional sky) |
| `da3metric-*` | ✅ | ✅ | same architecture as single-net |
| `da3nested-*` | ✅ (two files) | ✅ | metric alignment is done on the host in NumPy |
| `infer_gs=True` (3DGS branch) | ❌ | ❌ | depends on `gsplat`; not ONNX-compatible |
| `export_feat_layers` | baked-in at export time | ✅ | choose the layers at export, not at infer time |

Runtime execution providers reachable today: **CPU**, **CUDA** (via
`onnxruntime-gpu`). The wrapper passes through `providers=[…]` unchanged, so
**TensorRT** works the moment you install an `onnxruntime-gpu` build with the
`TensorrtExecutionProvider` and pass `--providers tensorrt,cuda,cpu`.

---

## 2. Install

You need **two** installs: one for *exporting* the model (requires torch),
one for *running* it (only `onnxruntime`).

### 2a. CPU-only (minimal install — no torch)

```bash
# in a fresh venv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .[onnx]
```

What this gives you: `numpy`, `opencv-python-headless`, `pillow`,
`imageio`, `huggingface_hub`, `onnxruntime`, `tqdm`, `addict`, `typer`.
That's it — **no** `torch`, `torchvision`, `xformers`, `e3nn`, `evo`,
`safetensors`, `omegaconf`, `pre-commit`, `fastapi`, `uvicorn`,
`requests`, `pycolmap`, `trimesh`, or `matplotlib`. With the minimum
you can run the model and save raw depth as a `.npz`
(`--export-format mini_npz`).

The OpenCV dep is intentionally `opencv-python-headless`, the no-GTK/Qt
build. The codebase has zero `cv2.imshow` / `waitKey` / `namedWindow`
calls, so the headless build covers every cv2 call site (preprocess
resize, postprocess resize, video frame I/O, benchmark datasets). If
you actually need the GUI build (e.g. for an interactive demo outside
of this repo), `pip install opencv-python` replaces the headless wheel
in place.

Sizes measured on Python 3.10 with `uv 0.6.6` (fresh venv → disk):

| Install spec | Total venv size | What you get |
| --- | --- | --- |
| `.[onnx]`                 | **332 MB** | numpy mini_npz export, raw depth |
| `.[onnx,viz]`             | **~342 MB** | + matplotlib → `depth_vis` |
| `.[onnx,viz,glb]`         | **~386 MB** | + trimesh → `glb` export |
| `.[onnx,viz,glb,colmap]`  | **~410 MB** | + pycolmap → `colmap` export |
| `.[onnx-gpu]`             | depends on CUDA wheel | GPU runtime |

For reference, before this slim-down a `.[onnx]` install pulled in
~600 MB of deps because it included the entire `[torch]` indirect set
(`safetensors`, `omegaconf`, `evo`, `onnx` itself, etc.) and the full
backend HTTP stack (`fastapi`, `uvicorn`, `requests`).

Where the remaining 332 MB goes:

| Package | Size | Why |
| --- | ---: | --- |
| `opencv_python_headless.libs/` + `cv2/` | 137 MB | image resize — single biggest line |
| `numpy.libs/` + `numpy/` | 65 MB | numerics |
| `onnxruntime` | 49 MB | inference runtime |
| `sympy` | 30 MB | transitive from `onnxruntime` (symbolic shape inference) |
| `pillow.libs/` + `PIL/` | 20 MB | image loading |
| `hf_xet` | 11 MB | HF Hub fast-download backend |
| everything else (typer, tqdm, etc.) | ~20 MB | CLI + small utilities |

Add the visualization extras when you want them:

```bash
pip install -e .[onnx,viz]     # depth_vis colour-mapped depth output
pip install -e .[onnx,viz,glb] # GLB point-cloud + camera wireframes
pip install -e .[onnx,colmap]  # COLMAP scene export
```

### 2b. CUDA inference

```bash
uv pip install -e .[onnx-gpu]
```

`onnxruntime-gpu` needs your CUDA toolkit version to match its build's
requirement (see [the ORT compatibility matrix][ort-cuda]). Once installed,
you can pass `--providers cuda,cpu` to the CLI or
`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` to the Python
API.

[ort-cuda]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

### 2c. Exporting the model (needs torch)

The export tool turns the torch checkpoint into one (or two, for nested) ONNX
files. You only need to do this **once per architecture**, then ship the
`.onnx` to the deployment environment that has only `[onnx]` installed.

```bash
# In a venv that can install torch (e.g. CPU-only):
uv pip install -e .[torch,onnx] --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2d. Everything-in-one

```bash
uv pip install -e .[all]   # torch + onnx-gpu + viz + glb + colmap + backend + app + gs + bench
```

---

## 3. Export a model to ONNX

The DA3 backbone uses Python-int shape arithmetic for the positional
embedding interpolation and the DPT upsample target sizes; TorchScript
bakes those values as constants at trace time. We therefore **export at a
fixed square shape** and rely on Python pre/postprocessing to handle
arbitrary input resolutions at inference (same approach as
[MoonCodeMaster/Depth-Anything-3-Onnx][moon]).

[moon]: https://github.com/MoonCodeMaster/Depth-Anything-3-Onnx

```bash
# Single-net (e.g. DA3-LARGE, DA3METRIC-LARGE):
da3 onnx-export depth-anything/DA3METRIC-LARGE \
    --out exports/da3metric-large_504.onnx \
    --views 1 --height 504 --width 504 \
    --opset 17 --device cpu

# Higher-resolution variant (more accurate, slower):
da3 onnx-export depth-anything/DA3-LARGE \
    --out exports/da3-large_728.onnx \
    --views 1 --height 728 --width 728 \
    --opset 17 --device cpu

# Nested (e.g. DA3NESTED-GIANT-LARGE-1.1):
da3 onnx-export depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --out exports/da3nested \
    --nested \
    --views 1 --height 504 --width 504 \
    --opset 17 --device cpu
# Writes exports/da3nested.main.onnx + exports/da3nested.metric.onnx
```

The exported graph's `image` input has shape `[1, N, 3, S, S]` where:
- `S` (H and W) is fixed at the value you passed via `--height/--width`.
- `N` (number of views) is dynamic — one ONNX file can take 1, 2, ... N
  views without re-export.

Useful flags:
- `--with-camera` — bake `extrinsics` / `intrinsics` as additional inputs.
  Export twice (with and without) if you need both modes.
- `--use-ray-pose` — bake `use_ray_pose=True`.
- `--ref-view-strategy {saddle_balanced,first,middle,saddle_sim_range}` —
  baked at export time, since the backbone uses it via Python control flow.
- `--views N --height H --width W` — trace shape. **`H` and `W` are
  pinned**, so pick a value that matches what you'll use at inference.
  Common picks: `504` (= 36×14), `672` (= 48×14), `728` (= 52×14). Must be
  a multiple of `PATCH_SIZE = 14`.

Tracing happens via the legacy TorchScript-based exporter (`dynamo=False`)
because the backbone has data-dependent control flow that the new
`dynamo`-based exporter cannot trace. You'll see a wall of
`TracerWarning: Converting a tensor to a Python boolean ...` — these are
expected and harmless at the trace shape.

---

## 4. Run ONNX inference

The runtime path uses `process_res_method="upper_bound_resize_padded"`
by default. This is **literally `upper_bound_resize` plus a padding
step** — the image goes through the exact same aspect-preserving resize
that the DA3 torch path's `upper_bound_resize` mode uses; the only
addition is filling the rest of the canvas with neutral gray so the
tensor matches the fixed-square ONNX input shape.

For each input image:

1. **Aspect-preserving resize** so the longest side becomes the
   ONNX trace size `S` (e.g. 504). The shorter side is rounded down to
   a multiple of 14. — *Identical to `upper_bound_resize`.*
2. **Pad with the ImageNet mean color** (≈ gray 123, 117, 104) to fill
   the canvas to `(S, S)`. After ImageNet normalization these padded
   pixels become 0, so they look maximally neutral to the model — no
   aspect distortion, no fake edges. — *The only new step.*
3. **Run ONNX** at the fixed `(S, S)` shape.
4. **Crop** the un-padded `(scaled_h, scaled_w)` region from the depth
   output to discard the gray padding bars.
5. **Resize the crop back** to the input's original `(H, W)` with
   `cv2.INTER_LINEAR`. The sky mask is resized with `INTER_NEAREST`;
   intrinsics, if the graph emits them, are inverse-scaled and
   inverse-translated to the original pixel grid.

So you can think of `upper_bound_resize_padded` as `upper_bound_resize`
wrapped in `pad → run → crop → resize-back`. The pixels the model sees
in the un-padded region are the same pixels `upper_bound_resize` would
have fed it; the user receives back a depth map at the original input
resolution with no trace of the padding.

The aspect-distorting `square_resize` mode (matching the reference
[MoonCodeMaster][moon] implementation) is still available with
`--process-res-method square_resize`. It squashes the image to
`(S, S)` and resizes the depth back; faster and simpler but accuracy
suffers on non-square inputs because the model wasn't trained on
distorted aspect ratios.

```bash
# Single-net:
da3 onnx assets/examples/SOH/000.png \
    --model exports/da3metric-large_504.onnx \
    --providers cuda,cpu \
    --process-res 504 \
    --export-format depth_vis-mini_npz \
    --export-dir workspace/onnx/scene

# Nested-metric (pass both files; the API does the alignment on the host):
da3 onnx assets/examples/SOH/000.png \
    --model exports/da3nested.main.onnx \
    --metric-model exports/da3nested.metric.onnx \
    --providers cuda,cpu \
    --process-res 504 \
    --export-format glb \
    --export-dir workspace/onnx/scene
```

**Important:** the value of `--process-res` must match the H/W you
passed when running `da3 onnx-export`. If they disagree onnxruntime
errors out with `Got invalid dimensions for input: image`.

GLB export requires the model to emit camera intrinsics and
extrinsics. `DA3METRIC-LARGE` (single-net metric) does **not** produce
those, so use `depth_vis`, `mini_npz`, or `npz` for that model. Use
the non-metric `DA3-LARGE` / `DA3-GIANT` if you need GLB.

From Python:

```python
from depth_anything_3.onnx import DepthAnything3Onnx, DepthAnything3OnnxNested

# Single-net. `process_res` MUST match the trace shape of the .onnx.
runner = DepthAnything3Onnx(
    "exports/da3metric-large_504.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
prediction = runner.inference(
    image=["assets/examples/SOH/000.png", "assets/examples/SOH/010.png"],
    process_res=504,
    process_res_method="upper_bound_resize_padded",   # default; or "square_resize"
    export_dir="workspace/onnx/scene",
    export_format="mini_npz",
)
# Depth is back at the original (H, W) per image when all inputs share
# the same size, else an object array of per-image arrays.
print(prediction.depth.shape)            # e.g. (2, 680, 1208)

# Use unsquare=False if you want the raw SxS output (e.g. for parity).
prediction_native = runner.inference(
    image=["assets/examples/SOH/000.png"],
    process_res=504,
    unsquare=False,
)
print(prediction_native.depth.shape)     # (1, 504, 504)

# Nested-metric
nested = DepthAnything3OnnxNested(
    "exports/da3nested.main.onnx",
    "exports/da3nested.metric.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

The `Prediction` dataclass and all numpy-side exporters (`mini_npz`, `npz`,
`glb`, `depth_vis`, `colmap`) work exactly like in the torch path. The GS
exporters (`gs_ply`, `gs_video`, `feat_vis`) still require torch — they are
not reachable from the ONNX inference path.

---

## 5. Parity: ONNX vs torch on the same images

A built-in command runs both paths on the same images and emits diff
statistics + side-by-side depth visualizations:

```bash
da3 onnx-parity depth-anything/DA3METRIC-LARGE \
    --image assets/examples/SOH/000.png \
    --image assets/examples/SOH/010.png \
    --onnx exports/da3metric-large_504.onnx \
    --device cpu --providers cpu \
    --process-res 504 \
    --fail-threshold 0.2 \
    --out-dir workspace/parity-square
```

If `--onnx` is omitted, the tool exports a fresh `.onnx` to a temp dir
before comparing. `--out-dir` is what makes the tool write the visuals;
omit it for numbers-only.

Internally the parity tool runs both paths at the model's native
`(S, S)` canvas, **then crops the padding and resizes the un-padded
region back to each image's original `(H, W)` on both sides**. So the
compare images don't show the gray padding bars; you see what the user
of `da3 onnx` actually receives back. The crop+resize is symmetric on
both sides, so it doesn't inflate the parity diff.

### Example: DA3METRIC-LARGE @ 504×504, CPU/CPU

Stat summary printed to stdout:

```
== Parity report ==
  model        : depth-anything/DA3METRIC-LARGE
  onnx         : exports/da3metric-large_504.onnx
  images       : ['assets/examples/SOH/000.png', 'assets/examples/SOH/010.png']
  process_res  : 504 (upper_bound_resize_padded)
  providers    : ['cpu']
  visuals      : workspace/parity-square
  fields:
  depth   shape=(2, 504, 504) -> (2, 504, 504)  max|d|=7.06  mean|d|=0.0051  max-rel=0.71  allclose(1e-2)=False
  sky     shape=(2, 504, 504) -> (2, 504, 504)  max|d|=1.00  mean|d|=5.9e-06 max-rel=1e+08  allclose(1e-2)=False
✅ Parity OK (depth max-rel=0.71  -- but mean|d| = 5 mm on 7 m scene)
```

The model code is **identical** between the torch and ONNX paths now —
the only differences are fp32 rounding inside ORT's kernels and
fp32-vs-bicubic interpolation inside the model. So:

- **mean depth diff = 5 mm** on a scene that spans 2.8 m to 10 m.
- **sky agreement: 99.9994% of pixels match** (5.9e-6 mean abs diff on a
  boolean mask).
- The reported `max|d| ≈ 7 m` and `max-rel = 70 %` come from a tiny
  number of sky-boundary pixels where the sky-cap fires slightly
  differently between paths.

Timings on the same machine (CPU, 2 views @ 504×504):

| Path | Forward time |
| --- | --- |
| Torch (`DepthAnything3.inference`) | ~39 s |
| ONNX (`CPUExecutionProvider`) | ~10 s |

ONNX is ~4× faster at this resolution on CPU without any quantization.

How to read the numbers:
- **`mean|d| ≈ 4e-3` on depth** means the two paths agree in the bulk to
  better than half a percent. `max|d|` and `max-rel` are dominated by a
  handful of pixels — typically at sky / sky-boundary regions where the
  torch post-processing (`_process_mono_sky_estimation`) and our NumPy
  port may disagree by one pixel due to rounding inside
  `np.quantile` vs `torch.quantile`.
- **`sky` is a boolean mask**, so any disagreement registers as
  `|diff| = 1`. `mean|d| = 4.2e-05` means **0.004% of pixels disagree** —
  again at boundary pixels. The reported `max-rel = 1e+08` is the
  `1 / max(true, 1e-8)` floor; ignore it for binary masks.
- `allclose(1e-2)` is numpy's default. ORT-vs-torch parity at fp32 is
  typically a few percent on outlier pixels but well within 1% in the
  bulk. Use `--fail-threshold 0.2` if you want a green pass.

### Artifacts written to `--out-dir`

For each input image `i` (e.g. `000.png`, `010.png`):

| File | What it is |
| --- | --- |
| `{i:03d}_{stem}_input.jpg`   | Pre-processed input (ImageNet-denormalized, what the model sees) |
| `{i:03d}_{stem}_torch.jpg`   | Torch depth visualization (Spectral colormap) |
| `{i:03d}_{stem}_onnx.jpg`    | ONNX depth visualization on the **same** color scale as torch |
| `{i:03d}_{stem}_diff.jpg`    | `abs(torch_depth - onnx_depth)`, magma cmap, clipped at its 99th percentile |
| `{i:03d}_{stem}_compare.png` | Annotated comparison plot (see below) |

The `*_compare.png` is the headline artifact. Layout:

- **Row 1** — input · torch depth · onnx depth, sharing a single Spectral
  colorbar on the right. Both depth panels use the same `vmin`/`vmax`, so
  if you can see a difference by eye it's a real one.
- **Row 2** — wide `|Δdepth|` panel (magma cmap) with its own colorbar.
  Color is clipped at the 99th-percentile error so the bulk of the image
  isn't washed out by a couple of outlier pixels; the panel title shows
  both the clip value and the true max.
- **Suptitle** — depth-range context + `mean / max / p99` of `|Δdepth|`
  in the same units, plus the mean as a percentage of the depth span
  ("`mean ≈ 0.07% of depth span`"). This is the single best number to
  cite when somebody asks "is the ONNX one as accurate?".

**Units.** The CLI picks the unit label automatically:
- `--metric metric`   → label is `m` (meters). Use this for `DA3METRIC-*`
  and `DA3NESTED-*` (the "metric" branch makes the depth absolute).
- `--metric relative` → label is `depth-units` (relative / scale-free).
- `--metric auto` (default) → metric if the model id contains "metric"
  or "nested" *or* if `Prediction.is_metric == 1`; relative otherwise.

  Note: a *single-net* `DA3METRIC-LARGE` produces metric depth in meters,
  but the torch `forward()` doesn't set `is_metric=1` on the returned
  prediction (only the nested forward does), so the auto-detect falls
  back to the model-name heuristic. Pass `--metric metric` explicitly if
  you ever need to be certain.

### Sample numbers from the run above

`DA3METRIC-LARGE`, 2 views @ 504×504, CPU/CPU (model native shape, no
resize-back applied):

| View | depth range (m) | `mean|Δ|` (m) | `max|Δ|` (m) | `p99|Δ|` (m) | mean / depth span |
| --- | --- | --- | --- | --- | --- |
| 000 (SOH 000.png) | ~[2.79, 9.97] | 0.0056 | 7.06 | 0.0094 | 0.08 % |
| 010 (SOH 010.png) | ~[2.79, 9.97] | 0.0046 | 0.83 | 0.0090 | 0.06 % |

Read: at every pixel the two paths agree to within ~5 mm on average for
a scene that spans ~7 m, and ~9 mm at the 99th percentile. The "max" of
~7 m on view 000 is one sky-boundary pixel where the sky-cap fires
slightly differently between paths.

The raw arrays in `arrays.npz` let you reproduce these numbers offline:

```python
import numpy as np
arr = np.load("workspace/parity-onnx/arrays.npz")
print("metric?", bool(arr["is_metric"]))
diff = arr["abs_diff"]
print("mean", diff.mean(), "max", diff.max(), "p99", np.percentile(diff, 99))
```

Plus a single `arrays.npz` at the root of the output dir containing the raw
arrays for quantitative inspection:

```python
import numpy as np
arr = np.load("workspace/parity-onnx/arrays.npz")
arr["torch_depth"]   # (N, H, W) float32
arr["onnx_depth"]    # (N, H, W) float32
arr["abs_diff"]      # (N, H, W) float32
arr["torch_sky"]     # (N, H, W) bool   (if the model has a sky head)
arr["onnx_sky"]      # (N, H, W) bool
```

### Visual reading guide

In the `*_compare.jpg` mosaic, the torch and ONNX panels should be visually
indistinguishable. The diff panel should be dominated by black (low
difference), with brighter pixels concentrated at:

1. Sky / non-sky transitions (the strongest source of disagreement).
2. Object silhouettes (a one-pixel boundary shift between the two paths).
3. Saturated bright regions (when depth becomes very large, fp32 rounding
   amplifies relative error).

If you see a structurally different depth map (a different object, a flipped
sign, a posterized colormap), that's a real bug — not numerical noise.

Python API:

```python
from depth_anything_3.onnx.parity import compare_torch_vs_onnx

report = compare_torch_vs_onnx(
    model="depth-anything/DA3METRIC-LARGE",
    images=["assets/examples/SOH/000.png", "assets/examples/SOH/010.png"],
    onnx_path="exports/da3metric-large.onnx",  # skip re-export
    process_res=252,
    device="cpu",
    providers=["cpu"],
    out_dir="workspace/parity-onnx",            # write the visuals
)
print(report.summary())
for diff in report.diffs:
    print(diff.name, diff.mean_abs, diff.max_rel)
```

---

## 6. Notes / caveats

- **Sky post-processing is on the host.** The torch model has an `if
  non_sky_mask.sum() <= 10:` guard in `_process_mono_sky_estimation` that
  doesn't survive `torch.export` / `torch.jit.trace`. The ONNX graph emits
  the raw `depth` and `sky` heads; we replicate the sky-aware depth cap in
  NumPy (`depth_anything_3.onnx.postprocess.process_mono_sky`). Same idea
  for nested-metric: the alignment / least-squares scale / sky handling are
  reproduced in `apply_nested_metric_alignment`.
- **Fixed-square graph, dynamic N.** The exported `image` input has shape
  `[1, N, 3, S, S]` where `S` is fixed at trace time and `N` is free.
  Different image resolutions are handled in Python around the model
  rather than inside it: `upper_bound_resize_padded` aspect-preserving
  resizes any input to fit in (S, S) and pads the rest with neutral
  gray; the depth output is then cropped (to discard the pad) and
  resized back to the original (H, W). The cheaper alternative
  `square_resize` skips the pad/crop and directly squashes to (S, S)
  on the way in / stretches back on the way out — matching the
  reference [MoonCodeMaster/Depth-Anything-3-Onnx][moon] repo.

  Implications:
  - **Different aspect ratios in one batch are fine.** When all images
    share the same original (H, W), the returned `depth` / `conf` / `sky`
    are regular `(N, H, W)` ndarrays. When they differ, those fields
    become object arrays (one element per image, each at its own
    `(H_i, W_i)`).
  - **Aspect distortion is real.** A 16:9 image becomes 1:1 inside the
    model; thin vertical features get squashed slightly. For depth
    estimation this is usually fine (DA3 was trained on diverse aspect
    ratios), but if you need pixel-perfect alignment with the *processed*
    SxS input, pass `unsquare=False` in the Python API.
  - **`process_res` must match the trace shape.** Loading
    `da3metric-large_504.onnx` requires `process_res=504`. ORT errors out
    otherwise: `Got invalid dimensions for input: image`. Pick a single
    `S` per deployed model.

[moon]: https://github.com/MoonCodeMaster/Depth-Anything-3-Onnx
- **`infer_gs=True` raises.** The ONNX path raises a clear
  `NotImplementedError` if you pass `infer_gs=True` or request a `gs_*`
  export format. Use the torch path for Gaussian Splatting.
- **xformers is optional.** DinoV2 falls back to
  `F.scaled_dot_product_attention` when xformers is missing; the ONNX graph
  doesn't reference xformers.
- **Where the ONNX file lives.** The `--model` argument accepts either a
  local path (`./da3-large.onnx`) or a HuggingFace Hub spec
  (`user/repo:filename.onnx`).

---

## 7. CLI reference

```
da3 onnx-export <model_id_or_path>
    --out <path.onnx>
    [--nested --out-main <path> --out-metric <path>]
    [--with-camera] [--use-ray-pose] [--ref-view-strategy STRATEGY]
    [--views N] [--height H] [--width W] [--opset 17] [--device cuda|cpu]

da3 onnx <input>
    --model <path.onnx>
    [--metric-model <path.onnx>]
    [--providers cuda,cpu]
    [--export-dir DIR] [--export-format FORMAT]
    [--process-res 504] [--process-res-method upper_bound_resize]
    [--conf-thresh-percentile 40] [--num-max-points 1000000] [--show-cameras]

da3 onnx-parity <model_id_or_path>
    --image <img> [--image <img> ...]
    [--onnx <existing.onnx>]
    [--with-camera] [--use-ray-pose] [--ref-view-strategy STRATEGY]
    [--process-res 504] [--providers cuda,cpu] [--device cuda|cpu]
    [--fail-threshold 0.05]
    [--out-dir DIR]    # write side-by-side depth visualizations + arrays.npz
    [--metric auto|metric|relative]   # unit label for the compare plot
```

---

## 8. Sanity-tested combinations

| Model | Device | Provider | Trace shape | Inference shape | Result |
| --- | --- | --- | --- | --- | --- |
| `DA3METRIC-LARGE` | CPU (torch 2.11 + ORT 1.23) | `CPUExecutionProvider` | `(1, 1, 3, 504, 504)` | any (via `upper_bound_resize_padded`) | depth mean\|d\|=5 mm at original resolution, sky agreement 99.999%; ONNX ~4× faster than torch on CPU |

To reproduce:

```bash
da3 onnx-export depth-anything/DA3METRIC-LARGE \
    --out exports/da3metric-large_504.onnx \
    --views 1 --height 504 --width 504 --device cpu

# Run on a single image at its native resolution -- the model sees a
# squashed 504x504, the depth is resized back to the original
# (1208x680) before being saved.
da3 onnx assets/examples/SOH/000.png \
    --model exports/da3metric-large_504.onnx \
    --providers cpu --process-res 504 \
    --export-format depth_vis-mini_npz \
    --export-dir workspace/onnx/scene --auto-cleanup

# Parity check: both paths run at the model's native 504x504 (no
# resize-back), so we measure model-quality drift only.
da3 onnx-parity depth-anything/DA3METRIC-LARGE \
    --image assets/examples/SOH/000.png \
    --image assets/examples/SOH/010.png \
    --onnx exports/da3metric-large_504.onnx \
    --device cpu --providers cpu \
    --process-res 504 \
    --fail-threshold 0.2 \
    --out-dir workspace/parity-square
```

The two source images in `assets/examples/SOH/` are landscape Opera House
photos with sky in the top portion — useful because they exercise the
sky-cap post-processing path in both runtimes, and verify the
square-resize + unsquare pipeline handles non-square inputs correctly.
