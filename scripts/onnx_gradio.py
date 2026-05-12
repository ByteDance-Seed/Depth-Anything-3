#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Standalone Gradio app: ONNX inference + torch-vs-ONNX parity.

A single-file app for quickly testing the ONNX path of Depth Anything 3:

  Tab 1 (Inference)
      • Upload an image.
      • Pick an ONNX file (default: exports/da3metric-large_504.onnx).
      • Optionally supply camera intrinsics (fx, fy, cx, cy).
      • Run → colored depth preview + raw stats + .npz download.

  Tab 2 (Parity)
      • Upload an image.
      • Pick a HF model id (torch side) + the matching ONNX file.
      • Optionally supply intrinsics.
      • Run → side-by-side torch vs ONNX depth with |Δdepth| panel and stats.

The Gradio app itself only depends on the ``[onnx]`` + ``[viz]`` extras for
the inference tab. The parity tab additionally requires the ``[torch]``
extras; until you press the button on that tab, torch is not imported.

Usage::

    pip install -e ".[onnx,viz,glb]"     # for inference + GLB
    pip install gradio                    # the app itself
    python scripts/onnx_gradio.py

To open on a different port: ``--port 7861``. To make it reachable from
another machine: ``--host 0.0.0.0``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# Make the source tree importable without installing the package as editable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if (_REPO_ROOT / "src" / "depth_anything_3").is_dir():
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ----------------------------------------------------------------------
# Lazy imports so launching the UI is fast and doesn't error out if
# optional bits (torch, matplotlib) aren't installed.
# ----------------------------------------------------------------------
def _build_intrinsics(use_K: bool, fx: float, fy: float, cx: float, cy: float):
    """Build an (1, 3, 3) numpy intrinsics array, or None if the user
    didn't tick the checkbox.
    """
    if not use_K:
        return None
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = float(cx)
    K[1, 2] = float(cy)
    return K[None]  # (N=1, 3, 3)


def _colormap_depth(depth: np.ndarray) -> np.ndarray:
    """Reuse the project's `visualize_depth` if matplotlib is installed,
    else fall back to a simple grayscale rendering.
    """
    try:
        from depth_anything_3.utils.visualize import visualize_depth

        return visualize_depth(depth).astype(np.uint8)
    except Exception:
        d = depth.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            inv = 1.0 / d[valid]
            lo, hi = float(np.percentile(inv, 2)), float(np.percentile(inv, 98))
        else:
            lo, hi = 0.0, 1.0
        with np.errstate(divide="ignore"):
            inv = np.where(valid, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        norm = np.clip((inv - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        return (norm * 255).astype(np.uint8)


def _summarize_depth(depth: np.ndarray) -> str:
    finite = np.isfinite(depth) & (depth > 0)
    if not finite.any():
        return "no valid depth pixels"
    d = depth[finite]
    return (
        f"shape={depth.shape}  "
        f"min={d.min():.3f}  max={d.max():.3f}  "
        f"mean={d.mean():.3f}  median={np.median(d):.3f}  "
        f"p2={np.percentile(d, 2):.3f}  p98={np.percentile(d, 98):.3f}"
    )


# ----------------------------------------------------------------------
# Tab 1: ONNX inference
# ----------------------------------------------------------------------
def run_inference(
    image,                        # PIL.Image (Gradio "image" component)
    onnx_path: str,
    providers: str,
    process_res: int,
    process_res_method: str,
    use_K: bool,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """Returns (depth_vis_img, stats_text, npz_filepath, status_text)."""
    if image is None:
        return None, "Upload an image first.", None, "❌ No image"
    if not onnx_path or not os.path.exists(onnx_path):
        return None, f"ONNX file not found: {onnx_path!r}", None, "❌ Bad path"

    try:
        from depth_anything_3.onnx import DepthAnything3Onnx
    except ImportError as e:
        return None, f"Missing deps: {e}", None, "❌ Install [onnx]"

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    intrinsics = _build_intrinsics(use_K, fx, fy, cx, cy)

    t0 = time.time()
    try:
        runner = DepthAnything3Onnx(onnx_path, providers=provider_list)
        prediction = runner.inference(
            image=[np.asarray(image)],
            intrinsics=intrinsics,
            process_res=int(process_res),
            process_res_method=process_res_method,
            export_dir=None,
        )
    except Exception as e:
        return None, traceback.format_exc(), None, f"❌ Runtime error: {e}"
    elapsed = time.time() - t0

    depth = prediction.depth
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3:
        depth = depth[0]  # take first image

    depth_vis = _colormap_depth(depth)
    stats = (
        f"providers used: {runner.session.providers}\n"
        f"elapsed: {elapsed:.2f}s\n"
        f"depth: {_summarize_depth(depth)}\n"
    )
    if prediction.sky is not None:
        sky = prediction.sky
        if sky.ndim == 4 and sky.shape[0] == 1:
            sky = sky[0]
        if sky.ndim == 3:
            sky = sky[0]
        stats += f"sky pixels: {sky.sum()} / {sky.size} ({100 * sky.mean():.2f}%)\n"
    if intrinsics is not None:
        stats += f"input K (1 view): fx={fx} fy={fy} cx={cx} cy={cy}\n"
    if prediction.intrinsics is not None:
        stats += f"output intrinsics shape: {prediction.intrinsics.shape}\n"

    # Save raw arrays for download
    tmp = tempfile.NamedTemporaryFile(prefix="da3_onnx_", suffix=".npz", delete=False)
    np.savez_compressed(
        tmp.name,
        depth=depth.astype(np.float32),
        depth_vis=depth_vis,
        intrinsics=intrinsics if intrinsics is not None else np.zeros(0),
    )
    return depth_vis, stats, tmp.name, f"✅ {elapsed:.2f}s on {runner.session.providers[0]}"


# ----------------------------------------------------------------------
# Tab 2: parity
# ----------------------------------------------------------------------
def run_parity(
    image,
    model_id: str,
    onnx_path: str,
    providers: str,
    device: str,
    process_res: int,
    process_res_method: str,
    use_K: bool,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    """Returns (compare_img, stats_text, status_text)."""
    if image is None:
        return None, "Upload an image first.", "❌ No image"
    if not onnx_path or not os.path.exists(onnx_path):
        return None, f"ONNX file not found: {onnx_path!r}", "❌ Bad path"

    # Lazy: torch is only required for this tab.
    try:
        from depth_anything_3.onnx.parity import compare_torch_vs_onnx
    except ImportError as e:
        return None, f"Missing deps: {e}", "❌ Install [torch,onnx]"

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]

    # Save the image to a tmp file so the parity tool (which expects paths /
    # arrays) can pass it through to both backends without re-encoding.
    tmpdir = tempfile.mkdtemp(prefix="da3_parity_")
    out_dir = os.path.join(tmpdir, "viz")
    in_path = os.path.join(tmpdir, "input.png")
    image.save(in_path)

    _intrinsics_note = ""
    if use_K:
        # The compare_torch_vs_onnx() signature doesn't currently take
        # intrinsics through directly. They influence pose alignment only
        # when extrinsics are also supplied -- which the parity tool
        # doesn't do today. Surface this clearly rather than silently
        # ignoring the user's input.
        _intrinsics_note = (
            "\n[note] camera intrinsics are accepted but currently ignored on "
            "the parity path -- they only feed pose-alignment, which requires "
            "input extrinsics too. Use the Inference tab to verify that the "
            "intrinsics flow through to Prediction.intrinsics.\n"
        )

    t0 = time.time()
    try:
        report = compare_torch_vs_onnx(
            model=model_id,
            images=[in_path],
            onnx_path=onnx_path,
            with_camera=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
            process_res=int(process_res),
            process_res_method=process_res_method,
            providers=provider_list,
            device=device,
            keep_onnx=True,
            out_dir=out_dir,
        )
    except Exception as e:
        return None, traceback.format_exc(), f"❌ Runtime error: {e}"
    elapsed = time.time() - t0

    # The parity tool writes per-image PNGs; pick the only one.
    candidate = sorted(Path(out_dir).glob("*_compare.png"))
    compare_path = str(candidate[0]) if candidate else None
    summary = report.summary() + _intrinsics_note
    return compare_path, summary, f"✅ parity done in {elapsed:.1f}s"


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def build_ui(default_onnx: str | None):
    import gradio as gr

    with gr.Blocks(title="Depth Anything 3 — ONNX playground") as demo:
        gr.Markdown(
            "# Depth Anything 3 — ONNX playground\n"
            "Two tabs: **Inference** runs a single image through the ONNX model; "
            "**Parity** runs both the torch and ONNX paths and shows their diff. "
            "Camera intrinsics are an optional input to both."
        )

        # ----- Inference tab -----
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    inf_image = gr.Image(label="Input image", type="pil", height=320)
                    inf_onnx = gr.Textbox(
                        label="ONNX model path",
                        value=default_onnx or "",
                        placeholder="exports/da3metric-large_504.onnx",
                    )
                    with gr.Row():
                        inf_providers = gr.Textbox(
                            label="ORT providers (comma-separated)", value="cpu", scale=1
                        )
                        inf_res = gr.Number(
                            label="process_res (must match trace)", value=504, precision=0, scale=1
                        )
                    inf_method = gr.Dropdown(
                        label="process_res_method",
                        choices=["upper_bound_resize_padded", "square_resize"],
                        value="upper_bound_resize_padded",
                    )

                    with gr.Accordion("Camera intrinsics (optional)", open=False):
                        inf_use_K = gr.Checkbox(label="Provide intrinsics", value=False)
                        with gr.Row():
                            inf_fx = gr.Number(label="fx (px)", value=500.0)
                            inf_fy = gr.Number(label="fy (px)", value=500.0)
                        with gr.Row():
                            inf_cx = gr.Number(label="cx (px)", value=0.0)
                            inf_cy = gr.Number(label="cy (px)", value=0.0)
                        gr.Markdown(
                            "_Tip: leave cx/cy at 0 to default to image-center after "
                            "preprocessing. fx/fy are in pixels of the original image. Intrinsics are not used by the Metric Model_"
                        )

                    inf_btn = gr.Button("Run inference", variant="primary")
                    inf_status = gr.Markdown()

                with gr.Column(scale=1):
                    inf_out_img = gr.Image(label="Depth", height=320)
                    inf_stats = gr.Textbox(label="Stats", lines=10, max_lines=20)
                    inf_npz = gr.File(label="Raw arrays (.npz)")

            inf_btn.click(
                run_inference,
                inputs=[
                    inf_image, inf_onnx, inf_providers, inf_res, inf_method,
                    inf_use_K, inf_fx, inf_fy, inf_cx, inf_cy,
                ],
                outputs=[inf_out_img, inf_stats, inf_npz, inf_status],
            )

        # ----- Parity tab -----
        with gr.Tab("Parity (torch vs ONNX)"):
            with gr.Row():
                with gr.Column(scale=1):
                    par_image = gr.Image(label="Input image", type="pil", height=320)
                    par_model = gr.Textbox(
                        label="HF model id (torch side)",
                        value="depth-anything/DA3METRIC-LARGE",
                    )
                    par_onnx = gr.Textbox(
                        label="ONNX model path",
                        value=default_onnx or "",
                        placeholder="exports/da3metric-large_504.onnx",
                    )
                    with gr.Row():
                        par_providers = gr.Textbox(label="ORT providers", value="cpu", scale=1)
                        par_device = gr.Textbox(label="torch device", value="cpu", scale=1)
                    with gr.Row():
                        par_res = gr.Number(
                            label="process_res", value=504, precision=0, scale=1
                        )
                        par_method = gr.Dropdown(
                            label="process_res_method",
                            choices=["upper_bound_resize_padded", "square_resize"],
                            value="upper_bound_resize_padded",
                            scale=1,
                        )

                    with gr.Accordion("Camera intrinsics (optional)", open=False):
                        par_use_K = gr.Checkbox(label="Provide intrinsics", value=False)
                        with gr.Row():
                            par_fx = gr.Number(label="fx (px)", value=500.0)
                            par_fy = gr.Number(label="fy (px)", value=500.0)
                        with gr.Row():
                            par_cx = gr.Number(label="cx (px)", value=0.0)
                            par_cy = gr.Number(label="cy (px)", value=0.0)

                    par_btn = gr.Button("Run parity", variant="primary")
                    par_status = gr.Markdown()

                with gr.Column(scale=1):
                    par_compare = gr.Image(label="torch vs onnx compare", height=400)
                    par_stats = gr.Textbox(label="Diff stats", lines=14, max_lines=24)

            par_btn.click(
                run_parity,
                inputs=[
                    par_image, par_model, par_onnx, par_providers, par_device,
                    par_res, par_method,
                    par_use_K, par_fx, par_fy, par_cx, par_cy,
                ],
                outputs=[par_compare, par_stats, par_status],
            )

        gr.Markdown(
            "---\n"
            "Tips: \n"
            "- The ONNX export at `--height N --width N` *pins* the input "
            "shape, so `process_res` must equal `N`. \n"
            "- `upper_bound_resize_padded` preserves aspect ratio and pads "
            "with ImageNet-mean grey; depth is cropped + resized back to "
            "the original on the way out. `square_resize` directly squashes "
            "to NxN and stretches the depth back (faster, distorts).\n"
            "- For `DA3METRIC-LARGE` the ONNX graph has no camera head, so "
            "intrinsics are accepted by the API but only land in "
            "`Prediction.intrinsics` for downstream use."
        )

    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--onnx",
        default="exports/da3metric-large_504.onnx",
        help="Default ONNX file path shown in the UI",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Open a public Gradio tunnel")
    args = ap.parse_args()

    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        sys.exit("`gradio` is not installed.  pip install gradio")

    demo = build_ui(args.onnx if os.path.exists(args.onnx) else None)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
