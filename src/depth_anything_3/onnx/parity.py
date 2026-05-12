# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end parity check: torch DepthAnything3 vs onnxruntime DepthAnything3Onnx.

Loads a torch checkpoint, exports it to a temporary ONNX file, then runs both
paths on the same input images and reports per-output diff statistics.

Requires BOTH `[torch]` AND `[onnx]` extras.

Usage from Python::

    from depth_anything_3.onnx.parity import compare_torch_vs_onnx
    report = compare_torch_vs_onnx(
        model="depth-anything/DA3-LARGE-1.1",
        images=["assets/img1.jpg", "assets/img2.jpg"],
    )
    print(report.summary())

Usage from the CLI::

    da3 onnx-parity depth-anything/DA3-LARGE-1.1 --image assets/img1.jpg --image assets/img2.jpg
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass
class FieldDiff:
    """Per-field comparison summary."""

    name: str
    shape_torch: tuple
    shape_onnx: tuple
    max_abs: float
    mean_abs: float
    max_rel: float
    mean_rel: float
    allclose_rtol_1e_2: bool

    def __str__(self) -> str:
        return (
            f"  {self.name:14s} shape={self.shape_torch} -> {self.shape_onnx} "
            f"max|d|={self.max_abs:.4g}  mean|d|={self.mean_abs:.4g}  "
            f"max-rel={self.max_rel:.4g}  allclose(1e-2)={self.allclose_rtol_1e_2}"
        )


@dataclass
class ParityReport:
    """Comparison between the torch and ONNX predictions."""

    model: str
    onnx_path: str
    images: list[str]
    process_res: int
    process_res_method: str
    providers: list[str]
    diffs: list[FieldDiff] = field(default_factory=list)
    out_dir: str | None = None

    def summary(self) -> str:
        lines = [
            f"== Parity report ==",
            f"  model        : {self.model}",
            f"  onnx         : {self.onnx_path}",
            f"  images       : {self.images}",
            f"  process_res  : {self.process_res} ({self.process_res_method})",
            f"  providers    : {self.providers}",
        ]
        if self.out_dir:
            lines.append(f"  visuals      : {self.out_dir}")
        lines.append("  fields:")
        for d in self.diffs:
            lines.append(str(d))
        return "\n".join(lines)


def _diff(name: str, t: np.ndarray | None, o: np.ndarray | None) -> FieldDiff | None:
    if t is None or o is None:
        return None
    t = np.asarray(t, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    if t.shape != o.shape:
        # Try squeezing leading singleton dims so e.g. (1,N,H,W) vs (N,H,W) compares.
        t_sq = t.squeeze()
        o_sq = o.squeeze()
        if t_sq.shape == o_sq.shape:
            t, o = t_sq, o_sq
        else:
            return FieldDiff(
                name=name,
                shape_torch=tuple(t.shape),
                shape_onnx=tuple(o.shape),
                max_abs=float("nan"),
                mean_abs=float("nan"),
                max_rel=float("nan"),
                mean_rel=float("nan"),
                allclose_rtol_1e_2=False,
            )
    diff = np.abs(t - o)
    denom = np.maximum(np.abs(t), 1e-8)
    rel = diff / denom
    return FieldDiff(
        name=name,
        shape_torch=tuple(t.shape),
        shape_onnx=tuple(o.shape),
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        max_rel=float(rel.max()),
        mean_rel=float(rel.mean()),
        allclose_rtol_1e_2=bool(np.allclose(t, o, rtol=1e-2, atol=1e-2)),
    )


def compare_torch_vs_onnx(
    *,
    model: str,
    images: Sequence[str],
    onnx_path: str | None = None,
    with_camera: bool = False,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize_padded",
    providers: Sequence[str] | None = None,
    device: str = "cuda",
    keep_onnx: bool = False,
    opset_version: int = 17,
    height: int | None = None,
    width: int | None = None,
    out_dir: str | None = None,
    is_metric: bool | None = None,
) -> ParityReport:
    """Export ``model`` to ONNX (if ``onnx_path`` is not given) and compare the
    torch path against the ONNX path on ``images``.

    Returns a :class:`ParityReport`.
    """
    # Hard-fail early if either side is missing.
    try:
        import torch
    except ImportError as e:
        raise ImportError("parity test requires `[torch]` extras") from e
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise ImportError("parity test requires `[onnx]` or `[onnx-gpu]` extras") from e

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.onnx import DepthAnything3Onnx
    from depth_anything_3.onnx.export import export_depth_anything_3
    from depth_anything_3.onnx.input_processor import NumpyInputProcessor
    from depth_anything_3.onnx.output_processor import NumpyOutputProcessor
    from depth_anything_3.onnx.postprocess import (
        process_mono_sky,
        unletterbox_depth as _unletterbox_depth,
        unletterbox_mask as _unletterbox_mask,
    )

    # 1) Preprocess once with the numpy InputProcessor. This gives us a
    # single ``(N, 3, S, S)`` tensor and the per-image PreprocessInfo
    # records used for the inverse-letterbox step. Feeding the *same*
    # preprocessed array to both backends is the only way to make the
    # comparison apples-to-apples: the upstream torch InputProcessor
    # doesn't have ``upper_bound_resize_padded`` (and we don't want to
    # touch it), and even if it did, going through both processors
    # separately would introduce float-rounding drift before the model.
    np_proc = NumpyInputProcessor()
    batch_np, _ext_np, _ix_np, infos = np_proc(
        image=list(images),
        process_res=process_res,
        process_res_method=process_res_method,
        num_workers=1,
        sequential=True,
        return_original_sizes=True,
    )
    height = height or int(batch_np.shape[-2])
    width = width or int(batch_np.shape[-1])

    # 2) Export ONNX (if not pre-supplied)
    if onnx_path is None:
        tmpdir = tempfile.mkdtemp(prefix="da3-onnx-parity-")
        onnx_path = os.path.join(tmpdir, "model.onnx")
        export_depth_anything_3(
            model_id_or_path=model,
            out_path=onnx_path,
            with_camera=with_camera,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            views=len(images),
            height=height,
            width=width,
            opset_version=opset_version,
            device=device,
        )

    # 3) Torch forward on the same preprocessed tensor.
    torch_model = DepthAnything3.from_pretrained(model).to(device)
    torch_model.eval()
    image_tensor = torch.from_numpy(batch_np)[None].to(device).float()  # (1, N, 3, S, S)
    with torch.inference_mode():
        raw_torch = torch_model.forward(
            image_tensor,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],          # explicit: backbone iterates over this
            infer_gs=False,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
    # Move all tensors to CPU + numpy via the shared output processor
    out_proc = NumpyOutputProcessor()
    raw_torch_np = _torch_dict_to_numpy(raw_torch)
    pred_torch = out_proc(raw_torch_np)

    # 4) ONNX forward on the same preprocessed tensor.
    runner = DepthAnything3Onnx(onnx_path, providers=list(providers) if providers else None)
    raw_onnx = runner.session.run(image_tensor.cpu().numpy(), None, None)
    # Mirror the sky-cap post-processing the ONNX inference path applies
    # (`depth_anything_3.onnx.api.inference` step 3) so torch and ONNX
    # both have the same `_process_mono_sky_estimation`-equivalent
    # applied before we diff.
    if raw_onnx.get("sky") is not None and raw_onnx.get("depth") is not None:
        d = _drop_leading_batch(raw_onnx["depth"])
        s = _drop_leading_batch(raw_onnx["sky"])
        raw_onnx["depth"] = process_mono_sky(d, s)[None]
    pred_onnx = out_proc(raw_onnx)

    # 4b) Crop padding + resize each map back to its image's original
    # (H, W) on both sides. The compare image then shows what the user
    # actually receives -- no gray bars, no SxS canvas.
    def _to_orig_array(arr, kind: str):
        if arr is None:
            return None
        lst = (
            _unletterbox_mask(arr, infos)
            if kind == "mask"
            else _unletterbox_depth(arr, infos, "linear")
        )
        sizes = {info.original_size for info in infos}
        return np.stack(lst, axis=0) if len(sizes) == 1 else np.asarray(lst, dtype=object)

    pred_torch.depth = _to_orig_array(pred_torch.depth, "depth")
    pred_torch.conf = _to_orig_array(pred_torch.conf, "depth")
    if pred_torch.sky is not None:
        pred_torch.sky = _to_orig_array(pred_torch.sky, "mask")
    pred_onnx.depth = _to_orig_array(pred_onnx.depth, "depth")
    pred_onnx.conf = _to_orig_array(pred_onnx.conf, "depth")
    if pred_onnx.sky is not None:
        pred_onnx.sky = _to_orig_array(pred_onnx.sky, "mask")
    pred_torch.processed_images = _load_originals_like(list(images), infos)
    pred_onnx.processed_images = pred_torch.processed_images

    # 5) Compare per-field
    report = ParityReport(
        model=model,
        onnx_path=onnx_path,
        images=list(images),
        process_res=process_res,
        process_res_method=process_res_method,
        providers=list(providers) if providers else ["<default>"],
    )
    for name, t, o in (
        ("depth", pred_torch.depth, pred_onnx.depth),
        ("conf", pred_torch.conf, pred_onnx.conf),
        ("extrinsics", pred_torch.extrinsics, pred_onnx.extrinsics),
        ("intrinsics", pred_torch.intrinsics, pred_onnx.intrinsics),
        (
            "sky",
            pred_torch.sky.astype(np.float32) if pred_torch.sky is not None else None,
            pred_onnx.sky.astype(np.float32) if pred_onnx.sky is not None else None,
        ),
    ):
        d = _diff(name, t, o)
        if d is not None:
            report.diffs.append(d)

    # 6) Save side-by-side depth visualizations and raw arrays if asked
    if out_dir is not None:
        # Auto-detect metric from model id if neither prediction sets is_metric.
        effective_is_metric = is_metric
        if effective_is_metric is None:
            mlow = model.lower()
            if "metric" in mlow or "nested" in mlow:
                effective_is_metric = True
        _save_comparison_artifacts(
            out_dir=out_dir,
            torch_pred=pred_torch,
            onnx_pred=pred_onnx,
            image_paths=list(images),
            is_metric=effective_is_metric,
        )
        report.out_dir = out_dir

    if not keep_onnx and "tmpdir" in locals():
        # Don't auto-rm — leave the temp file so the user can inspect. Just
        # tell them where it is via the report.
        pass

    return report


def _save_comparison_artifacts(
    *,
    out_dir: str,
    torch_pred,
    onnx_pred,
    image_paths: list[str],
    is_metric: bool | None = None,
) -> None:
    """Save per-view torch/onnx/diff visualizations + raw arrays in ``out_dir``.

    Each input image gets:
      ``{i}_{stem}_compare.png`` -- 4-panel figure with units + colorbar
      ``{i}_{stem}_input.jpg``    -- the preprocessed input
      ``{i}_{stem}_torch.jpg``    -- depth from torch (Spectral, shared scale)
      ``{i}_{stem}_onnx.jpg``     -- depth from onnx (Spectral, shared scale)
      ``{i}_{stem}_diff.jpg``     -- abs diff (magma colormap, fixed-scale)
    ``arrays.npz`` at the root holds the raw float arrays.
    """
    import imageio
    import matplotlib.pyplot as plt

    from depth_anything_3.utils.visualize import visualize_depth

    os.makedirs(out_dir, exist_ok=True)

    def _as_bool(x):
        if x is None:
            return None
        return x.astype(bool) if x.dtype != bool else x

    torch_depth = torch_pred.depth
    onnx_depth = onnx_pred.depth
    N = torch_depth.shape[0]

    # Unit for the colorbar label. Metric models predict depth in meters;
    # non-metric ones are scale-free relative-depth units.
    if is_metric is None:
        is_metric = bool(getattr(torch_pred, "is_metric", 0)) or bool(
            getattr(onnx_pred, "is_metric", 0)
        )
    unit_label = "m" if is_metric else "depth-units"

    # Shared color range for the two depth panels.
    combined = np.concatenate([torch_depth.reshape(-1), onnx_depth.reshape(-1)])
    valid = combined[np.isfinite(combined) & (combined > 0)]
    if valid.size == 0:
        depth_min, depth_max = 0.0, 1.0
    else:
        inv = 1.0 / valid
        depth_min = float(np.percentile(inv, 2))
        depth_max = float(np.percentile(inv, 98))

    # Depth range in *real* units, for context lines on the figure.
    if valid.size == 0:
        depth_lo, depth_hi = 0.0, 0.0
    else:
        depth_lo = float(np.percentile(valid, 2))
        depth_hi = float(np.percentile(valid, 98))

    # Persist the raw arrays for offline inspection.
    raw = {
        "torch_depth": torch_depth,
        "onnx_depth": onnx_depth,
        "abs_diff": np.abs(torch_depth - onnx_depth),
        "is_metric": np.array(int(is_metric), dtype=np.int32),
    }
    if torch_pred.conf is not None and onnx_pred.conf is not None:
        raw["torch_conf"] = torch_pred.conf
        raw["onnx_conf"] = onnx_pred.conf
    if torch_pred.sky is not None and onnx_pred.sky is not None:
        raw["torch_sky"] = _as_bool(torch_pred.sky)
        raw["onnx_sky"] = _as_bool(onnx_pred.sky)
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), **raw)

    for i in range(N):
        # ---------- per-view individual JPGs (unchanged behaviour) ----------
        vis_t = visualize_depth(torch_depth[i], depth_min=depth_min, depth_max=depth_max)
        vis_o = visualize_depth(onnx_depth[i], depth_min=depth_min, depth_max=depth_max)

        diff = np.abs(torch_depth[i] - onnx_depth[i])
        if diff.size > 0 and diff.max() > 0:
            scale_99 = float(np.percentile(diff, 99))
            scale_99 = max(scale_99, 1e-6)
        else:
            scale_99 = 1.0
        diff_max = float(diff.max()) if diff.size > 0 else 0.0
        diff_mean = float(diff.mean()) if diff.size > 0 else 0.0

        # Magma-mapped diff with the same color scale used in the figure.
        cmap_diff = plt.get_cmap("magma")
        diff_norm = np.clip(diff / scale_99, 0.0, 1.0)
        diff_vis_rgb = (cmap_diff(diff_norm)[..., :3] * 255).astype(np.uint8)

        input_thumb = None
        if torch_pred.processed_images is not None:
            input_thumb = torch_pred.processed_images[i]

        stem = (
            os.path.splitext(os.path.basename(image_paths[i]))[0]
            if i < len(image_paths)
            else f"view{i:03d}"
        )
        if input_thumb is not None:
            imageio.imwrite(
                os.path.join(out_dir, f"{i:03d}_{stem}_input.jpg"), input_thumb, quality=95
            )
        imageio.imwrite(os.path.join(out_dir, f"{i:03d}_{stem}_torch.jpg"), vis_t, quality=95)
        imageio.imwrite(os.path.join(out_dir, f"{i:03d}_{stem}_onnx.jpg"), vis_o, quality=95)
        imageio.imwrite(os.path.join(out_dir, f"{i:03d}_{stem}_diff.jpg"), diff_vis_rgb, quality=95)

        # ---------- matplotlib compare PNG (2 rows + readable colorbars) ----------
        # Row 1: input | torch depth | onnx depth   (shared depth colorbar)
        # Row 2: |Δdepth| (wide, with magma colorbar)
        H, W = diff.shape
        aspect = W / max(H, 1)
        panel_w_in = max(2.6, aspect * 2.6)
        fig_w = 3 * panel_w_in + 1.0   # 3 panels + 1.0 inch for the depth colorbar
        fig_h = 2 * panel_w_in / max(aspect, 0.4) + 1.6
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 1.0, 1.0, 0.05],
            height_ratios=[1.0, 1.05],
            hspace=0.25,
            wspace=0.08,
        )

        ax_in = fig.add_subplot(gs[0, 0])
        ax_t = fig.add_subplot(gs[0, 1])
        ax_o = fig.add_subplot(gs[0, 2])
        cax_d = fig.add_subplot(gs[0, 3])
        ax_diff = fig.add_subplot(gs[1, :3])
        cax_e = fig.add_subplot(gs[1, 3])

        # Panel: input
        if input_thumb is not None:
            ax_in.imshow(input_thumb)
        else:
            ax_in.imshow(np.zeros_like(diff))
        ax_in.set_title("input")
        ax_in.set_axis_off()

        # Panel: torch depth (Spectral, shared scale)
        im_t = ax_t.imshow(torch_depth[i], cmap="Spectral", vmin=depth_lo, vmax=depth_hi)
        ax_t.set_title("torch depth")
        ax_t.set_axis_off()

        # Panel: ONNX depth (same scale)
        ax_o.imshow(onnx_depth[i], cmap="Spectral", vmin=depth_lo, vmax=depth_hi)
        ax_o.set_title("onnx depth")
        ax_o.set_axis_off()

        # Depth colorbar shared between torch / onnx
        cbar_d = fig.colorbar(im_t, cax=cax_d)
        cbar_d.set_label(f"depth [{unit_label}]")

        # Panel: |Δdepth| with magma + own colorbar (real units, clipped at p99)
        im_d = ax_diff.imshow(diff, cmap="magma", vmin=0.0, vmax=scale_99)
        ax_diff.set_title(
            f"|Δdepth| (color clipped at p99 = {scale_99:.3g} {unit_label};  "
            f"true max = {diff_max:.3g} {unit_label})"
        )
        ax_diff.set_axis_off()
        cbar_e = fig.colorbar(im_d, cax=cax_e)
        cbar_e.set_label(f"|Δdepth| [{unit_label}]")

        # Header with context so a reader sees the diff *relative to* the depth
        # range, not as a standalone-magnitude scary number.
        diff_ratio = (diff_mean / max(depth_hi - depth_lo, 1e-8)) * 100.0
        fig.suptitle(
            f"view {i:03d} ({stem})   "
            f"depth range ~[{depth_lo:.2f}, {depth_hi:.2f}] {unit_label}   |   "
            f"|Δdepth|  mean={diff_mean:.3g}  max={diff_max:.3g}  p99={scale_99:.3g} {unit_label}   "
            f"(mean ≈ {diff_ratio:.2f}% of depth span)",
            fontsize=10,
        )
        fig.savefig(
            os.path.join(out_dir, f"{i:03d}_{stem}_compare.png"),
            dpi=110,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def _drop_leading_batch(arr) -> np.ndarray:
    """Squeeze a leading B==1 axis from a 4D output if present."""
    a = np.asarray(arr)
    if a.ndim == 5 and a.shape[0] == 1:
        a = a[0]
        if a.shape[1] == 1:
            a = a.squeeze(1)
        elif a.shape[-1] == 1:
            a = a.squeeze(-1)
    elif a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a


def _torch_dict_to_numpy(raw):
    """Convert a torch-side model output dict to a numpy dict the
    NumpyOutputProcessor can consume. Mirrors what the torch
    OutputProcessor would do, minus the construction of a Prediction
    (we want a dict so the same downstream code handles both backends).
    """
    import torch  # local: this module is only loaded by parity tooling

    def _to_np(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return v

    return {k: _to_np(v) for k, v in raw.items() if v is not None}


def _load_originals_like(images, infos):
    """Reload the source images at their original (H, W) so the compare
    visualization shows the un-padded scene. Returns ``np.ndarray`` of
    shape ``(N, H, W, 3)`` if all sizes agree, else an object array.
    """
    from PIL import Image as _PIL

    same_size = len({info.original_size for info in infos}) == 1
    out = []
    for img_input, info in zip(images, infos):
        H, W = info.original_size
        if isinstance(img_input, str):
            pil = _PIL.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            pil = _PIL.fromarray(img_input).convert("RGB")
        else:
            pil = img_input.convert("RGB")
        if pil.size != (W, H):
            pil = pil.resize((W, H), _PIL.BILINEAR)
        out.append(np.asarray(pil))
    return np.stack(out, axis=0) if same_size else np.asarray(out, dtype=object)


__all__ = ["ParityReport", "FieldDiff", "compare_torch_vs_onnx"]
