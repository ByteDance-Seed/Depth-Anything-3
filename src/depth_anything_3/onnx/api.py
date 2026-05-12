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

"""ONNX-runtime-backed inference API for Depth Anything 3.

This module intentionally avoids importing torch. It can be loaded on a system
that has only ``numpy``, ``onnxruntime``, ``opencv-python``, ``pillow``, plus
the lightweight project deps (huggingface_hub, omegaconf, ...).
"""

from __future__ import annotations

import time
from typing import Optional, Sequence
import numpy as np
from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

from .input_processor import NumpyInputProcessor, denormalize_to_uint8
from .numpy_geometry import normalize_extrinsics_np
from .output_processor import NumpyOutputProcessor
from .postprocess import (
    apply_nested_metric_alignment,
    process_mono_sky,
    unletterbox_depth,
    unletterbox_mask,
)
from .session import OrtSession

_UNSUPPORTED_GS_MESSAGE = (
    "infer_gs=True is not supported on the ONNX inference path: 3D Gaussian "
    "Splatting depends on gsplat (CUDA) and cannot be exported to ONNX. Use "
    "the torch path (DepthAnything3) for GS inference."
)


def _maybe_download(model_path: str) -> str:
    """If ``model_path`` is an HF Hub repo id, download the ONNX file from it."""
    import os

    if os.path.isfile(model_path):
        return model_path

    # Treat anything that's not a local path as a HF Hub spec. Two forms accepted:
    #   "user/repo"                 -> download "model.onnx"
    #   "user/repo:filename.onnx"   -> download "filename.onnx"
    from huggingface_hub import hf_hub_download

    repo, _, filename = model_path.partition(":")
    filename = filename or "model.onnx"
    return hf_hub_download(repo_id=repo, filename=filename)


class DepthAnything3Onnx:
    """Single-net ONNX inference (depth + camera + optional sky).

    Use this for ``da3-large``, ``da3-giant``, ``da3-base``, ``da3-small`` and
    the metric variants (``da3metric-*``). For ``da3nested-*`` (metric +
    main combined) use :class:`DepthAnything3OnnxNested`.
    """

    def __init__(
        self,
        model_path: str,
        providers: Sequence[str | tuple[str, dict]] | None = None,
    ) -> None:
        self.session = OrtSession(_maybe_download(model_path), providers=providers)
        self.input_processor = NumpyInputProcessor()
        self.output_processor = NumpyOutputProcessor()

    # ------------------------------------------------------------------
    # Public inference entry point (mirrors DepthAnything3.inference)
    # ------------------------------------------------------------------
    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,  # noqa: ARG002 -- baked at export time
        ref_view_strategy: str = "saddle_balanced",  # noqa: ARG002 -- baked at export time
        render_exts: np.ndarray | None = None,  # noqa: ARG002 -- GS only
        render_ixts: np.ndarray | None = None,  # noqa: ARG002 -- GS only
        render_hw: tuple[int, int] | None = None,  # noqa: ARG002 -- GS only
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize_padded",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,  # noqa: ARG002 -- baked at export
        # GLB
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis
        feat_vis_fps: int = 15,
        # Other export kwargs
        export_kwargs: Optional[dict] = None,
        # When True (default), depth/conf/sky are cropped + resized back
        # to each input's original (H, W) after running the SxS ONNX
        # graph. Set False to keep outputs at the model's native SxS
        # canvas (e.g. for parity testing against the raw torch output).
        unsquare: bool = True,
    ) -> Prediction:
        if infer_gs:
            raise NotImplementedError(_UNSUPPORTED_GS_MESSAGE)
        if export_format and "gs" in export_format:
            raise NotImplementedError(_UNSUPPORTED_GS_MESSAGE)

        # 1) Preprocess. With process_res_method in
        # {upper_bound_resize_padded, square_resize} the processed images
        # are SxS regardless of aspect ratio, and we keep a PreprocessInfo
        # per image so we can invert the transform afterwards.
        imgs_arr, ex_np, in_np, infos = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # 2) Add batch dim, run the session
        image_input = imgs_arr[None]  # (1, N, 3, H, W)
        ex_input = ex_np[None] if ex_np is not None else None
        in_input = in_np[None] if in_np is not None else None
        ex_input_norm = normalize_extrinsics_np(ex_input)

        raw = self._run(image_input, ex_input_norm, in_input)

        # 3) Optional sky-aware depth capping (mirrors the torch _process_mono_sky)
        depth = raw.get("depth")
        sky = raw.get("sky", None)
        if depth is not None and sky is not None:
            # depth shape (B, N, 1, H, W) or (B, N, H, W) -> drop dummy axes for sky
            depth_arr = np.asarray(depth)
            sky_arr = np.asarray(sky)
            if depth_arr.ndim == 5:
                # squeeze trailing channel if present
                if depth_arr.shape[-1] == 1:
                    depth_arr = depth_arr.squeeze(-1)
                elif depth_arr.shape[2] == 1:
                    depth_arr = depth_arr.squeeze(2)
            if sky_arr.ndim == 4 and sky_arr.shape[0] == 1:
                sky_arr_unbatched = sky_arr[0]
            else:
                sky_arr_unbatched = sky_arr
            if depth_arr.ndim == 4 and depth_arr.shape[0] == 1:
                depth_unbatched = depth_arr[0]
            else:
                depth_unbatched = depth_arr
            depth_post = process_mono_sky(depth_unbatched, sky_arr_unbatched)
            raw["depth"] = depth_post[None]

        # 4) Output processor -> Prediction
        prediction = self.output_processor(raw)

        # 5) Pose alignment to input extrinsics, same code path as torch API
        prediction = self._align_to_input_ext(
            ex_np, in_np, prediction, align_to_input_ext_scale
        )

        # 6) Crop padding + resize each depth/conf/sky map back to its
        # input's original aspect ratio. Applies to both
        # ``upper_bound_resize_padded`` and ``square_resize`` modes (and
        # to the legacy ``upper_bound_resize`` family, where the crop is
        # a no-op because the canvas already matches the scaled region).
        # Skipped when ``unsquare=False`` (parity testing keeps the
        # native SxS output of the model).
        if process_res_method in ("upper_bound_resize_padded", "square_resize") and unsquare:
            prediction = self._unletterbox(prediction, infos)
            prediction.processed_images = self._load_original_images(
                image, [info.original_size for info in infos]
            )
        else:
            prediction.processed_images = denormalize_to_uint8(imgs_arr)

        # 8) Export
        if export_dir is not None:
            self._dispatch_export(
                prediction,
                image,
                export_dir,
                export_format,
                conf_thresh_percentile=conf_thresh_percentile,
                num_max_points=num_max_points,
                show_cameras=show_cameras,
                feat_vis_fps=feat_vis_fps,
                process_res_method=process_res_method,
                export_kwargs=export_kwargs or {},
            )

        return prediction

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _preprocess_inputs(
        self,
        image,
        extrinsics,
        intrinsics,
        process_res,
        process_res_method,
    ):
        start = time.time()
        imgs, ex, ix, infos = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
            return_original_sizes=True,
        )
        logger.info(
            f"Processed images: shape={imgs.shape} in {time.time() - start:.3f}s"
        )
        return imgs, ex, ix, infos

    @staticmethod
    def _load_original_images(
        image: list,
        original_sizes: list[tuple[int, int]],
    ):
        """Reload the input images at their original (H, W) for downstream
        export (e.g. depth_vis). Returns ``np.ndarray`` of shape
        ``(N, H, W, 3)`` when all sizes agree, else an object array."""
        same_size = len(set(original_sizes)) == 1
        out = []
        for img_input, (H, W) in zip(image, original_sizes):
            if isinstance(img_input, str):
                pil = Image.open(img_input).convert("RGB")
            elif isinstance(img_input, np.ndarray):
                pil = Image.fromarray(img_input).convert("RGB")
            elif isinstance(img_input, Image.Image):
                pil = img_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type for original load: {type(img_input)}")
            # PIL .size is (W, H); cmp against (W, H).
            if pil.size != (W, H):
                pil = pil.resize((W, H), Image.BILINEAR)
            out.append(np.asarray(pil))
        return np.stack(out, axis=0) if same_size else np.asarray(out, dtype=object)

    def _unletterbox(
        self,
        prediction: Prediction,
        infos: list,    # list[PreprocessInfo]
    ) -> Prediction:
        """Map depth / conf / sky / intrinsics from the model's canvas back
        to each input image's original (H, W). When all images share the
        same original size we keep a single ndarray; otherwise the field
        becomes an object array of per-image arrays so downstream code can
        still iterate.
        """
        original_sizes = [info.original_size for info in infos]
        same_size = len(set(original_sizes)) == 1

        def _stackable(lst):
            return np.stack(lst, axis=0) if same_size else np.asarray(lst, dtype=object)

        if prediction.depth is not None:
            prediction.depth = _stackable(unletterbox_depth(prediction.depth, infos, "linear"))
        if prediction.conf is not None:
            prediction.conf = _stackable(unletterbox_depth(prediction.conf, infos, "linear"))
        if prediction.sky is not None:
            prediction.sky = _stackable(unletterbox_mask(prediction.sky, infos))

        # Intrinsics: invert the per-image scale+translate the preprocessor
        # applied. If the graph emits intrinsics (cam_dec branch), they're
        # in canvas-pixel units; we want them in original-image-pixel units.
        if prediction.intrinsics is not None:
            adj = prediction.intrinsics.astype(np.float32).copy()
            for i, info in enumerate(infos):
                # cx_orig = (cx_canvas - pad_left) / scale_x
                adj[i, 0, 2] = (adj[i, 0, 2] - info.pad_left) / info.scale_x
                adj[i, 1, 2] = (adj[i, 1, 2] - info.pad_top) / info.scale_y
                adj[i, 0, 0] = adj[i, 0, 0] / info.scale_x  # fx
                adj[i, 1, 1] = adj[i, 1, 1] / info.scale_y  # fy
            prediction.intrinsics = adj

        return prediction

    def _run(
        self,
        image_input: np.ndarray,
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        start = time.time()
        outputs = self.session.run(image_input, extrinsics, intrinsics)
        logger.info(f"ONNX forward pass: {time.time() - start:.3f}s")
        return outputs

    def _align_to_input_ext(
        self,
        ex_in: np.ndarray | None,
        ix_in: np.ndarray | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        if ex_in is None:
            return prediction
        prediction.intrinsics = ix_in if ix_in is not None else prediction.intrinsics
        _, _, scale, aligned_ext = align_poses_umeyama(
            prediction.extrinsics,
            ex_in,
            ransac=len(ex_in) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = ex_in[..., :3, :]
            prediction.depth = prediction.depth / scale
        else:
            prediction.extrinsics = aligned_ext
        return prediction

    def _dispatch_export(
        self,
        prediction: Prediction,
        original_images,
        export_dir: str,
        export_format: str,
        *,
        conf_thresh_percentile: float,
        num_max_points: int,
        show_cameras: bool,
        feat_vis_fps: int,
        process_res_method: str,
        export_kwargs: dict,
    ) -> None:
        if "colmap" in export_format:
            assert isinstance(
                original_images[0], str
            ), "`image` must be image paths for COLMAP export."
            export_kwargs.setdefault("colmap", {}).update(
                {
                    "image_paths": original_images,
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "process_res_method": process_res_method,
                }
            )
        if "glb" in export_format:
            export_kwargs.setdefault("glb", {}).update(
                {
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "num_max_points": num_max_points,
                    "show_cameras": show_cameras,
                }
            )
        if "feat_vis" in export_format:
            export_kwargs.setdefault("feat_vis", {}).update({"fps": feat_vis_fps})
        start = time.time()
        export(prediction, export_format, export_dir, **export_kwargs)
        logger.info(f"Export ({export_format}) took {time.time() - start:.3f}s")


class DepthAnything3OnnxNested:
    """Nested-metric ONNX inference.

    The nested model wraps two sub-nets (``da3`` for general depth/camera, and
    ``da3_metric`` for sky + metric depth). Because the original
    ``NestedDepthAnything3Net.forward`` uses control flow and aggregations
    (``torch.quantile``, dynamic masking) that don't ONNX-export cleanly, we
    export the two sub-nets separately and do the alignment / sky handling on
    the host via :func:`postprocess.apply_nested_metric_alignment`.
    """

    def __init__(
        self,
        main_model_path: str,
        metric_model_path: str,
        providers: Sequence[str | tuple[str, dict]] | None = None,
    ) -> None:
        self.main_session = OrtSession(_maybe_download(main_model_path), providers=providers)
        self.metric_session = OrtSession(
            _maybe_download(metric_model_path), providers=providers
        )
        self.input_processor = NumpyInputProcessor()
        self.output_processor = NumpyOutputProcessor()

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize_padded",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
        export_kwargs: Optional[dict] = None,
        **_unused,
    ) -> Prediction:
        if infer_gs:
            raise NotImplementedError(_UNSUPPORTED_GS_MESSAGE)
        if export_format and "gs" in export_format:
            raise NotImplementedError(_UNSUPPORTED_GS_MESSAGE)

        # 1) Preprocess (shared between the two sub-nets); track the
        # per-image PreprocessInfo for the unletterbox step below.
        imgs, ex_np, in_np, infos = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
            return_original_sizes=True,
        )
        image_input = imgs[None]  # (1, N, 3, H, W)
        ex_input = ex_np[None] if ex_np is not None else None
        in_input = in_np[None] if in_np is not None else None
        ex_input_norm = normalize_extrinsics_np(ex_input)

        # 2) Main branch (depth, depth_conf, extrinsics, intrinsics)
        main_out = self.main_session.run(image_input, ex_input_norm, in_input)

        # 3) Metric branch (depth, sky). Image-only.
        metric_out = self.metric_session.run(image_input)

        # 4) Align / scale / sky
        depth_main = _drop_channel_dim(main_out["depth"])  # (B, N, H, W)
        depth_conf = _drop_channel_dim(main_out["depth_conf"])
        extrinsics_pred = main_out["extrinsics"]
        intrinsics_pred = main_out["intrinsics"]
        metric_depth = _drop_channel_dim(metric_out["depth"])
        metric_sky = _drop_channel_dim(metric_out["sky"])

        depth_final, depth_conf_final, ext_scaled, scale_factor = apply_nested_metric_alignment(
            depth=depth_main,
            depth_conf=depth_conf,
            extrinsics=extrinsics_pred,
            intrinsics=intrinsics_pred,
            metric_depth=metric_depth,
            metric_sky=metric_sky,
        )

        # 5) Build a unified raw-output dict and let the output processor squeeze
        # away the batch axis exactly like the single-net path.
        raw = {
            "depth": depth_final,  # (B, N, H, W)
            "depth_conf": depth_conf_final,  # (B, N, H, W)
            "extrinsics": ext_scaled,
            "intrinsics": intrinsics_pred,
            "sky": metric_sky,
            "is_metric": 1,
            "scale_factor": scale_factor,
        }
        prediction = self.output_processor(raw)
        prediction.scale_factor = scale_factor

        # 6) Pose alignment to input extrinsics, if any
        prediction = DepthAnything3Onnx._align_to_input_ext(
            self,
            ex_np,
            in_np,
            prediction,
            align_to_input_ext_scale,
        )

        # 7) Crop padding + resize depth back to original size, then load
        # the originals for processed_images so depth_vis lines up.
        if process_res_method in ("upper_bound_resize_padded", "square_resize"):
            prediction = DepthAnything3Onnx._unletterbox(self, prediction, infos)
            prediction.processed_images = DepthAnything3Onnx._load_original_images(
                image, [info.original_size for info in infos]
            )
        else:
            prediction.processed_images = denormalize_to_uint8(imgs)

        if export_dir is not None:
            DepthAnything3Onnx._dispatch_export(
                self,
                prediction,
                image,
                export_dir,
                export_format,
                conf_thresh_percentile=conf_thresh_percentile,
                num_max_points=num_max_points,
                show_cameras=show_cameras,
                feat_vis_fps=feat_vis_fps,
                process_res_method=process_res_method,
                export_kwargs=export_kwargs or {},
            )
        return prediction


def _drop_channel_dim(arr) -> np.ndarray:
    """Some graphs emit (B, N, 1, H, W); collapse to (B, N, H, W)."""
    a = np.asarray(arr)
    if a.ndim == 5:
        if a.shape[-1] == 1:
            return a.squeeze(-1)
        if a.shape[2] == 1:
            return a.squeeze(2)
    return a


__all__ = ["DepthAnything3Onnx", "DepthAnything3OnnxNested"]
