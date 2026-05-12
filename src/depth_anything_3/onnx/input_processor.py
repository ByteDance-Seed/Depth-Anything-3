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

"""NumPy-only port of `depth_anything_3.utils.io.input_processor.InputProcessor`.

Same pipeline (boundary resize -> patch-size enforcement -> ImageNet
normalization -> stack), same intrinsic bookkeeping, same ``(1, N, 3, H, W)``
output -- but the return is a `np.ndarray` instead of a `torch.Tensor`, and no
`torchvision` / `torch` imports happen anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np
from PIL import Image

from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.parallel_utils import parallel_execution

# ImageNet mean / std used by all models in this family.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Padding color for letterbox, in [0, 255] uint8 (equals _IMAGENET_MEAN * 255).
# Padding with the ImageNet mean color means the padded pixels become 0 after
# normalization, so they look maximally "neutral" to the model.
_PAD_COLOR_U8 = (_IMAGENET_MEAN * 255.0).round().astype(np.uint8)


@dataclass
class PreprocessInfo:
    """Per-image record of what the preprocessor did, so postprocess can
    invert it uniformly across modes.

    The contract is:

        input image (orig_h, orig_w)
          --- resize by (scale_x, scale_y) ---> (scaled_h, scaled_w)
          --- pad to (canvas_h, canvas_w) with `pad_left`, `pad_top` ---> the
              tensor fed to the model

    For the ``letterbox`` and ``square_resize`` modes, ``(canvas_h, canvas_w)``
    is the fixed ``(S, S)`` of a square ONNX graph. For the legacy DA3
    modes (``upper_bound_resize`` etc.) ``pad_left == pad_top == 0`` and
    ``(canvas_h, canvas_w) == (scaled_h, scaled_w)``.

    Postprocessing for any mode:

        depth_at_canvas[..., canvas_h, canvas_w]
          --- crop [pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w] -->
              depth_at_scaled
          --- cv2.resize to (orig_h, orig_w) -->
              depth at the original input resolution
    """

    original_size: tuple[int, int]   # (H, W) of the input image
    scale_x: float                   # W_orig  → W_scaled
    scale_y: float                   # H_orig  → H_scaled
    pad_left: int
    pad_top: int
    scaled_size: tuple[int, int]     # (H, W) of real image content inside the canvas
    canvas_size: tuple[int, int]     # (H, W) of the tensor fed to the model


class NumpyInputProcessor:
    """Drop-in replacement for `InputProcessor` that uses NumPy/PIL/OpenCV only."""

    PATCH_SIZE = 14

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        num_workers: int = 8,
        print_progress: bool = False,
        sequential: bool | None = None,
        desc: str | None = "Preprocess",
        return_original_sizes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Returns ``(batch, extrinsics, intrinsics)``.

        - ``batch`` has shape ``(N, 3, H, W)`` and dtype ``float32``. The caller
          is responsible for adding the leading batch axis if needed.
        - ``extrinsics`` / ``intrinsics`` are ``np.ndarray`` of shape
          ``(N, 4, 4)`` / ``(N, 3, 3)``, or ``None`` if not provided.

        Two modes target a fixed-square ONNX graph:

        - ``process_res_method="upper_bound_resize_padded"`` (recommended):
          identical to the DA3 ``upper_bound_resize`` aspect-preserving
          resize (longest side ≤ ``process_res``, snap dims to
          ``PATCH_SIZE``), then pad with ImageNet-mean grey to
          ``(process_res, process_res)``. The model sees the exact same
          pixels as `upper_bound_resize` would feed it — the only addition
          is neutral padding bars to fill the fixed canvas.
        - ``process_res_method="square_resize"``: direct aspect-distorting
          resize to ``(process_res, process_res)``. No padding, but the
          image is squashed. Mirrors `MoonCodeMaster/Depth-Anything-3-Onnx`.

        Combined with ``return_original_sizes=True``, the caller can later
        invert either mode via `postprocess.unletterbox_depth` (which
        consumes the per-image `PreprocessInfo` records).

        If ``return_original_sizes=True`` the return becomes ``(batch, ext, ixt,
        infos)`` where ``infos`` is a list of `PreprocessInfo`, one per
        input image, in input order.
        """
        sequential = self._resolve_sequential(sequential, num_workers)
        exts_list, ixts_list = self._validate_and_pack_meta(image, extrinsics, intrinsics)

        results = parallel_execution(
            image,
            exts_list,
            ixts_list,
            action=self._process_one,
            num_processes=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if not results:
            raise RuntimeError("No preprocessing results returned.")

        proc_imgs, out_sizes, out_ixts, out_exts, infos = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(proc_imgs, out_sizes, out_ixts)

        batch_array = np.stack(proc_imgs, axis=0).astype(np.float32, copy=False)
        out_exts = (
            np.asarray(out_exts, dtype=np.float32)
            if out_exts is not None and out_exts[0] is not None
            else None
        )
        out_ixts = (
            np.asarray(out_ixts, dtype=np.float32)
            if out_ixts is not None and out_ixts[0] is not None
            else None
        )
        if return_original_sizes:
            return batch_array, out_exts, out_ixts, infos
        return batch_array, out_exts, out_ixts

    # -----------------------------
    # Helpers
    # -----------------------------
    def _resolve_sequential(self, sequential: bool | None, num_workers: int) -> bool:
        return (num_workers <= 1) if sequential is None else sequential

    def _validate_and_pack_meta(
        self,
        images: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[list[np.ndarray | None] | None, list[np.ndarray | None] | None]:
        if extrinsics is not None and len(extrinsics) != len(images):
            raise ValueError("Length of extrinsics must match images when provided.")
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError("Length of intrinsics must match images when provided.")
        exts_list = [e for e in extrinsics] if extrinsics is not None else None
        ixts_list = [k for k in intrinsics] if intrinsics is not None else None
        return exts_list, ixts_list

    def _unpack_results(self, results):
        try:
            (
                processed_images,
                out_sizes,
                out_intrinsics,
                out_extrinsics,
                infos,
            ) = zip(*results)
        except Exception as e:
            raise RuntimeError(
                "Unexpected results structure from parallel_execution: "
                f"{type(results)} / sample: {results[0]}"
            ) from e
        return (
            list(processed_images),
            list(out_sizes),
            list(out_intrinsics),
            list(out_extrinsics),
            list(infos),
        )

    def _unify_batch_shapes(
        self,
        processed_images: list[np.ndarray],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[np.ndarray], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all (3, H, W) arrays to the smallest H, W; adjust K's cx/cy."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)
        logger.warn(
            f"Images in batch have different sizes {out_sizes}; "
            f"center-cropping all to smallest ({min_h},{min_w})"
        )

        new_imgs, new_sizes, new_ixts = [], [], []
        for img_arr, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
            crop_top = max(0, (H - min_h) // 2)
            crop_left = max(0, (W - min_w) // 2)
            cropped = img_arr[:, crop_top : crop_top + min_h, crop_left : crop_left + min_w]
            new_imgs.append(cropped)
            new_sizes.append((min_h, min_w))
            if K is None:
                new_ixts.append(None)
            else:
                K_adj = K.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)
        return new_imgs, new_sizes, new_ixts

    # -----------------------------
    # Per-item worker (mirrors the torch version exactly)
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[np.ndarray, tuple[int, int], np.ndarray | None, np.ndarray | None, PreprocessInfo]:
        pil_img = self._load_image(img)
        orig_w, orig_h = pil_img.size

        if process_res_method == "upper_bound_resize_padded":
            # This is literally `upper_bound_resize` followed by a constant
            # padding step. Step 1 is identical to the DA3 mode of the same
            # name (aspect-preserving resize so the longest side ≤ S, snap
            # the result to multiples of PATCH_SIZE); step 2 just adds
            # ImageNet-mean gray bars to fill the canvas to ``(S, S)`` so
            # the tensor matches a fixed-square ONNX graph's input shape.
            # The model sees the *exact same pixels* as it would under
            # plain `upper_bound_resize` -- the only difference is the
            # neutral padding around them. Postprocess crops the pad and
            # resizes the un-padded region back to ``(orig_h, orig_w)``;
            # see `depth_anything_3.onnx.postprocess.unletterbox_depth`.
            assert process_res % self.PATCH_SIZE == 0, (
                f"upper_bound_resize_padded requires process_res "
                f"({process_res}) to be a multiple of PATCH_SIZE "
                f"({self.PATCH_SIZE})"
            )
            pil_img, info = self._upper_bound_resize_padded(
                pil_img, process_res, orig_w, orig_h
            )
            intrinsic = self._adjust_intrinsic(intrinsic, info)
        elif process_res_method == "square_resize":
            # Direct aspect-DISTORTING resize to (S, S). Same fixed-shape
            # consumer as `letterbox`, but no padding bars at the cost of
            # squashing the image. Matches MoonCodeMaster/Depth-Anything-3-Onnx
            # behaviour; kept as an option for users who want it.
            assert process_res % self.PATCH_SIZE == 0, (
                f"square_resize requires process_res ({process_res}) to be a "
                f"multiple of PATCH_SIZE ({self.PATCH_SIZE})"
            )
            pil_img = self._square_resize_pil(pil_img, process_res)
            info = PreprocessInfo(
                original_size=(orig_h, orig_w),
                scale_x=process_res / float(orig_w),
                scale_y=process_res / float(orig_h),
                pad_left=0,
                pad_top=0,
                scaled_size=(process_res, process_res),
                canvas_size=(process_res, process_res),
            )
            intrinsic = self._adjust_intrinsic(intrinsic, info)
        else:
            # Legacy DA3 modes (upper_bound_resize, lower_bound_resize,
            # upper_bound_crop, lower_bound_crop) -- aspect-preserving but
            # produce non-square outputs. Only useful when the consumer
            # accepts dynamic shapes (i.e. NOT a fixed-square ONNX).
            pil_img = self._resize_image(pil_img, process_res, process_res_method)
            w, h = pil_img.size

            if process_res_method.endswith("resize"):
                pil_img = self._make_divisible_by_resize(pil_img, self.PATCH_SIZE)
            elif process_res_method.endswith("crop"):
                pil_img = self._make_divisible_by_crop(pil_img, self.PATCH_SIZE)
            else:
                raise ValueError(f"Unsupported process_res_method: {process_res_method}")
            new_w, new_h = pil_img.size
            info = PreprocessInfo(
                original_size=(orig_h, orig_w),
                scale_x=new_w / float(orig_w),
                scale_y=new_h / float(orig_h),
                pad_left=0,
                pad_top=0,
                scaled_size=(new_h, new_w),
                canvas_size=(new_h, new_w),
            )
            intrinsic = self._adjust_intrinsic(intrinsic, info)

        img_arr = self._normalize_image(pil_img)  # (3, H, W) float32
        _, H, W = img_arr.shape
        assert (H, W) == info.canvas_size, (
            f"normalized array size {(H, W)} != canvas_size {info.canvas_size}"
        )
        return img_arr, (H, W), intrinsic, extrinsic, info

    def _square_resize_pil(self, img: Image.Image, target: int) -> Image.Image:
        w, h = img.size
        if (w, h) == (target, target):
            return img
        upscale = (target > w) or (target > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (target, target), interpolation=interpolation)
        return Image.fromarray(arr)

    def _upper_bound_resize_padded(
        self,
        img: Image.Image,
        target: int,
        orig_w: int,
        orig_h: int,
    ) -> tuple[Image.Image, PreprocessInfo]:
        """``upper_bound_resize`` followed by constant padding to a square.

        The first part is the same shape math the legacy DA3
        `upper_bound_resize` mode uses (see
        ``_resize_longest_side`` + ``_make_divisible_by_resize`` in this
        file, and the matching path in
        ``depth_anything_3.utils.io.input_processor.InputProcessor``):

            scale     = target / max(orig_w, orig_h)
            scaled_w  = round(orig_w * scale)
            scaled_h  = round(orig_h * scale)
            (snap both to multiples of PATCH_SIZE = 14)

        The second part is purely there to satisfy a fixed-shape ONNX
        graph: pad with the ImageNet mean color (``(123, 117, 104)`` in
        uint8) so the un-padded image is centered in a ``(target,
        target)`` canvas. After normalization the padded pixels become
        zero, i.e. maximally neutral input to the model.

        The PreprocessInfo we return carries the scale + pad metadata so
        ``unletterbox_depth`` can crop and resize the un-padded region
        back to ``(orig_h, orig_w)``.
        """
        scale = target / float(max(orig_w, orig_h))
        scaled_w = max(self.PATCH_SIZE, int(round(orig_w * scale)))
        scaled_h = max(self.PATCH_SIZE, int(round(orig_h * scale)))
        # Snap to multiples of PATCH_SIZE within the canvas, never larger
        # than `target` itself.
        scaled_w = min(target, (scaled_w // self.PATCH_SIZE) * self.PATCH_SIZE)
        scaled_h = min(target, (scaled_h // self.PATCH_SIZE) * self.PATCH_SIZE)
        if scaled_w == 0:
            scaled_w = self.PATCH_SIZE
        if scaled_h == 0:
            scaled_h = self.PATCH_SIZE

        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(np.asarray(img), (scaled_w, scaled_h), interpolation=interpolation)

        pad_left = (target - scaled_w) // 2
        pad_top = (target - scaled_h) // 2

        # Pad with the ImageNet mean color (≈ gray; → 0 after normalization).
        canvas = np.empty((target, target, 3), dtype=np.uint8)
        canvas[:] = _PAD_COLOR_U8
        canvas[pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w] = resized

        info = PreprocessInfo(
            original_size=(orig_h, orig_w),
            # Per-axis scale from original-image pixels to **scaled-region**
            # pixels (NOT to the canvas itself). These coincide because
            # `upper_bound_resize` is uniform-scale.
            scale_x=scaled_w / float(orig_w),
            scale_y=scaled_h / float(orig_h),
            pad_left=pad_left,
            pad_top=pad_top,
            scaled_size=(scaled_h, scaled_w),
            canvas_size=(target, target),
        )
        return Image.fromarray(canvas), info

    def _adjust_intrinsic(
        self,
        intrinsic: np.ndarray | None,
        info: PreprocessInfo,
    ) -> np.ndarray | None:
        """Apply the same scale+translate to a 3x3 intrinsic as the image
        preprocessor applied to the pixels."""
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        K[0, 0] *= info.scale_x          # fx
        K[0, 2] = K[0, 2] * info.scale_x + info.pad_left   # cx
        K[1, 1] *= info.scale_y          # fy
        K[1, 2] = K[1, 2] * info.scale_y + info.pad_top    # cy
        return K

    # -----------------------------
    # Intrinsics transforms (identical to the torch version)
    # -----------------------------
    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def _crop_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        crop_h = (orig_h - h) // 2
        crop_w = (orig_w - w) // 2
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        return K

    # -----------------------------
    # I/O & normalization (numpy versions)
    # -----------------------------
    def _load_image(self, img: np.ndarray | Image.Image | str) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            return Image.fromarray(img).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _normalize_image(self, img: Image.Image) -> np.ndarray:
        # PIL -> HWC float32 in [0, 1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # ImageNet normalize (broadcast over H, W)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW, contiguous so ORT zero-copies
        return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)

    # -----------------------------
    # Boundary resizing (cv2-based, same as the torch version)
    # -----------------------------
    def _resize_image(self, img: Image.Image, target_size: int, method: str) -> Image.Image:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        if method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        return self._cv2_resize(img, w, h, scale)

    def _resize_shortest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        return self._cv2_resize(img, w, h, scale)

    def _cv2_resize(self, img: Image.Image, w: int, h: int, scale: float) -> Image.Image:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    # -----------------------------
    # Make divisible by PATCH_SIZE (identical to torch version)
    # -----------------------------
    def _make_divisible_by_crop(self, img: Image.Image, patch: int) -> Image.Image:
        w, h = img.size
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def _make_divisible_by_resize(self, img: Image.Image, patch: int) -> Image.Image:
        w, h = img.size

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        upscale = (new_w > w) or (new_h > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)


def denormalize_to_uint8(batch: np.ndarray) -> np.ndarray:
    """Inverse of `_normalize_image`. ``batch`` is ``(N, 3, H, W)`` float; returns
    ``(N, H, W, 3)`` uint8 ready for `Prediction.processed_images`.
    """
    arr = batch.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


__all__ = ["NumpyInputProcessor", "denormalize_to_uint8"]


# ===========================
# Minimal test runner
# ===========================
if __name__ == "__main__":
    """Smoke test mirroring the original InputProcessor's test runner."""

    def fmt_k_line(K: np.ndarray | None) -> str:
        if K is None:
            return "None"
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        return f"fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}"

    def show_result(tag: str, arr: np.ndarray, Ks_in=None, Ks_out=None):
        N, C, H, W = arr.shape
        print(f"[{tag}] shape={arr.shape}  HxW=({H},{W})  div14=({H % 14 == 0},{W % 14 == 0})")
        assert H % 14 == 0 and W % 14 == 0, f"{tag}: output size not divisible by 14!"
        if Ks_in is not None or Ks_out is not None:
            Ks_in = Ks_in or [None] * N
            Ks_out = Ks_out or [None] * N
            for i in range(N):
                print(f"  K[{i}]: {fmt_k_line(Ks_in[i])}  ->  {fmt_k_line(Ks_out[i])}")

    proc = NumpyInputProcessor()
    methods = ["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"]
    sizes = [(680, 1208), (1208, 680)]
    for w, h in sizes:
        img = Image.new("RGB", (w, h), color=(123, 222, 100))
        batch_imgs = [img, img]
        for m in methods:
            arr, _, _ = proc(image=batch_imgs, process_res=504, process_res_method=m, num_workers=4)
            show_result(f"size=({w},{h}) | {m}", arr)
