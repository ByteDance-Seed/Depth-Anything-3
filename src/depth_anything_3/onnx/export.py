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

"""Torch -> ONNX exporter for Depth Anything 3.

Importing this module requires ``torch``. Install with::

    pip install depth-anything-3[torch,onnx]

Typical usage::

    from depth_anything_3.onnx.export import export_depth_anything_3
    export_depth_anything_3(
        model_id_or_path="depth-anything/DA3-LARGE-1.1",
        out_path="/tmp/da3-large.onnx",
        with_camera=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
        opset_version=17,
    )

For nested-metric models, both sub-nets are exported -- pass
``out_main_path`` and ``out_metric_path``.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover -- friendly message
    raise ImportError(
        "depth_anything_3.onnx.export requires torch. Install with "
        "`pip install depth-anything-3[torch,onnx]`."
    ) from e


# Default trace shapes. The exported graph fixes ``H`` and ``W`` at these
# values (the underlying model uses Python-int shape arithmetic that
# torch.jit.trace bakes into the graph as constants). To run at arbitrary
# resolutions, the ONNX inference path resizes inputs to (H, W) on the
# way in and resizes the depth output back on the way out -- see
# ``depth_anything_3.onnx.input_processor.NumpyInputProcessor`` with
# ``process_res_method="square_resize"`` and
# ``depth_anything_3.onnx.postprocess.unsquare_resize_depth``.
_DEFAULT_VIEWS = 1
_DEFAULT_H = 504  # 36 * 14
_DEFAULT_W = 504  # square; matches the upstream approach in MoonCodeMaster/Depth-Anything-3-Onnx


class _SingleNetWrapper(nn.Module):
    """Wrap ``DepthAnything3Net.forward`` with all string/list/bool kwargs baked
    in, so the exported graph only depends on the tensor inputs we care about.

    We *intentionally* skip ``_process_mono_sky_estimation`` and aux feature
    extraction in the exported graph: both contain data-dependent control flow
    (``if non_sky_mask.sum() <= 10``, conditional dict assignment) that don't
    survive ``torch.export`` / ``torch.jit.trace`` cleanly. The post-processing
    is reproduced in NumPy on the host side -- see
    ``depth_anything_3.onnx.postprocess.process_mono_sky``.
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        use_camera: bool,
        use_ray_pose: bool,
        ref_view_strategy: str,
    ) -> None:
        super().__init__()
        self.net = net
        self.use_camera = use_camera
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy

    # Names of fields we look for in the model's output dict, in stable order.
    # `which_outputs()` returns the subset that actually exist for a given net.
    _CANDIDATE_OUTPUTS = ("depth", "depth_conf", "extrinsics", "intrinsics", "sky")

    def which_outputs(self) -> list[str]:
        """Probe the net once to discover which output fields it produces.
        Returns the subset of ``_CANDIDATE_OUTPUTS`` that are present.
        """
        device = next(self.parameters()).device
        sample_image = torch.zeros(1, 1, 3, 14, 14, dtype=torch.float32, device=device)
        sample_ex = sample_ix = None
        if self.use_camera:
            sample_ex = torch.eye(4, dtype=torch.float32, device=device)[None, None]
            sample_ix = torch.eye(3, dtype=torch.float32, device=device)[None, None]
        with torch.inference_mode():
            out = self._raw_forward(sample_image, sample_ex, sample_ix)
        return [k for k in self._CANDIDATE_OUTPUTS if out.get(k) is not None]

    def _raw_forward(self, image, extrinsics, intrinsics):
        """The actual forward, returning the raw output dict."""
        if extrinsics is not None:
            cam_token = self.net.cam_enc(extrinsics, intrinsics, image.shape[-2:])
        else:
            cam_token = None

        feats, _aux = self.net.backbone(
            image,
            cam_token=cam_token,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )
        H, W = image.shape[-2], image.shape[-1]

        output = self.net._process_depth_head(feats, H, W)
        if self.use_ray_pose:
            output = self.net._process_ray_pose_estimation(output, H, W)
        else:
            output = self.net._process_camera_estimation(feats, H, W, output)
        return output

    def forward(self, image, extrinsics=None, intrinsics=None):
        # Replicate DepthAnything3Net.forward but skip the post-processing
        # branches that break torch.export / torch.jit.trace.
        ex = extrinsics if self.use_camera else None
        ix = intrinsics if self.use_camera else None
        output = self._raw_forward(image, ex, ix)

        # Only emit non-None outputs; ordering must match `which_outputs()`.
        out_tuple = tuple(
            output[k] for k in self._CANDIDATE_OUTPUTS if output.get(k) is not None
        )
        return out_tuple


class _MetricWrapper(nn.Module):
    """Wraps the metric sub-net (image-only forward). Outputs depth + sky."""

    def __init__(self, metric_net: nn.Module) -> None:
        super().__init__()
        self.metric_net = metric_net

    def forward(self, image):
        out = self.metric_net(image)
        depth = out.get("depth")
        sky = out.get("sky")
        return depth, sky


def _load_torch_model(model_id_or_path: str):
    """Local import so this file can be imported by the export tool without
    pulling DepthAnything3 until needed."""
    from depth_anything_3.api import DepthAnything3

    if os.path.isdir(model_id_or_path) or os.path.isfile(model_id_or_path):
        model = DepthAnything3.from_pretrained(model_id_or_path)
    else:
        model = DepthAnything3.from_pretrained(model_id_or_path)
    model.eval()
    return model


def _make_dummy_inputs(
    *,
    views: int,
    H: int,
    W: int,
    use_camera: bool,
    device: torch.device,
):
    image = torch.randn(1, views, 3, H, W, dtype=torch.float32, device=device)
    if not use_camera:
        return (image,)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)[None, None].repeat(
        1, views, 1, 1
    )
    intrinsics = torch.eye(3, dtype=torch.float32, device=device)[None, None].repeat(
        1, views, 1, 1
    )
    intrinsics[..., 0, 0] = W / 2.0
    intrinsics[..., 1, 1] = H / 2.0
    intrinsics[..., 0, 2] = W / 2.0
    intrinsics[..., 1, 2] = H / 2.0
    return (image, extrinsics, intrinsics)


def export_depth_anything_3(
    *,
    model_id_or_path: str,
    out_path: str,
    with_camera: bool = False,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    views: int = _DEFAULT_VIEWS,
    height: int = _DEFAULT_H,
    width: int = _DEFAULT_W,
    opset_version: int = 17,
    device: str = "cuda",
    do_constant_folding: bool = True,
) -> str:
    """Export a single-net DepthAnything3 model to a single ONNX file.

    Returns the path to the written ``.onnx`` file.

    Parameters
    ----------
    model_id_or_path
        HF Hub repo id (e.g. ``depth-anything/DA3-LARGE-1.1``) or a local path
        understood by :meth:`DepthAnything3.from_pretrained`.
    out_path
        Destination ``.onnx`` file path. Parent dirs are created.
    with_camera
        If True, the exported graph takes ``extrinsics`` and ``intrinsics`` as
        additional inputs. Two separate exports are typically shipped (one
        with, one without) since the choice is a control-flow split inside
        the torch model.
    use_ray_pose
        Baked-in value of the ``use_ray_pose`` flag.
    ref_view_strategy
        Baked-in value of the reference-view selection strategy. Note: this
        is consumed by Python control flow inside the backbone, so the
        choice is frozen at export time.
    views, height, width
        Dummy-input shapes. The exported graph has dynamic axes for all three
        so the values at inference time can differ; pick something
        representative for tracer-friendliness.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu" and device != "cpu":
        # Be loud but don't fail -- some setups deliberately export on CPU.
        print(f"[onnx-export] CUDA not available; exporting on CPU.")

    torch_model = _load_torch_model(model_id_or_path).to(dev)
    underlying = getattr(torch_model, "model", torch_model)
    if underlying.__class__.__name__ == "NestedDepthAnything3Net":
        raise ValueError(
            "export_depth_anything_3() received a NestedDepthAnything3Net. "
            "Use export_depth_anything_3_nested() instead."
        )

    wrapper = _SingleNetWrapper(
        underlying,
        use_camera=with_camera,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    ).to(dev).eval()

    dummy = _make_dummy_inputs(
        views=views, H=height, W=width, use_camera=with_camera, device=dev
    )

    input_names = ["image"] + (["extrinsics", "intrinsics"] if with_camera else [])

    # Discover the actual output fields for this specific checkpoint (some
    # variants don't produce conf / extrinsics / sky).
    output_names = wrapper.which_outputs()

    # Only ``N`` (number of views) is left dynamic. H/W are pinned at the
    # trace shape -- callers should square-resize their inputs to (H, W)
    # before feeding the model and unsquare the depth output afterwards.
    _dynamic_per_output = {
        "depth": {1: "N"},
        "depth_conf": {1: "N"},
        "extrinsics": {1: "N"},
        "intrinsics": {1: "N"},
        "sky": {1: "N"},
    }
    dynamic_axes = {"image": {1: "N"}}
    for name in output_names:
        dynamic_axes[name] = _dynamic_per_output[name]
    if with_camera:
        dynamic_axes["extrinsics"] = {1: "N"}
        dynamic_axes["intrinsics"] = {1: "N"}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            # Use the legacy TorchScript-based exporter. The new dynamo exporter
            # cannot trace some data-dependent control flow inside the backbone.
            dynamo=False,
        )
    print(f"[onnx-export] wrote {out_path} (opset={opset_version})")
    _try_check(out_path)
    return out_path


def export_depth_anything_3_nested(
    *,
    model_id_or_path: str,
    out_main_path: str,
    out_metric_path: str,
    with_camera: bool = False,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    views: int = _DEFAULT_VIEWS,
    height: int = _DEFAULT_H,
    width: int = _DEFAULT_W,
    opset_version: int = 17,
    device: str = "cuda",
    do_constant_folding: bool = True,
) -> tuple[str, str]:
    """Export a NestedDepthAnything3Net as two ONNX files (main + metric).

    The metric branch is exported with image-only inputs and produces
    ``(depth, sky)``. The main branch is exported just like the single-net
    case.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch_model = _load_torch_model(model_id_or_path).to(dev)
    underlying = getattr(torch_model, "model", torch_model)
    if underlying.__class__.__name__ != "NestedDepthAnything3Net":
        raise ValueError(
            f"export_depth_anything_3_nested() requires a NestedDepthAnything3Net; "
            f"got {underlying.__class__.__name__}. "
            f"Use export_depth_anything_3() instead."
        )

    # ---------------- main branch ----------------
    main_wrapper = _SingleNetWrapper(
        underlying.da3,
        use_camera=with_camera,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    ).to(dev).eval()
    main_dummy = _make_dummy_inputs(
        views=views, H=height, W=width, use_camera=with_camera, device=dev
    )
    main_input_names = ["image"] + (["extrinsics", "intrinsics"] if with_camera else [])
    main_output_names = main_wrapper.which_outputs()
    _main_dyn_per_output = {
        "depth": {1: "N"},
        "depth_conf": {1: "N"},
        "extrinsics": {1: "N"},
        "intrinsics": {1: "N"},
        "sky": {1: "N"},
    }
    main_dynamic = {"image": {1: "N"}}
    for name in main_output_names:
        main_dynamic[name] = _main_dyn_per_output[name]

    os.makedirs(os.path.dirname(out_main_path) or ".", exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            main_wrapper,
            main_dummy,
            out_main_path,
            input_names=main_input_names,
            output_names=main_output_names,
            dynamic_axes=main_dynamic,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )
    print(f"[onnx-export] wrote main: {out_main_path}")
    _try_check(out_main_path)

    # ---------------- metric branch ----------------
    metric_wrapper = _MetricWrapper(underlying.da3_metric).to(dev).eval()
    metric_dummy = (
        torch.randn(1, views, 3, height, width, dtype=torch.float32, device=dev),
    )
    metric_dynamic = {
        "image": {1: "N"},
        "depth": {1: "N"},
        "sky": {1: "N"},
    }
    os.makedirs(os.path.dirname(out_metric_path) or ".", exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            metric_wrapper,
            metric_dummy,
            out_metric_path,
            input_names=["image"],
            output_names=["depth", "sky"],
            dynamic_axes=metric_dynamic,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )
    print(f"[onnx-export] wrote metric: {out_metric_path}")
    _try_check(out_metric_path)

    return out_main_path, out_metric_path


def _try_check(path: str) -> None:
    """Run ``onnx.checker.check_model`` if `onnx` is importable."""
    try:
        import onnx
    except ImportError:
        print(
            "[onnx-export] `onnx` package not installed; skipping checker. "
            "Install `onnx` to validate the exported file."
        )
        return
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"[onnx-export] {path} passed onnx.checker")
    except Exception as e:
        print(f"[onnx-export] WARNING: {path} failed onnx.checker: {e}")


__all__ = [
    "export_depth_anything_3",
    "export_depth_anything_3_nested",
]
