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

"""NumPy-only port of `depth_anything_3.utils.io.output_processor.OutputProcessor`.

Accepts a `dict[str, np.ndarray]` from an `onnxruntime.InferenceSession.run`
call and produces a `Prediction`. No torch import.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from addict import Dict as AddictDict

from depth_anything_3.specs import Prediction


class NumpyOutputProcessor:
    """Drop-in replacement for `OutputProcessor` (numpy in, numpy out)."""

    def __call__(self, model_output: Mapping[str, Any]) -> Prediction:
        depth = self._extract_depth(model_output)
        conf = self._extract_conf(model_output)
        extrinsics = self._extract_extrinsics(model_output)
        intrinsics = self._extract_intrinsics(model_output)
        sky = self._extract_sky(model_output)
        aux = self._extract_aux(model_output)

        # `is_metric` and `scale_factor` are not produced by the ONNX graph;
        # they are filled in by the Nested API after metric alignment.
        return Prediction(
            depth=depth,
            sky=sky,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            is_metric=int(model_output.get("is_metric", 0)),
            gaussians=None,  # GS path is not supported via ONNX
            aux=aux,
            scale_factor=model_output.get("scale_factor", None),
        )

    # -----------------------------
    # Field extractors
    # -----------------------------
    @staticmethod
    def _extract_depth(model_output: Mapping[str, Any]) -> np.ndarray:
        depth = np.asarray(model_output["depth"])
        # Accepted input shapes:
        #   (B, N, 1, H, W) / (B, N, H, W, 1) / (B, N, H, W) -> (N, H, W)
        if depth.ndim == 5:
            depth = depth[0]
            if depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            elif depth.shape[1] == 1:
                depth = depth.squeeze(1)
        elif depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        return depth

    @staticmethod
    def _extract_conf(model_output: Mapping[str, Any]) -> np.ndarray | None:
        conf = model_output.get("depth_conf", None)
        if conf is None:
            return None
        conf = np.asarray(conf)
        if conf.ndim == 5:
            conf = conf[0]
            if conf.shape[-1] == 1:
                conf = conf.squeeze(-1)
            elif conf.shape[1] == 1:
                conf = conf.squeeze(1)
        elif conf.ndim == 4 and conf.shape[0] == 1:
            conf = conf[0]
        return conf

    @staticmethod
    def _extract_extrinsics(model_output: Mapping[str, Any]) -> np.ndarray | None:
        ext = model_output.get("extrinsics", None)
        if ext is None:
            return None
        ext = np.asarray(ext)
        if ext.ndim == 4:  # (B, N, 4, 4) or (B, N, 3, 4)
            ext = ext[0]
        return ext

    @staticmethod
    def _extract_intrinsics(model_output: Mapping[str, Any]) -> np.ndarray | None:
        ixt = model_output.get("intrinsics", None)
        if ixt is None:
            return None
        ixt = np.asarray(ixt)
        if ixt.ndim == 4:  # (B, N, 3, 3)
            ixt = ixt[0]
        return ixt

    @staticmethod
    def _extract_sky(model_output: Mapping[str, Any]) -> np.ndarray | None:
        sky = model_output.get("sky", None)
        if sky is None:
            return None
        sky = np.asarray(sky)
        if sky.ndim == 5:
            sky = sky[0]
            if sky.shape[-1] == 1:
                sky = sky.squeeze(-1)
            elif sky.shape[1] == 1:
                sky = sky.squeeze(1)
        elif sky.ndim == 4 and sky.shape[0] == 1:
            sky = sky[0]
        return sky >= 0.5

    @staticmethod
    def _extract_aux(model_output: Mapping[str, Any]) -> AddictDict:
        aux = model_output.get("aux", None)
        ret = AddictDict()
        if aux is None:
            return ret
        for k, v in aux.items():
            if isinstance(v, np.ndarray) and v.ndim >= 4:
                ret[k] = v[0]  # drop batch axis
            else:
                ret[k] = v
        return ret


__all__ = ["NumpyOutputProcessor"]
