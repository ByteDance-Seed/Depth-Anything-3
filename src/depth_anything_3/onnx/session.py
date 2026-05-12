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

"""Thin onnxruntime wrapper.

Resolves a sensible default provider list (CUDA if available, else CPU) and
hides the boilerplate of feeding optional inputs (`extrinsics`, `intrinsics`)
that may or may not exist in a given exported graph.

Adding a new execution provider (e.g. ``TensorrtExecutionProvider``) is just a
matter of passing ``providers=[...]`` -- no class changes are required.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import os
import time
import numpy as np

from depth_anything_3.utils.logger import logger


_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def _resolve_providers(
    providers: Sequence[str | tuple[str, dict]] | None,
) -> list[str | tuple[str, dict]]:
    """Normalise a user-friendly provider list."""
    import onnxruntime as ort  # local import: this module is the only place that needs ORT

    available = set(ort.get_available_providers())

    if providers is None:
        # Prefer CUDA if installed, otherwise CPU.
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    resolved: list[str | tuple[str, dict]] = []
    for p in providers:
        if isinstance(p, tuple):
            name = _PROVIDER_ALIASES.get(p[0].lower(), p[0])
            resolved.append((name, p[1]))
            continue
        name = _PROVIDER_ALIASES.get(p.lower(), p)
        resolved.append(name)
    return resolved


class OrtSession:
    """Wraps a single ONNX file and runs it.

    The wrapped graph is expected to take ``image`` as a required input and
    optionally ``extrinsics`` / ``intrinsics``. Outputs are returned as a dict
    keyed by the graph's output names (e.g. ``depth``, ``depth_conf``,
    ``extrinsics``, ``intrinsics``, ``sky``).
    """

    def __init__(
        self,
        model_path: str,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = session_options or ort.SessionOptions()
        resolved = _resolve_providers(providers)

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=resolved
        )
        self.model_path = model_path
        self.providers = self.session.get_providers()
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"OrtSession: loaded {model_path} with providers={self.providers}")

    @property
    def input_names(self) -> set[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def run(
        self,
        image: np.ndarray,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        extra: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the session. Returns ``{name: ndarray}``."""
        feeds: dict[str, np.ndarray] = {}
        # Required
        if "image" not in self._input_names:
            raise RuntimeError(
                f"ONNX model {self.model_path} has no input named 'image'. "
                f"Available: {sorted(self._input_names)}"
            )
        feeds["image"] = np.ascontiguousarray(image, dtype=np.float32)
        # Optional
        if extrinsics is not None and "extrinsics" in self._input_names:
            feeds["extrinsics"] = np.ascontiguousarray(extrinsics, dtype=np.float32)
        if intrinsics is not None and "intrinsics" in self._input_names:
            feeds["intrinsics"] = np.ascontiguousarray(intrinsics, dtype=np.float32)
        if extra:
            for k, v in extra.items():
                if k in self._input_names:
                    feeds[k] = np.ascontiguousarray(v)

        # Missing inputs that the user did not provide -> warn loudly; many models
        # expose `extrinsics`/`intrinsics` as required inputs and ORT will raise.
        missing = self._input_names - feeds.keys()
        if missing:
            raise RuntimeError(
                f"ONNX model {self.model_path} requires inputs {sorted(missing)} "
                "that were not supplied. Re-export with those inputs baked in if "
                "you don't have them at inference time."
            )

        start = time.time()
        outputs = self.session.run(self._output_names, feeds)
        logger.info(f"OrtSession.run: {time.time() - start:.3f}s")
        return dict(zip(self._output_names, outputs))


__all__ = ["OrtSession"]
