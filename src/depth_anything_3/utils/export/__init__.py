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

from depth_anything_3.specs import Prediction

from .depth_vis import export_to_depth_vis
from .feat_vis import export_to_feat_vis
from .glb import export_to_glb
from .npz import export_to_mini_npz, export_to_npz

try:
    from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video
except Exception as exc:  # noqa: BLE001
    export_to_gs_ply = None
    export_to_gs_video = None
    _GS_IMPORT_ERROR = exc
else:
    _GS_IMPORT_ERROR = None

try:
    from .colmap import export_to_colmap
except Exception as exc:  # noqa: BLE001
    export_to_colmap = None
    _COLMAP_IMPORT_ERROR = exc
else:
    _COLMAP_IMPORT_ERROR = None


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        export_formats = export_format.split("-")
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return  # Prevent falling through to single-format handling

    if export_format == "glb":
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        if export_to_gs_ply is None:
            raise ImportError(
                "gs_ply export requires optional dependencies (e.g., moviepy/gsplat)."
            ) from _GS_IMPORT_ERROR
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        if export_to_gs_video is None:
            raise ImportError(
                "gs_video export requires optional dependencies (e.g., moviepy/gsplat)."
            ) from _GS_IMPORT_ERROR
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        if export_to_colmap is None:
            raise ImportError(
                "colmap export requires optional dependency `pycolmap`. "
                "Install pycolmap wheel or build COLMAP/PyCOLMAP from source."
            ) from _COLMAP_IMPORT_ERROR
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    export,
]
