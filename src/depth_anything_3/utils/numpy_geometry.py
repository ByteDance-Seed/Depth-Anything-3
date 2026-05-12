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

"""NumPy-only geometry helpers.

These were factored out of `utils/geometry.py` (which has a hard `import
torch` at module level) so the ONNX inference path -- and `pose_align` --
can import them without pulling torch in.
"""

from __future__ import annotations

import numpy as np


def transpose_last_two_axes(arr: np.ndarray) -> np.ndarray:
    """Swap the last two axes. Equivalent to ``arr.mT`` for torch tensors but
    works on numpy<2."""
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) (...,4,4) or (...,3,4) extrinsic matrix."""
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    Rt = transpose_last_two_axes(R)
    return np.concatenate(
        [np.concatenate([Rt, -Rt @ T], axis=-1), P],
        axis=-2,
    )
