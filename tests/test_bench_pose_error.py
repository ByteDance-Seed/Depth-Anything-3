"""Regression tests for world-to-camera relative pose error in bench.utils."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


def _build_torch_shim() -> types.ModuleType:
    """Construct a minimal NumPy-backed torch module for pose metric unit tests."""

    class _Tensor:
        def __init__(self, data):
            arr = np.asarray(data)
            if arr.dtype == np.float32:
                arr = arr.astype(np.float64)
            self._arr = arr

        @property
        def shape(self):
            return self._arr.shape

        @property
        def dtype(self):
            return self._arr.dtype

        @property
        def device(self):
            return "cpu"

        def __len__(self):
            return len(self._arr)

        def __getitem__(self, idx):
            if isinstance(idx, _Tensor):
                idx = idx._arr.astype(np.int64)
            return _Tensor(self._arr[idx])

        def __add__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(self._arr + other_val)

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(self._arr - other_val)

        def __rsub__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(other_val - self._arr)

        def __mul__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(self._arr * other_val)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(self._arr / other_val)

        def __neg__(self):
            return _Tensor(-self._arr)

        def __pow__(self, power):
            return _Tensor(self._arr**power)

        def __or__(self, other):
            other_val = other._arr if isinstance(other, _Tensor) else other
            return _Tensor(np.logical_or(self._arr, other_val))

        def __setitem__(self, idx, value):
            if isinstance(idx, _Tensor):
                idx = idx._arr
            val = value._arr if isinstance(value, _Tensor) else value
            self._arr[idx] = val

        def bmm(self, other):
            return _Tensor(np.matmul(self._arr, other._arr))

        def transpose(self, d0, d1):
            axes = list(range(self._arr.ndim))
            axes[d0], axes[d1] = axes[d1], axes[d0]
            return _Tensor(np.transpose(self._arr, axes))

        def repeat(self, *sizes):
            return _Tensor(np.tile(self._arr, sizes))

        def reshape(self, *shape):
            return _Tensor(self._arr.reshape(*shape))

        def unbind(self, dim):
            return tuple(_Tensor(x) for x in np.moveaxis(self._arr, dim, 0))

        def sum(self, dim=None):
            return _Tensor(np.sum(self._arr, axis=dim))

        def clamp(self, min=None, max=None):
            return _Tensor(np.clip(self._arr, min, max))

        def abs(self):
            return _Tensor(np.abs(self._arr))

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _Tensor
    torch_mod.from_numpy = lambda arr: _Tensor(arr)
    torch_mod.tensor = lambda data: _Tensor(data)
    torch_mod.eye = lambda n, m=None: _Tensor(np.eye(n, m or n, dtype=np.float64))
    torch_mod.arange = lambda n: _Tensor(np.arange(n, dtype=np.int64))
    torch_mod.matmul = lambda a, b: _Tensor(np.matmul(a._arr, b._arr))
    torch_mod.bmm = lambda a, b: _Tensor(np.matmul(a._arr, b._arr))
    torch_mod.arccos = lambda x: _Tensor(np.arccos(x._arr))
    torch_mod.acos = lambda x: _Tensor(
        np.acos(x._arr) if hasattr(np, "acos") else np.arccos(x._arr)
    )
    torch_mod.clamp_min = lambda x, min_val: _Tensor(np.maximum(x._arr, min_val))
    torch_mod.sum = lambda x, dim=None: _Tensor(np.sum(x._arr, axis=dim))
    torch_mod.sqrt = lambda x: _Tensor(np.sqrt(x._arr))
    torch_mod.isnan = lambda x: _Tensor(np.isnan(x._arr))
    torch_mod.isinf = lambda x: _Tensor(np.isinf(x._arr))
    torch_mod.min = lambda a, b: _Tensor(np.minimum(a._arr, b._arr))
    torch_mod.norm = lambda x, dim=None, keepdim=False: _Tensor(
        np.linalg.norm(x._arr, axis=dim, keepdims=keepdim)
    )

    def _combinations(t, r=2, with_replacement=False):
        n = len(t._arr)
        pairs = [(t._arr[i], t._arr[j]) for i in range(n) for j in range(i + 1, n)]
        return _Tensor(np.array(pairs, dtype=np.int64))

    torch_mod.combinations = _combinations
    return torch_mod


def _load_bench_utils():
    """Load depth_anything_3.bench.utils without leaking fallback shims into sys.modules."""
    shims: dict[str, types.ModuleType] = {}

    try:
        importlib.import_module("open3d")
    except ImportError:
        o3d_mod = types.ModuleType("open3d")
        o3d_mod.geometry = types.SimpleNamespace(PointCloud=object, TriangleMesh=object)
        o3d_mod.pipelines = types.SimpleNamespace(
            integration=types.SimpleNamespace(ScalableTSDFVolume=object)
        )
        shims["open3d"] = o3d_mod

    try:
        importlib.import_module("addict")
    except ImportError:

        class _AttrDict(dict):
            def __getattr__(self, key):
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        addict_mod = types.ModuleType("addict")
        addict_mod.Dict = _AttrDict
        shims["addict"] = addict_mod

    try:
        torch_api = importlib.import_module("torch")
    except ImportError:
        torch_api = _build_torch_shim()
        shims["torch"] = torch_api
        geom_mod = types.ModuleType("depth_anything_3.utils.geometry")
        geom_mod.mat_to_quat = lambda mat: torch_api.from_numpy(
            Rotation.from_matrix(mat.numpy()).as_quat()
        )
        shims["depth_anything_3.utils.geometry"] = geom_mod
    else:
        try:
            importlib.import_module("depth_anything_3.utils.geometry")
        except ImportError:
            geom_mod = types.ModuleType("depth_anything_3.utils.geometry")
            geom_mod.mat_to_quat = lambda mat: torch_api.from_numpy(
                Rotation.from_matrix(mat.detach().cpu().numpy()).as_quat()
            ).to(dtype=mat.dtype, device=mat.device)
            shims["depth_anything_3.utils.geometry"] = geom_mod

    if not shims:
        return torch_api, importlib.import_module("depth_anything_3.bench.utils")

    with patch.dict(sys.modules):
        if "depth_anything_3.utils.geometry" in shims:
            importlib.import_module("depth_anything_3.utils")
        sys.modules.update(shims)
        sys.modules.pop("depth_anything_3.bench.utils", None)
        bench_utils_mod = importlib.import_module("depth_anything_3.bench.utils")
    return torch_api, bench_utils_mod


torch, bench_utils = _load_bench_utils()


def _make_se3(rotvec: list[float], translation: list[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    mat[:3, 3] = translation
    return mat


def _sample_w2c_trajectory() -> np.ndarray:
    return np.stack(
        [
            _make_se3([0.3, -0.4, 0.2], [1.2, -0.5, 2.0]),
            _make_se3([-0.2, 0.5, 0.1], [-0.8, 1.1, 3.5]),
            _make_se3([0.6, 0.2, -0.5], [0.4, -1.3, 4.2]),
            _make_se3([-0.5, -0.3, 0.4], [2.1, 0.7, 1.8]),
            _make_se3([0.1, 0.7, -0.2], [-1.5, -0.2, 5.0]),
        ],
        axis=0,
    )


def test_relative_pose_error_unaffected_by_frame0_error_on_other_pairs():
    gt_np = _sample_w2c_trajectory()
    pred_np = gt_np.copy()
    # Perturb only camera 0; relative poses among cameras 1..4 are exact.
    err0 = _make_se3([0.5, -0.3, 0.4], [1.5, -2.0, 0.8])
    pred_np[0] = err0 @ pred_np[0]

    gt_aligned = bench_utils.align_to_first_camera(torch.from_numpy(gt_np))
    pred_aligned = bench_utils.align_to_first_camera(torch.from_numpy(pred_np))

    rel_r, rel_t = bench_utils.se3_to_relative_pose_error(pred_aligned, gt_aligned, len(gt_np))
    rel_r_np = rel_r.cpu().numpy()
    rel_t_np = rel_t.cpu().numpy()

    i1, _i2 = bench_utils.build_pair_index(len(gt_np))
    non_first_mask = i1.cpu().numpy() >= 1

    np.testing.assert_allclose(rel_r_np[non_first_mask], 0.0, atol=1e-5)
    np.testing.assert_allclose(rel_t_np[non_first_mask], 0.0, atol=1e-5)


def test_se3_to_relative_pose_error_invariant_to_world_frame_transform():
    gt_np = _sample_w2c_trajectory()
    # Right-multiplying all w2c poses by a rigid world transform W leaves all
    # pairwise relative camera poses unchanged.
    world_transform = _make_se3([0.9, -0.6, 0.3], [3.0, -1.5, 2.5])
    pred_np = gt_np @ world_transform

    rel_r, rel_t = bench_utils.se3_to_relative_pose_error(
        torch.from_numpy(pred_np),
        torch.from_numpy(gt_np),
        len(gt_np),
    )

    np.testing.assert_allclose(rel_r.cpu().numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(rel_t.cpu().numpy(), 0.0, atol=1e-5)


def test_compute_pose_rotation_errors_invariant_to_frame_order():
    gt_np = _sample_w2c_trajectory()
    rng = np.random.default_rng(123)
    pred_np = gt_np.copy()
    for idx in range(len(pred_np)):
        noise = _make_se3(
            (rng.normal(scale=0.15, size=3)).tolist(),
            (rng.normal(scale=0.3, size=3)).tolist(),
        )
        pred_np[idx] = noise @ pred_np[idx]

    perm = [3, 1, 4, 0, 2]
    gt_aligned = bench_utils.align_to_first_camera(torch.from_numpy(gt_np))
    pred_aligned = bench_utils.align_to_first_camera(torch.from_numpy(pred_np))
    rel_r_orig, _ = bench_utils.se3_to_relative_pose_error(pred_aligned, gt_aligned, len(gt_np))

    gt_perm_aligned = bench_utils.align_to_first_camera(torch.from_numpy(gt_np[perm]))
    pred_perm_aligned = bench_utils.align_to_first_camera(torch.from_numpy(pred_np[perm]))
    rel_r_perm, _ = bench_utils.se3_to_relative_pose_error(
        pred_perm_aligned, gt_perm_aligned, len(gt_np)
    )

    np.testing.assert_allclose(
        np.sort(rel_r_orig.cpu().numpy()),
        np.sort(rel_r_perm.cpu().numpy()),
        atol=1e-5,
    )


def test_compute_pose_end_to_end_auc_and_single_camera_error():
    gt_np = _sample_w2c_trajectory()
    pred_np = gt_np.copy()
    # Apply a 15-degree rotation error solely to camera 1.
    angle_rad = np.deg2rad(15.0)
    pred_np[1] = _make_se3([0.0, 0.0, angle_rad], [0.0, 0.0, 0.0]) @ pred_np[1]

    metrics = bench_utils.compute_pose(torch.from_numpy(pred_np), torch.from_numpy(gt_np))
    assert set(metrics.keys()) == {"auc30", "auc15", "auc05", "auc03"}
    assert 0.0 < metrics["auc30"] < 1.0

    # Pairs not involving camera 1 (6 of 10) still have 0 error.
    gt_aligned = bench_utils.align_to_first_camera(torch.from_numpy(gt_np))
    pred_aligned = bench_utils.align_to_first_camera(torch.from_numpy(pred_np))
    rel_r, rel_t = bench_utils.se3_to_relative_pose_error(pred_aligned, gt_aligned, len(gt_np))
    i1, i2 = bench_utils.build_pair_index(len(gt_np))
    involves_cam1 = (i1.cpu().numpy() == 1) | (i2.cpu().numpy() == 1)
    np.testing.assert_allclose(rel_r.cpu().numpy()[involves_cam1], 15.0, atol=1e-5)
    np.testing.assert_allclose(rel_r.cpu().numpy()[~involves_cam1], 0.0, atol=1e-5)
    np.testing.assert_allclose(rel_t.cpu().numpy()[~involves_cam1], 0.0, atol=1e-5)


def test_nonzero_baseline_translation_angle_independent_of_camera0_rotation():
    gt_np = np.stack(
        [
            _make_se3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            _make_se3([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            _make_se3([0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]),
        ],
        axis=0,
    )
    pred_np = gt_np.copy()
    # Rotate camera 0 by 90 degrees about Y so the old conjugated formula would
    # rotate pair (1, 2)'s [-1, -1, 0] baseline into [0, -1, 1] (90 deg vs 45 deg).
    pred_np[0] = _make_se3([0.0, np.pi / 2, 0.0], [3.0, -2.0, 1.0])
    pred_np[2] = _make_se3([0.0, 0.0, 0.0], [-2.0, -1.0, 0.0])

    gt_aligned = bench_utils.align_to_first_camera(torch.from_numpy(gt_np))
    pred_aligned = bench_utils.align_to_first_camera(torch.from_numpy(pred_np))
    rel_r, rel_t = bench_utils.se3_to_relative_pose_error(pred_aligned, gt_aligned, len(gt_np))

    # Pair index 2 corresponds to (i1=1, i2=2), where gt baseline is [-1, 0, 0]
    # and pred baseline is [-1, -1, 0] (exact 45-degree angle, 0-degree rotation).
    np.testing.assert_allclose(rel_r.cpu().numpy()[2], 0.0, atol=1e-5)
    np.testing.assert_allclose(rel_t.cpu().numpy()[2], 45.0, atol=1e-5)
