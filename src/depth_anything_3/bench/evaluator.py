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

"""
Main Evaluator class for DepthAnything3 benchmark evaluation.

Supports multiple datasets and evaluation modes:
- pose: Camera pose estimation (AUC metrics)
- recon_unposed: 3D reconstruction with predicted poses
- recon_posed: 3D reconstruction with GT poses
- view_syn: Novel view synthesis (TODO)
"""

import json
import os
import random
from typing import Dict as TDict, Iterable, List

import numpy as np
import torch
from addict import Dict
from tqdm import tqdm

from depth_anything_3.bench.depth_metrics import (
    DepthEvalConfig,
    compute_depth_metrics,
    resize_depth_nearest,
)
from depth_anything_3.bench.print_metrics import MetricsPrinter
from depth_anything_3.utils.parallel_utils import parallel_execution
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY


class Evaluator:
    """
    Main evaluation orchestrator for DepthAnything3 benchmarks.

    Usage:
        evaluator = Evaluator(
            work_dir="./eval_workspace",
            datas=["dtu"],
            modes=["pose", "recon_unposed", "recon_posed"],
        )
        api = DepthAnything3.from_pretrained("...")
        evaluator.infer(api)
        metrics = evaluator.eval()
        evaluator.print_metrics()
    """

    VALID_MODES = {"pose", "recon_unposed", "recon_posed", "view_syn", "rel_depth", "metric_depth"}

    def __init__(
        self,
        work_dir: str = "./eval_workspace",
        datas: List[str] = ("dtu",),
        modes: List[str] = ("recon_unposed",),
        ref_view_strategy: str = EVAL_REF_VIEW_STRATEGY,
        scenes: List[str] = None,
        debug: bool = False,
        num_fusion_workers: int = 4,
        max_frames: int = 100,
        gpu_id: int = 0,
        total_gpus: int = 1,
    ):
        """
        Initialize the evaluator.

        Args:
            work_dir: Base directory for model outputs and metric files
            datas: List of dataset names (must be registered in MV_REGISTRY)
            modes: List of evaluation modes to run
            ref_view_strategy: Reference view selection strategy for inference
                               ("first", "saddle_balanced", etc.)
            scenes: Specific scenes to evaluate (None = all scenes)
            debug: Enable verbose debug output
            num_fusion_workers: Number of parallel workers for TSDF fusion (default: 4)
            max_frames: Maximum number of frames per scene (default: 100).
                        If a scene has more frames, randomly sample to this limit.
                        Set to -1 to disable sampling.
            gpu_id: GPU index for multi-GPU (0-indexed)
            total_gpus: Total number of GPUs for task distribution
        """
        self.work_dir = work_dir
        self.datas = list(datas)
        self.modes = set(modes)
        self.ref_view_strategy = ref_view_strategy
        self.scenes_filter = scenes
        self.debug = debug
        self.num_fusion_workers = num_fusion_workers
        self.max_frames = max_frames
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus

        # Validate modes
        unknown = self.modes - self.VALID_MODES
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}. Valid: {sorted(self.VALID_MODES)}")

        os.makedirs(self.work_dir, exist_ok=True)

        # Initialize datasets
        self.datasets = Dict()
        for data in self.datas:
            if not MV_REGISTRY.has(data):
                available = list(MV_REGISTRY.all().keys())
                raise ValueError(f"Dataset '{data}' not found. Available: {available}")
            self.datasets[data] = MV_REGISTRY.get(data)()

        # Initialize metrics printer
        self._printer = MetricsPrinter()

    # -------------------- Public APIs -------------------- #

    def all(self, api) -> TDict[str, dict]:
        """
        Run complete evaluation pipeline: inference + evaluation.

        Args:
            api: DepthAnything3 API instance

        Returns:
            Combined metrics dictionary
        """
        self.infer(api)
        return self.eval()

    def _get_scenes(self, dataset) -> List[str]:
        """Get list of scenes to evaluate, optionally filtered."""
        all_scenes = dataset.SCENES
        if self.scenes_filter:
            # Normalize scenes_filter to a list of strings (handles single str/int inputs).
            if isinstance(self.scenes_filter, (str, int)):
                scene_filter = [str(self.scenes_filter)]
            else:
                scene_filter = [str(s) for s in self.scenes_filter]

            scenes = [s for s in all_scenes if s in scene_filter]
            if self.debug:
                print(f"[DEBUG] Filtered scenes: {scenes} (from {len(all_scenes)} total)")
            return scenes
        return all_scenes

    def infer(self, api, model_path: str = None) -> None:
        """
        Run inference according to requested modes.

        - Unposed export if 'pose' or 'recon_unposed' is in modes
        - Posed export if 'recon_posed' or 'view_syn' is in modes

        Multi-GPU: Use --gpu_id and --total_gpus to distribute tasks.
        Example: Launch 4 processes with gpu_id=0,1,2,3 and total_gpus=4

        Args:
            api: DepthAnything3 API instance
            model_path: Model path (unused, kept for API compatibility)
        """
        need_unposed = {"pose", "recon_unposed", "rel_depth", "metric_depth"} & self.modes
        need_posed = {"recon_posed", "view_syn"} & self.modes
        export_format = "mini_npz-glb" if self.debug else "mini_npz"
        # export_format="mini_npz-glb-npz-colmap"

        # Collect all tasks
        all_tasks = []
        for data in self.datas:
            dataset = self.datasets[data]
            for scene in self._get_scenes(dataset):
                all_tasks.append((data, scene))

        # Distribute tasks across GPUs
        if self.total_gpus > 1:
            tasks = [t for i, t in enumerate(all_tasks) if i % self.total_gpus == self.gpu_id]
            print(f"[INFO] GPU {self.gpu_id}/{self.total_gpus}: {len(tasks)}/{len(all_tasks)} tasks")
        else:
            tasks = all_tasks
            print(f"[INFO] Total inference tasks: {len(tasks)}")

        for data, scene in tqdm(tasks, desc=f"Inference (GPU {self.gpu_id})"):
            dataset = self.datasets[data]
            scene_data = dataset.get_data(scene)
            scene_data = self._sample_frames(scene_data, scene)

            if need_unposed:
                export_dir = self._export_dir(data, scene, posed=False)
                api.inference(
                    scene_data.image_files,
                    export_dir=export_dir,
                    export_format=export_format,
                    ref_view_strategy=self.ref_view_strategy,
                )
                self._save_gt_meta(export_dir, scene_data)

            if need_posed:
                export_dir = self._export_dir(data, scene, posed=True)
                api.inference(
                    scene_data.image_files,
                    scene_data.extrinsics,
                    scene_data.intrinsics,
                    export_dir=export_dir,
                    export_format=export_format,
                    ref_view_strategy=self.ref_view_strategy,
                )
                self._save_gt_meta(export_dir, scene_data)

    def eval(self) -> TDict[str, dict]:
        """
        Evaluate for all configured modes and write JSON files.
        
        Evaluation order by mode (all datasets per mode):
        1. pose - all datasets
        2. recon_unposed - all datasets
        3. recon_posed - all datasets

        Returns:
            Summary mapping: {"<data>_<mode>": metrics_dict}
        """
        summary: TDict[str, dict] = {}

        # Evaluate by mode (all datasets per mode)
        if "pose" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating POSE for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_pose():
                summary[f"{data}_pose"] = result

        # Depth quality (relative + metric)
        if "rel_depth" in self.modes:
            print(f"\n{'='*60}")
            print(f"📏 Evaluating RELATIVE DEPTH for all datasets...")
            print(f"{'='*60}")
            summary.update(self._eval_depth(mode="rel_depth"))

        if "metric_depth" in self.modes:
            print(f"\n{'='*60}")
            print(f"📏 Evaluating METRIC DEPTH for all datasets...")
            print(f"{'='*60}")
            summary.update(self._eval_depth(mode="metric_depth"))

        if "recon_unposed" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating RECON_UNPOSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_unposed"):
                summary[f"{data}_recon_unposed"] = result

        if "recon_posed" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating RECON_POSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_posed"):
                summary[f"{data}_recon_posed"] = result

        if "view_syn" in self.modes:
            # TODO: Add view synthesis metrics here when available
            pass

        return summary

    def print_metrics(self, metrics: TDict[str, dict] = None) -> None:
        """
        Print evaluation metrics in a beautiful tabular format.

        Args:
            metrics: Metrics dictionary. If None, loads from saved JSON files.
        """
        if metrics is None:
            metrics = self._load_metrics()

        self._printer.print_results(metrics)

    # -------------------- Evaluation Methods -------------------- #

    def _eval_pose(self) -> Iterable[tuple]:
        """Compute pose-estimation metrics for each dataset and scene."""
        os.makedirs(self._metric_dir, exist_ok=True)

        for data in tqdm(self.datas, desc="Datasets (pose eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            for scene in tqdm(scenes, desc=f"{data} scenes", leave=False):
                export_dir = self._export_dir(data, scene, posed=False)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
                
                # Check if result file exists and is valid
                if not os.path.exists(result_path):
                    print(f"\n[ERROR] Result file not found: {result_path}")
                    print(f"[ERROR] CWD: {os.getcwd()}")
                    print(f"[ERROR] Please run inference first (remove --eval_only)")
                    continue
                
                try:
                    # Use saved GT meta (handles frame sampling correctly)
                    gt_meta = self._load_gt_meta(export_dir)
                    if gt_meta is not None:
                        result = self._compute_pose_with_gt(result_path, gt_meta)
                    else:
                        # Fallback to dataset GT (no sampling was done)
                        result = dataset.eval_pose(scene, result_path)
                    dataset_results[scene] = self._to_float_dict(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to evaluate pose for {data}/{scene}: {e}")
                    print(f"[ERROR] File path: {os.path.abspath(result_path)}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            if not dataset_results:
                print(f"[WARNING] No valid results for {data}")
                continue
                
            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_pose.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    def _eval_depth(self, mode: str) -> TDict[str, dict]:
        """Evaluate depth quality against per-view GT depth maps."""
        assert mode in {"rel_depth", "metric_depth"}
        os.makedirs(self._metric_dir, exist_ok=True)

        depth_datasets = {"hiroom", "eth3d", "7scenes", "scannetpp"}
        eval_datasets = [d for d in self.datas if d in depth_datasets]

        cfg = DepthEvalConfig()

        all_metrics: TDict[str, dict] = {}
        for data in eval_datasets:
            dataset = self.datasets[data]
            dataset_results = Dict()

            scenes = self._get_scenes(dataset)
            success = 0
            for scene in scenes:
                try:
                    out_dir = self._export_dir(data, scene, posed=False)
                    result_path = os.path.join(out_dir, "exports", "mini_npz", "results.npz")
                    if not os.path.exists(result_path):
                        raise FileNotFoundError(result_path)

                    pred_npz = np.load(result_path)
                    if "depth" not in pred_npz:
                        raise KeyError("results.npz missing 'depth'")
                    pred_depths = pred_npz["depth"]  # (N,H,W)

                    # Load full GT data and map sampled frames -> full indices
                    full_gt_data = dataset.get_data(scene)

                    gt_meta_path = os.path.join(out_dir, "exports", "gt_meta.npz")
                    if os.path.exists(gt_meta_path):
                        gt_meta = np.load(gt_meta_path, allow_pickle=True)
                        sampled_files = [str(p) for p in gt_meta["image_files"]]
                        image_indices = [full_gt_data.image_files.index(f) for f in sampled_files]
                    else:
                        image_indices = list(range(len(full_gt_data.image_files)))

                    per_img_metrics = []
                    for i, img_idx in enumerate(image_indices[: pred_depths.shape[0]]):
                        pred_depth = pred_depths[i].astype(np.float32)

                        # Load GT depth + valid mask
                        gt_depth, gt_valid = self._load_gt_depth_and_mask(
                            data=data,
                            dataset=dataset,
                            scene=scene,
                            full_gt_data=full_gt_data,
                            img_idx=img_idx,
                        )

                        # Resize prediction to GT resolution if needed
                        pred_depth = resize_depth_nearest(pred_depth, gt_depth.shape[:2])

                        m = compute_depth_metrics(pred_depth, gt_depth, gt_valid, cfg=cfg)
                        if not m:
                            continue
                        per_img_metrics.append(m)

                    if not per_img_metrics:
                        raise RuntimeError("No valid GT depth pixels for evaluation.")

                    # Mean over images
                    scene_metrics = self._mean_of_dicts(per_img_metrics)
                    scene_metrics["num_views"] = float(len(per_img_metrics))
                    scene_metrics["num_eval_imgs"] = float(len(per_img_metrics))

                    # Prune per-mode JSON outputs to keep them readable:
                    #  - metric_depth: only metric/no-alignment depth quality + scale error
                    #  - rel_depth: only scale/affine-aligned (relative) depth quality
                    if mode == "metric_depth":
                        keep = [
                            "abs_rel",
                            "rmse",
                            "rmse_log",
                            "si_log",
                            *[f"delta_{t}" for t in cfg.delta_thresholds],
                            "scale_med",
                            "metric_scale_rel",
                            "metric_scale_log",
                            "valid_pixels_pct",
                            "num_views",
                            "num_eval_imgs",
                        ]
                        scene_metrics = {k: scene_metrics[k] for k in keep if k in scene_metrics}
                    elif mode == "rel_depth":
                        keep = [
                            "abs_rel_scale_med",
                            *[f"delta_{t}_scale_med" for t in cfg.delta_thresholds],
                            "abs_rel_affine_depth",
                            *[f"delta_{t}_affine_depth" for t in cfg.delta_thresholds],
                            "abs_rel_affine_disp",
                            *[f"delta_{t}_affine_disp" for t in cfg.delta_thresholds],
                            "valid_pixels_pct",
                            "num_views",
                            "num_eval_imgs",
                        ]
                        scene_metrics = {k: scene_metrics[k] for k in keep if k in scene_metrics}

                    dataset_results[scene] = scene_metrics
                    success += 1
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] depth eval failed for {data}/{scene}: {e}")

            if success == 0:
                continue

            # Aggregate over scenes
            dataset_results["mean"] = self._mean_of_dicts([v for k, v in dataset_results.items() if k != "mean"])
            dataset_results["success_rate"] = float(success) / float(len(scenes)) * 100.0
            dataset_results["_meta"] = {
                "mode": mode,
                "delta_thresholds": list(cfg.delta_thresholds),
                "note": "Depth metrics are averaged per-image then per-scene.",
            }

            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            all_metrics[f"{data}_{mode}"] = dataset_results

        return all_metrics

    def _eval_reconstruction(self, mode: str) -> Iterable[tuple]:
        """
        Compute reconstruction metrics for each dataset and scene.

        Args:
            mode: "recon_unposed" or "recon_posed"
        """
        assert mode in {"recon_unposed", "recon_posed"}
        os.makedirs(self._metric_dir, exist_ok=True)

        posed_flag = mode == "recon_posed"
        
        # Filter out datasets that don't support reconstruction (e.g., dtu64)
        recon_datas = [d for d in self.datas if d != "dtu64"]

        for data in tqdm(recon_datas, desc=f"Datasets ({mode} eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            # Prepare paths for all scenes
            scene_list = []
            result_paths = []
            fuse_paths = []
            for scene in scenes:
                export_dir = self._export_dir(data, scene, posed=posed_flag)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
                fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
                scene_list.append(scene)
                result_paths.append(result_path)
                fuse_paths.append(fuse_path)

            # Parallel fusion (default 4 workers)
            # DTU uses CUDA operations in fusion, which doesn't work well with ThreadPool
            use_sequential = (data == "dtu")
            parallel_execution(
                scene_list,
                result_paths,
                fuse_paths,
                action=lambda s, rp, fp: dataset.fuse3d(s, rp, fp, mode),
                num_processes=self.num_fusion_workers,
                print_progress=True,
                desc=f"{data} fusion",
                sequential=use_sequential,
            )

            # Sequential evaluation (fast, no need to parallelize)
            for scene, fuse_path in zip(scene_list, fuse_paths):
                # DTU supports CPU-based evaluation
                if data == "dtu" and hasattr(dataset, "eval3d"):
                    result = dataset.eval3d(scene, fuse_path)
                else:
                    result = dataset.eval3d(scene, fuse_path)
                dataset_results[scene] = self._to_float_dict(result)
                print(f"  {mode} | {data} | {scene}: {result}")

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    # -------------------- Helpers -------------------- #

    def _save_gt_meta(self, export_dir: str, scene_data: Dict) -> None:
        """
        Save GT extrinsics/intrinsics/image_files for evaluation.

        This is needed when frames are sampled, so eval_pose and fuse3d can use
        the correct (sampled) GT instead of full dataset GT.

        Args:
            export_dir: Export directory for the scene
            scene_data: Sampled scene data
        """
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.savez_compressed(
            meta_path,
            extrinsics=scene_data.extrinsics,
            intrinsics=scene_data.intrinsics,
            image_files=np.array(scene_data.image_files, dtype=object),
        )

    def _load_gt_meta(self, export_dir: str) -> Dict:
        """
        Load saved GT extrinsics/intrinsics for evaluation.

        Returns:
            Dict with extrinsics and intrinsics, or None if not found
        """
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if os.path.exists(meta_path):
            data = np.load(meta_path)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
            })
        return None

    def _compute_pose_with_gt(self, result_path: str, gt_meta: Dict) -> TDict[str, float]:
        """
        Compute pose metrics using saved GT meta (handles frame sampling).

        Args:
            result_path: Path to npz with predicted extrinsics
            gt_meta: Dict with GT extrinsics from saved meta

        Returns:
            Dict with pose metrics
        """
        from depth_anything_3.bench.dataset import _wait_for_file_ready
        from depth_anything_3.bench.utils import compute_pose
        from depth_anything_3.utils.geometry import as_homogeneous

        _wait_for_file_ready(result_path)
        pred = np.load(result_path)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
        )

    def _load_gt_depth_and_mask(self, data: str, dataset, scene: str, full_gt_data, img_idx: int):
        """Load GT depth (meters) and a boolean valid mask (True = valid)."""
        import cv2
        import imageio

        data = data.lower()

        if data == "eth3d":
            img_path = full_gt_data.image_files[img_idx]
            img_name = os.path.basename(img_path)

            scene_dir = os.path.join(dataset.data_root, scene)
            gt_depth_path = os.path.join(scene_dir, "ground_truth_depth", "dslr_images", img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_candidates = [
                os.path.join(scene_dir, "masks_for_images", "dslr_images", mask_name),
                os.path.join(scene_dir, "ground_truth_masks", "dslr_images", img_name),
            ]
            gt_mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

            if hasattr(full_gt_data, "aux") and hasattr(full_gt_data.aux, "heights"):
                orig_h = int(full_gt_data.aux.heights[img_idx])
                orig_w = int(full_gt_data.aux.widths[img_idx])
            else:
                im = cv2.imread(img_path)
                if im is None:
                    raise FileNotFoundError(f"Failed to read image for size: {img_path}")
                orig_h, orig_w = im.shape[:2]

            gt_depth = np.fromfile(gt_depth_path, dtype=np.float32).reshape(orig_h, orig_w)
            invalid_from_depth = (gt_depth == 0) | (~np.isfinite(gt_depth))

            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED) if gt_mask_path else None

            if gt_mask is None:
                invalid_from_mask = np.zeros_like(invalid_from_depth)
            else:
                invalid_from_mask = gt_mask == 1

            valid = (~invalid_from_depth) & (~invalid_from_mask)
            return gt_depth.astype(np.float32), valid.astype(bool)

        aux = getattr(full_gt_data, "aux", None)
        if aux is None or not hasattr(aux, "gt_depth_files"):
            raise RuntimeError(f"Dataset '{data}' does not expose gt_depth_files for depth eval.")

        depth_path = aux.gt_depth_files[img_idx]

        if data == "7scenes":
            raw = imageio.imread(depth_path).astype(np.float32)
            invalid = raw == 65535
            gt_depth = raw / 1000.0
            gt_depth[invalid] = 0.0
            valid = gt_depth > 0
            return gt_depth.astype(np.float32), valid.astype(bool)

        if data == "hiroom":
            raw = imageio.imread(depth_path).astype(np.float32)
            gt_depth = raw / 65535.0 * 100.0
            valid = gt_depth > 0
            if hasattr(aux, "aliasing_mask_files"):
                alias_path = aux.aliasing_mask_files[img_idx]
                if alias_path and os.path.exists(alias_path):
                    alias = imageio.imread(alias_path)
                    valid &= alias == 0
            return gt_depth.astype(np.float32), valid.astype(bool)

        if data == "scannetpp":
            raw = imageio.imread(depth_path).astype(np.float32)
            gt_depth = raw / 1000.0
            valid = (gt_depth > 0) & np.isfinite(gt_depth)
            return gt_depth.astype(np.float32), valid.astype(bool)

        raw = imageio.imread(depth_path).astype(np.float32)
        gt_depth = raw / 1000.0
        valid = (gt_depth > 0) & np.isfinite(gt_depth)
        return gt_depth.astype(np.float32), valid.astype(bool)

    def _sample_frames(self, scene_data: Dict, scene: str) -> Dict:
        """
        Sample frames if scene has more than max_frames.

        Uses fixed random seed (42) for reproducibility.

        Args:
            scene_data: Scene data dict with image_files, extrinsics, intrinsics, aux
            scene: Scene name (for logging)

        Returns:
            Sampled scene_data if num_frames > max_frames, otherwise original
        """
        if self.max_frames <= 0:
            return scene_data

        num_frames = len(scene_data.image_files)
        if num_frames <= self.max_frames:
            return scene_data

        # Sample with fixed seed for reproducibility
        random.seed(42)
        indices = list(range(num_frames))
        random.shuffle(indices)
        sampled_indices = sorted(indices[:self.max_frames])

        print(f"  [Sampling] {scene}: {num_frames} -> {self.max_frames} frames")

        # Create new scene_data with sampled frames
        sampled = Dict()
        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]

        # Copy aux data, sampling lists if needed
        sampled.aux = Dict()
        for key, val in scene_data.aux.items():
            if isinstance(val, list) and len(val) == num_frames:
                sampled.aux[key] = [val[i] for i in sampled_indices]
            elif isinstance(val, np.ndarray) and len(val) == num_frames:
                sampled.aux[key] = val[sampled_indices]
            else:
                sampled.aux[key] = val

        return sampled

    @property
    def _metric_dir(self) -> str:
        """Directory for storing metric JSON files."""
        return os.path.join(self.work_dir, "metric_results")

    def _export_dir(self, data: str, scene: str, posed: bool) -> str:
        """
        Get export directory path.

        Structure: .../model_results/{data}/{scene}/{posed|unposed}
        """
        suffix = "posed" if posed else "unposed"
        export_dir = os.path.join(self.work_dir, "model_results", data, scene, suffix)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    @staticmethod
    def _to_float_dict(d: TDict[str, float]) -> dict:
        """Convert numpy scalars to plain Python floats for JSON safety."""
        return {k: float(v) for k, v in d.items()}

    @staticmethod
    def _mean_of_dicts(dicts: Iterable[dict]) -> dict:
        """Compute elementwise mean across a list of homogeneous metric dicts."""
        dicts = list(dicts)
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts]).item()) for k in keys}

    @staticmethod
    def _dump_json(path: str, obj: dict, indent: int = 4) -> None:
        """Write JSON with UTF-8 and pretty indentation."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    def _load_metrics(self) -> TDict[str, dict]:
        """Load evaluation metrics from JSON files."""
        metrics = {}
        metric_dir = self._metric_dir

        if not os.path.exists(metric_dir):
            return metrics

        for filename in os.listdir(metric_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metric_dir, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    key = filename[:-5]  # Remove .json extension
                    metrics[key] = data
                except Exception as e:
                    print(f"Warning: Failed to read metrics file: {filename} - {e}")

        return metrics


# -------------------- CLI Entry Point -------------------- #


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from depth_anything_3.cfg import load_config

    # Get default config path (relative to this file)
    _default_config = os.path.join(
        os.path.dirname(__file__), "configs", "eval_bench.yaml"
    )

    # Check for help flag first (we need to handle this before OmegaConf)
    if "--help" in sys.argv or "-h" in sys.argv:
        pass  # Will handle after config loading

    # Set up argv for OmegaConf processing
    argv = sys.argv[1:]

    # Check if user provides custom config
    config_path = _default_config
    if "--config" in argv:
        config_idx = argv.index("--config")
        if config_idx + 1 < len(argv):
            config_path = argv[config_idx + 1]
            # Remove --config and its value
            argv = argv[:config_idx] + argv[config_idx + 2:]

    # Print help if requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
DepthAnything3 Benchmark Evaluation

Usage:
  python -m depth_anything_3.bench.evaluator [OPTIONS] [KEY=VALUE ...]

Configuration:
  --config PATH                      Config YAML file (default: bench/configs/eval_bench.yaml)

Config Overrides (using dotlist notation):
  model.path=VALUE                   Model path or HuggingFace ID
  workspace.work_dir=VALUE           Working directory for outputs
  eval.datasets=[dataset1,dataset2]  Datasets to evaluate (eth3d,7scenes,scannetpp,hiroom,dtu,dtu64)
  eval.modes=[mode1,mode2]           Evaluation modes (pose,recon_unposed,recon_posed)
  eval.scenes=[scene1,scene2]        Specific scenes to evaluate (null=all)
  eval.max_frames=VALUE              Max frames per scene (-1=no limit, default: 100)
  eval.ref_view_strategy=VALUE       Reference view strategy (default: first)
  eval.eval_only=VALUE               Only run evaluation (skip inference) (true/false)
  eval.print_only=VALUE              Only print saved metrics (true/false)
  inference.num_fusion_workers=VALUE Number of parallel workers (default: 4)
  inference.debug=VALUE              Enable debug mode (true/false)

Special Flags:
  --help, -h                         Show this help message

Multi-GPU:
  Use CUDA_VISIBLE_DEVICES to specify GPUs (auto-detected and distributed)

Examples:
  # Use default config
  python -m depth_anything_3.bench.evaluator

  # Override model path
  python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-LARGE

  # Evaluate specific datasets and modes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d,hiroom] \\
      eval.modes=[pose]

  # Use custom config with overrides
  python -m depth_anything_3.bench.evaluator \\
      --config my_config.yaml \\
      model.path=/path/to/model \\
      eval.max_frames=50

  # Multi-GPU inference (auto-distributed)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m depth_anything_3.bench.evaluator

  # Debug specific scenes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d] \\
      eval.scenes=[courtyard] \\
      inference.debug=true

  # Only evaluate (skip inference)
  python -m depth_anything_3.bench.evaluator eval.eval_only=true

  # Only print saved metrics
  python -m depth_anything_3.bench.evaluator eval.print_only=true

          """)
        sys.exit(0)

    # Load config with CLI overrides using OmegaConf dotlist
    # Example: python evaluator.py model.path=/path/to/model eval.datasets=[eth3d,dtu]
    config = load_config(config_path, argv=argv)

    # Extract config values
    work_dir = config.workspace.work_dir
    model_path = config.model.path
    datasets = config.eval.datasets
    modes = config.eval.modes
    ref_view_strategy = config.eval.ref_view_strategy
    scenes = config.eval.scenes
    max_frames = config.eval.max_frames
    eval_only = config.eval.eval_only
    print_only = config.eval.print_only
    debug = config.inference.debug
    num_fusion_workers = config.inference.num_fusion_workers

    # GPU settings: parse from CLI dotlist args (gpu_id=X total_gpus=Y)
    # These are passed by the main process when spawning workers
    gpu_id = 0
    total_gpus = 1
    for arg in argv:
        if arg.startswith("gpu_id="):
            gpu_id = int(arg.split("=")[1])
        elif arg.startswith("total_gpus="):
            total_gpus = int(arg.split("=")[1])

    # Override dataset scenes if specified
    if scenes:
        print(f"[INFO] Running on specific scenes: {scenes}")

    evaluator = Evaluator(
        work_dir=work_dir,
        datas=datasets,
        modes=modes,
        ref_view_strategy=ref_view_strategy,
        scenes=scenes,
        debug=debug,
        num_fusion_workers=num_fusion_workers,
        max_frames=max_frames,
        gpu_id=gpu_id,
        total_gpus=total_gpus,
    )

    if print_only:
        evaluator.print_metrics()
    elif eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
    else:
        # Parse CUDA_VISIBLE_DEVICES to get GPU list
        # If not set, use all available GPUs
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None and cuda_devices.strip():
            gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
        else:
            # CUDA_VISIBLE_DEVICES not set, use all available GPUs
            num_available = torch.cuda.device_count()
            gpu_list = [str(i) for i in range(num_available)] if num_available > 0 else ["0"]

        # Auto multi-GPU: if multiple GPUs and not a worker process
        is_worker = os.environ.get("_DA3_WORKER") == "1"

        if len(gpu_list) > 1 and not is_worker:
            # Launch worker processes
            import subprocess

            num_gpus = len(gpu_list)
            print(f"[INFO] Detected {num_gpus} GPUs: {gpu_list}")
            print(f"[INFO] Launching {num_gpus} workers...")

            # Build base command
            base_cmd = [sys.executable, "-m", "depth_anything_3.bench.evaluator"]
            # Pass config via dotlist instead of CLI args
            if config_path != _default_config:
                base_cmd += ["--config", config_path]
            base_cmd += [f"model.path={model_path}"]
            base_cmd += [f"workspace.work_dir={work_dir}"]
            base_cmd += [f"eval.datasets=[{','.join(datasets)}]"]
            base_cmd += [f"eval.modes=[{','.join(modes)}]"]
            if scenes:
                base_cmd += [f"eval.scenes=[{','.join(scenes)}]"]
            base_cmd += [f"eval.max_frames={max_frames}"]
            base_cmd += [f"eval.ref_view_strategy={ref_view_strategy}"]
            base_cmd += [f"inference.debug={str(debug).lower()}"]
            base_cmd += [f"inference.num_fusion_workers={num_fusion_workers}"]

            # Launch workers
            processes = []
            for idx, gpu_id in enumerate(gpu_list):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                env["_DA3_WORKER"] = "1"  # Mark as worker process

                cmd = base_cmd.copy()
                # GPU-specific worker config
                cmd += [f"gpu_id={idx}", f"total_gpus={num_gpus}"]

                print(f"[INFO] Starting worker {idx} on GPU {gpu_id}")
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)

            # Wait for all workers
            for p in processes:
                p.wait()

            print(f"[INFO] All {num_gpus} workers completed")

            # Run evaluation after all inference is done
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
        else:
            # Single GPU or worker process
            from depth_anything_3.api import DepthAnything3

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            api = DepthAnything3.from_pretrained(model_path)
            api = api.to(device)

            evaluator.infer(api, model_path=model_path)

            # Only run eval if single GPU mode (workers don't eval)
            if not is_worker:
                metrics = evaluator.eval()
                evaluator.print_metrics(metrics)
