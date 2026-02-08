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
Unified Inference Service
Provides unified interface for local and remote inference
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import requests
import typer

from ..api import DepthAnything3


class InferenceService:
    """Unified inference service class"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):
        """Load model"""
        if self.model is None:
            typer.echo(f"Loading model from {self.model_dir}...")
            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
        return self.model

    @staticmethod
    def _normalize_export_feat_layers(export_feat_layers: Optional[List[int]]) -> List[int]:
        """Normalize optional export feature layer list."""
        return list(export_feat_layers or [])

    @staticmethod
    def _build_common_inference_params(
        export_format: str,
        process_res: int,
        process_res_method: str,
        export_feat_layers: List[int],
        align_to_input_ext_scale: bool,
        use_ray_pose: bool,
        ref_view_strategy: str,
        conf_thresh_percentile: float,
        num_max_points: int,
        show_cameras: bool,
        feat_vis_fps: int,
    ) -> Dict[str, Any]:
        """Build inference options shared between local and backend modes."""
        return {
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
        }

    @staticmethod
    def _attach_pose_data(
        payload: Dict[str, Any],
        extrinsics: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        json_compatible: bool,
    ) -> None:
        """Attach pose data to payload (encoded for JSON when needed)."""
        if extrinsics is not None:
            payload["extrinsics"] = (
                [ext.astype(np.float64).tolist() for ext in extrinsics]
                if json_compatible
                else extrinsics
            )
        if intrinsics is not None:
            payload["intrinsics"] = (
                [intr.astype(np.float64).tolist() for intr in intrinsics]
                if json_compatible
                else intrinsics
            )

    def run_local_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Any:
        """Run local inference"""
        export_feat_layers = self._normalize_export_feat_layers(export_feat_layers)

        model = self.load_model()
        common_params = self._build_common_inference_params(
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
        inference_kwargs = {
            "image": image_paths,
            "export_dir": export_dir,
            **common_params,
        }
        self._attach_pose_data(
            inference_kwargs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            json_compatible=False,
        )

        # Run inference
        typer.echo(f"Running inference on {len(image_paths)} images...")
        prediction = model.inference(**inference_kwargs)

        typer.echo(f"Results saved to {export_dir}")
        typer.echo(f"Export format: {export_format}")

        return prediction

    def run_backend_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        backend_url: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Dict[str, Any]:
        """Run backend inference"""
        export_feat_layers = self._normalize_export_feat_layers(export_feat_layers)

        # Check backend status
        if not self._check_backend_status(backend_url):
            raise typer.BadParameter(f"Backend service is not running at {backend_url}")

        common_params = self._build_common_inference_params(
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
        payload = {
            "image_paths": image_paths,
            "export_dir": export_dir,
            **common_params,
        }
        self._attach_pose_data(
            payload,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            json_compatible=True,
        )

        # Submit task
        typer.echo("Submitting inference task to backend...")
        try:
            response = requests.post(f"{backend_url}/inference", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["success"]:
                task_id = result["task_id"]
                typer.echo("Task submitted successfully!")
                typer.echo(f"Task ID: {task_id}")
                typer.echo(f"Results will be saved to: {export_dir}")
                typer.echo(f"Check backend logs for progress updates with task ID: {task_id}")
                return result
            else:
                raise typer.BadParameter(
                    f"Backend inference submission failed: {result['message']}"
                )
        except requests.exceptions.RequestException as e:
            raise typer.BadParameter(f"Backend inference submission failed: {e}") from e

    def _check_backend_status(self, backend_url: str) -> bool:
        """Check backend status"""
        try:
            response = requests.get(f"{backend_url}/status", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


def run_inference(
    image_paths: List[str],
    export_dir: str,
    model_dir: str,
    device: str = "cuda",
    backend_url: Optional[str] = None,
    export_format: str = "mini_npz-glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    export_feat_layers: List[int] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    align_to_input_ext_scale: bool = True,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    feat_vis_fps: int = 15,
) -> Union[Any, Dict[str, Any]]:
    """Unified inference interface"""

    service = InferenceService(model_dir, device)
    common_kwargs = {
        "export_format": export_format,
        "process_res": process_res,
        "process_res_method": process_res_method,
        "export_feat_layers": export_feat_layers,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "align_to_input_ext_scale": align_to_input_ext_scale,
        "use_ray_pose": use_ray_pose,
        "ref_view_strategy": ref_view_strategy,
        "conf_thresh_percentile": conf_thresh_percentile,
        "num_max_points": num_max_points,
        "show_cameras": show_cameras,
        "feat_vis_fps": feat_vis_fps,
    }

    if backend_url:
        return service.run_backend_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            backend_url=backend_url,
            **common_kwargs,
        )

    return service.run_local_inference(
        image_paths=image_paths,
        export_dir=export_dir,
        **common_kwargs,
    )
