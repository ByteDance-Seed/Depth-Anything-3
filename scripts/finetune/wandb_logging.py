from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import FinetuneConfig


class WandbLogger:
    """Thin wrapper that is a no-op when wandb is disabled or fails to init."""

    def __init__(self, cfg: FinetuneConfig):
        self.enabled = False
        self._wandb = None
        if not cfg.wandb:
            return
        try:
            import wandb
        except Exception as e:  # pragma: no cover
            print(f"[wandb] import failed ({e}); logging disabled")
            return
        cfg_dict = {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()
        }
        try:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.run_name,
                mode=cfg.wandb_mode,
                config=cfg_dict,
            )
        except Exception as e:
            print(f"[wandb] init failed ({e}); logging disabled")
            return
        self._wandb = wandb
        self.enabled = True

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._wandb.log(data, step=step)

    def log_sample_triplet(
        self,
        rgb: torch.Tensor,         # (3, H, W), ImageNet-normalized
        gt_m: torch.Tensor,        # (H, W), meters
        pred_m: torch.Tensor,      # (H, W), meters
        valid: torch.Tensor,       # (H, W), bool
        step: int,
        tag: str = "train",
    ) -> None:
        if not self.enabled:
            return
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(3, 1, 1)
        rgb_vis = (rgb.float() * std + mean).clamp(0, 1)
        rgb_np = (rgb_vis.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Normalize depth maps to [0, 1] using GT-valid range for both, so colors are comparable
        gt_np = gt_m.float().detach().cpu().numpy()
        pred_np = pred_m.float().detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        if valid_np.any():
            lo, hi = np.percentile(gt_np[valid_np], 2), np.percentile(gt_np[valid_np], 98)
        else:
            lo, hi = float(gt_np.min()), float(gt_np.max() + 1e-6)
        rng = max(hi - lo, 1e-6)

        def _viz(arr: np.ndarray) -> np.ndarray:
            a = np.clip((arr - lo) / rng, 0, 1)
            return (a * 255).astype(np.uint8)

        self._wandb.log(
            {
                f"{tag}/rgb": self._wandb.Image(rgb_np, caption="rgb"),
                f"{tag}/gt_depth_m": self._wandb.Image(_viz(gt_np), caption=f"gt [{lo:.2f},{hi:.2f}]m"),
                f"{tag}/pred_depth_m": self._wandb.Image(_viz(pred_np), caption="pred"),
                f"{tag}/valid_mask": self._wandb.Image((valid_np * 255).astype(np.uint8)),
            },
            step=step,
        )

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
