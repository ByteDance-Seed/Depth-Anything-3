from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from depth_anything_3.utils.io.input_processor import InputProcessor

from .config import FinetuneConfig


def _split_root(cfg: FinetuneConfig, split: str) -> Path:
    return cfg.data_root / split


def _parse_yolo_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    """Parse YOLO-format label file -> list of (cx, cy, w, h) in [0,1]. Missing
    file or empty content -> []."""
    if not path.is_file():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for ln in path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h = parts[:5]
        boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def _build_valid_index(cfg: FinetuneConfig, split: str) -> list[str]:
    """Return a list of stems where images/<stem>.jpeg and depths/<stem>.npy exist
    and the depth map is not all zeros. If cfg.bbox_loss is on, also require a
    non-empty labels/<stem>.txt."""
    split_dir = _split_root(cfg, split)
    depths_dir = split_dir / "depths"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    assert depths_dir.is_dir(), f"missing depths dir: {depths_dir}"
    assert images_dir.is_dir(), f"missing images dir: {images_dir}"
    if cfg.bbox_loss:
        assert labels_dir.is_dir(), f"missing labels dir (required for bbox_loss): {labels_dir}"

    stems: list[str] = []
    total = 0
    rejected_empty = 0
    rejected_missing_img = 0
    rejected_no_bbox = 0
    for p in sorted(depths_dir.glob("*.npy")):
        total += 1
        stem = p.stem
        if not (images_dir / f"{stem}.jpeg").is_file():
            rejected_missing_img += 1
            continue
        # Memory-map to avoid loading 260KB per file into RAM
        arr = np.load(p, mmap_mode="r")
        if np.asarray(arr).max() <= 0:
            rejected_empty += 1
            continue
        if cfg.bbox_loss:
            if not _parse_yolo_boxes(labels_dir / f"{stem}.txt"):
                rejected_no_bbox += 1
                continue
        stems.append(stem)

    print(
        f"[dataset/{split}] index: kept {len(stems)}/{total} "
        f"(empty={rejected_empty}, missing_img={rejected_missing_img}, "
        f"no_bbox={rejected_no_bbox})"
    )
    return stems


def _index_cache_path(cfg: FinetuneConfig, split: str) -> Path:
    suffix = "_bbox" if cfg.bbox_loss else ""
    return cfg.valid_index_cache / f"valid_{split}{suffix}_index.txt"


def get_or_build_index(cfg: FinetuneConfig, split: str) -> list[str]:
    cache = _index_cache_path(cfg, split)
    if cache.is_file() and not cfg.rebuild_index:
        stems = [ln.strip() for ln in cache.read_text().splitlines() if ln.strip()]
        print(f"[dataset/{split}] loaded index from {cache} ({len(stems)} samples)")
        return stems
    stems = _build_valid_index(cfg, split)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(stems) + "\n")
    print(f"[dataset/{split}] wrote index to {cache}")
    return stems


class DroneDepthDataset(Dataset):
    """Drone-v59 monocular depth dataset.

    Each item yields a dict:
      rgb:      FloatTensor (3, proc_res, proc_res), ImageNet-normalized
      depth_m:  FloatTensor (proc_res, proc_res), meters at proc_res
      valid:    BoolTensor  (proc_res, proc_res), True on usable pixels
      bbox_mask: BoolTensor (proc_res, proc_res), present iff cfg.bbox_loss;
                 True inside the union of YOLO bboxes at proc_res
      focal_px_input: float scalar (focal at orig_res)
      stem:     str
    """

    def __init__(
        self,
        cfg: FinetuneConfig,
        split: str,
        subset: Optional[int] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.split = split
        self.split_dir = _split_root(cfg, split)
        self.images_dir = self.split_dir / "images"
        self.depths_dir = self.split_dir / "depths"
        self.labels_dir = self.split_dir / "labels"

        all_stems = get_or_build_index(cfg, split)
        if subset is not None and subset < len(all_stems):
            rng = random.Random(seed)
            all_stems = rng.sample(all_stems, subset)
        self.stems = all_stems

        # One InputProcessor per worker (it's stateless, cheap to reuse)
        self._input_processor = InputProcessor()

    def __len__(self) -> int:
        return len(self.stems)

    def _load_rgb(self, stem: str) -> torch.Tensor:
        img_path = str(self.images_dir / f"{stem}.jpeg")
        # InputProcessor returns (N, 3, H, W) — collapse N=1 here
        tensor, _, _ = self._input_processor(
            [img_path],
            process_res=self.cfg.proc_res,
            process_res_method="upper_bound_resize",
            num_workers=1,
            sequential=True,
            desc=None,
        )
        # Shape is (N=1, 3, H, W); squeeze N only
        return tensor.squeeze(0).contiguous()

    def _load_depth_and_valid(self, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        depth_path = self.depths_dir / f"{stem}.npy"
        depth = np.load(depth_path).astype(np.float32)  # (orig, orig)
        orig_valid = np.isfinite(depth) & (depth > 0)

        tgt = self.cfg.proc_res
        # Linear resize for depth values; nearest resize for the mask to avoid
        # fake-valid pixels appearing through interpolation at mask boundaries.
        depth_resized = cv2.resize(depth, (tgt, tgt), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(
            orig_valid.astype(np.uint8), (tgt, tgt), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        mask_resized &= np.isfinite(depth_resized) & (depth_resized > 0)

        depth_t = torch.from_numpy(depth_resized).float()
        valid_t = torch.from_numpy(mask_resized)
        return depth_t, valid_t

    def _load_bbox_mask(self, stem: str) -> torch.Tensor:
        """Build a (proc_res, proc_res) bool mask from the union of YOLO bboxes,
        optionally padded by cfg.bbox_pad (fraction of box size)."""
        tgt = self.cfg.proc_res
        mask = np.zeros((tgt, tgt), dtype=bool)
        boxes = _parse_yolo_boxes(self.labels_dir / f"{stem}.txt")
        pad = float(self.cfg.bbox_pad)
        for cx, cy, w, h in boxes:
            w_p = w * (1.0 + 2.0 * pad)
            h_p = h * (1.0 + 2.0 * pad)
            x0 = int(np.floor(max(0.0, (cx - w_p / 2.0) * tgt)))
            y0 = int(np.floor(max(0.0, (cy - h_p / 2.0) * tgt)))
            x1 = int(np.ceil(min(1.0, (cx + w_p / 2.0) * tgt) * 1.0))
            y1 = int(np.ceil(min(1.0, (cy + h_p / 2.0) * tgt) * 1.0))
            x1 = min(x1, tgt)
            y1 = min(y1, tgt)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
        return torch.from_numpy(mask)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        rgb = self._load_rgb(stem)
        depth_m, valid = self._load_depth_and_valid(stem)
        item = {
            "rgb": rgb,
            "depth_m": depth_m,
            "valid": valid,
            "focal_px_input": float(self.cfg.dataset_focal_at_orig),
            "stem": stem,
        }
        if self.cfg.bbox_loss:
            item["bbox_mask"] = self._load_bbox_mask(stem)
        return item
