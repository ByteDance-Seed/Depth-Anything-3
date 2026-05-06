from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class FinetuneConfig:
    # data
    data_root: Path = Path("/home/derek_austin/Depth-Anything-3/drone_v59_depth")
    train_split: str = "train"
    val_split: Optional[str] = "val"
    orig_res: int = 256
    proc_res: int = 504
    canonical_focal: float = 300.0
    dataset_focal_at_orig: float = 300.0
    subset_size: Optional[int] = 1000
    valid_index_cache: Path = Path("./finetune_cache")
    rebuild_index: bool = False

    # model
    pretrained_id: str = "depth-anything/DA3METRIC-LARGE"
    freeze_sky_head: bool = True

    # loss
    loss: str = "log_l1"
    silog_lambda: float = 0.15
    bbox_loss: bool = False
    bbox_pad: float = 0.0

    # optim
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"

    # logging / checkpoint
    log_every: int = 20
    val_every_steps: int = 500
    val_max_batches: int = 20
    ckpt_dir: Path = Path("./checkpoints/drone_v59")
    run_name: str = "smoke"
    seed: int = 0

    # wandb
    wandb: bool = True
    wandb_project: str = "da3-drone-finetune"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    log_images_every: int = 500  # 0 to disable image panels

    # debug
    dry_run: bool = False


def _opt_int(v):
    return None if str(v).lower() in ("none", "null", "") else int(v)


def _opt_str(v):
    return None if str(v).lower() in ("none", "null", "") else str(v)


_TYPE_MAP = {
    "data_root": Path,
    "train_split": str,
    "val_split": _opt_str,
    "orig_res": int,
    "proc_res": int,
    "canonical_focal": float,
    "dataset_focal_at_orig": float,
    "subset_size": _opt_int,
    "valid_index_cache": Path,
    "rebuild_index": "bool",
    "pretrained_id": str,
    "freeze_sky_head": "bool",
    "loss": str,
    "silog_lambda": float,
    "bbox_loss": "bool",
    "bbox_pad": float,
    "batch_size": int,
    "num_workers": int,
    "epochs": int,
    "lr": float,
    "weight_decay": float,
    "grad_clip": float,
    "amp_dtype": str,
    "log_every": int,
    "val_every_steps": int,
    "val_max_batches": int,
    "ckpt_dir": Path,
    "run_name": str,
    "seed": int,
    "wandb": "bool",
    "wandb_project": str,
    "wandb_entity": _opt_str,
    "wandb_mode": str,
    "log_images_every": int,
    "dry_run": "bool",
}


def parse_cli(argv: Optional[list[str]] = None) -> FinetuneConfig:
    parser = argparse.ArgumentParser(description="Fine-tune DA3METRIC-LARGE on drone_v59_depth")
    defaults = FinetuneConfig()
    for f in fields(FinetuneConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        caster = _TYPE_MAP[f.name]
        if caster == "bool":
            # bool fields get both --flag and --no-flag to toggle from their default
            grp = parser.add_mutually_exclusive_group()
            grp.add_argument(flag, dest=f.name, action="store_true", default=default)
            grp.add_argument(
                "--no-" + f.name.replace("_", "-"),
                dest=f.name, action="store_false",
            )
        else:
            parser.add_argument(flag, dest=f.name, default=default, type=caster)
    ns = parser.parse_args(argv)
    return FinetuneConfig(**vars(ns))
