from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import FinetuneConfig, parse_cli
from .dataset import DroneDepthDataset
from .losses import depth_metrics, log_l1, silog
from .model_utils import (
    bypass_sky_postproc,
    freeze_and_report,
    load_model,
    trainable_params,
)
from .wandb_logging import WandbLogger


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _canonical_over_processed(cfg: FinetuneConfig) -> float:
    focal_processed = cfg.dataset_focal_at_orig * cfg.proc_res / cfg.orig_res
    return cfg.canonical_focal / focal_processed


def _squeeze_depth(depth_out: torch.Tensor) -> torch.Tensor:
    """Normalize (B,1,H,W) or (B,1,1,H,W) -> (B,H,W)."""
    while depth_out.dim() > 3:
        if depth_out.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected view dim on depth output: shape={tuple(depth_out.shape)}"
            )
        depth_out = depth_out.squeeze(1)
    return depth_out


def compute_loss(cfg: FinetuneConfig, pred_raw, gt_canonical, valid, bbox_mask=None):
    if cfg.loss == "log_l1":
        return log_l1(pred_raw, gt_canonical, valid, bbox_mask=bbox_mask)
    if cfg.loss == "silog":
        return silog(pred_raw, gt_canonical, valid, bbox_mask=bbox_mask, lam=cfg.silog_lambda)
    raise ValueError(f"unknown loss: {cfg.loss}")


@torch.no_grad()
def run_validation(
    net,
    val_loader,
    cfg,
    device,
    canonical_over_processed,
    step: int,
    logger: WandbLogger | None = None,
) -> dict[str, float]:
    net.eval()
    ratios_absrel, ratios_delta1 = [], []
    losses = []
    focal_processed = cfg.dataset_focal_at_orig * cfg.proc_res / cfg.orig_res
    amp = _amp_dtype(cfg.amp_dtype)
    logged_sample = False
    n_val_batches = min(cfg.val_max_batches, len(val_loader))
    val_pbar = tqdm(val_loader, total=n_val_batches, desc=f"val @ step {step}", leave=False)
    for i, batch in enumerate(val_pbar):
        if i >= cfg.val_max_batches:
            break
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt_m = batch["depth_m"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        bbox_mask = None
        if cfg.bbox_loss:
            bbox_mask = batch["bbox_mask"].to(device, non_blocking=True)
        gt_canonical = gt_m * canonical_over_processed

        x = rgb.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=amp):
            out = net(x)
            pred_raw = _squeeze_depth(out["depth"]).float()
        loss = compute_loss(cfg, pred_raw, gt_canonical, valid, bbox_mask=bbox_mask)
        pred_m = pred_raw * focal_processed / cfg.canonical_focal
        m = depth_metrics(pred_m, gt_m, valid)
        ratios_absrel.append(m["absrel"])
        ratios_delta1.append(m["delta1"])
        losses.append(loss.item())

        if logger is not None and not logged_sample and cfg.log_images_every > 0:
            logger.log_sample_triplet(
                rgb[0], gt_m[0], pred_m[0], valid[0], step=step, tag="val"
            )
            logged_sample = True

        val_pbar.set_postfix(loss=f"{loss.item():.4f}", absrel=f"{m['absrel']:.3f}")
    val_pbar.close()

    finite = [x for x in ratios_absrel if math.isfinite(x)]
    absrel = sum(finite) / len(finite) if finite else float("nan")
    finite_d = [x for x in ratios_delta1 if math.isfinite(x)]
    delta1 = sum(finite_d) / len(finite_d) if finite_d else float("nan")
    val_loss = sum(losses) / max(len(losses), 1)
    print(
        f"[val @ step {step}] loss={val_loss:.4f}  absrel={absrel:.4f}  delta1={delta1:.4f}"
    )
    if logger is not None:
        logger.log(
            {"val/loss": val_loss, "val/absrel": absrel, "val/delta1": delta1}, step=step,
        )
    net.train()
    return {"loss": val_loss, "absrel": absrel, "delta1": delta1}


def save_checkpoint(cfg: FinetuneConfig, net, epoch: int, step: int) -> Path:
    out_dir = cfg.ckpt_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
    payload = {
        "model": net.state_dict(),
        "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "epoch": epoch,
        "step": step,
    }
    torch.save(payload, ckpt_path)
    print(f"[ckpt] saved {ckpt_path}")
    return ckpt_path


def main() -> None:
    cfg = parse_cli()
    _seed_everything(cfg.seed)
    print("[config]", json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        indent=2,
    ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available; training on CPU will be extremely slow.")

    model = load_model(cfg)
    freeze_and_report(model, cfg)
    bypass_sky_postproc(model)
    net = model.model
    net.to(device)
    net.train()

    train_ds = DroneDepthDataset(cfg, cfg.train_split, subset=cfg.subset_size, seed=cfg.seed)
    val_ds = None
    if cfg.val_split is not None and (cfg.data_root / cfg.val_split).is_dir():
        val_ds = DroneDepthDataset(cfg, cfg.val_split)
    else:
        print(f"[dataset] val split not available — skipping validation")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=min(cfg.num_workers, 2), pin_memory=True,
        )

    canonical_over_processed = _canonical_over_processed(cfg)
    print(f"[conv] canonical_over_processed = {canonical_over_processed:.6f}")

    opt = torch.optim.AdamW(
        trainable_params(model), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    amp = _amp_dtype(cfg.amp_dtype)

    logger = WandbLogger(cfg)
    focal_processed = cfg.dataset_focal_at_orig * cfg.proc_res / cfg.orig_res

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{cfg.epochs - 1}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt_m = batch["depth_m"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            bbox_mask = None
            if cfg.bbox_loss:
                bbox_mask = batch["bbox_mask"].to(device, non_blocking=True)
            gt_canonical = gt_m * canonical_over_processed

            x = rgb.unsqueeze(1)  # (B, 1, 3, H, W) for DA3's view dim
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
                out = net(x)
                depth_out = out["depth"]
                if step == 0:
                    print(f"[shape] out['depth']={tuple(depth_out.shape)} rgb={tuple(rgb.shape)}")
                pred_raw = _squeeze_depth(depth_out).float()

            loss = compute_loss(cfg, pred_raw, gt_canonical, valid, bbox_mask=bbox_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params(model), cfg.grad_clip,
            )
            opt.step()

            pbar.set_postfix(
                step=step,
                loss=f"{loss.item():.4f}",
                gnorm=f"{float(grad_norm):.2f}",
            )
            if step % cfg.log_every == 0:
                logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": float(grad_norm),
                        "train/valid_px": int(valid.sum().item()),
                        "train/lr": opt.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=step,
                )
            if (
                logger.enabled
                and cfg.log_images_every > 0
                and step > 0
                and step % cfg.log_images_every == 0
            ):
                with torch.no_grad():
                    pred_m_sample = pred_raw[0].float() * focal_processed / cfg.canonical_focal
                logger.log_sample_triplet(
                    rgb[0], gt_m[0], pred_m_sample, valid[0], step=step, tag="train",
                )
            if (
                val_loader is not None
                and step > 0
                and step % cfg.val_every_steps == 0
            ):
                run_validation(
                    net, val_loader, cfg, device, canonical_over_processed, step,
                    logger=logger,
                )

            step += 1
            if cfg.dry_run and step >= 2:
                pbar.close()
                print("[dry-run] stopping after 2 steps")
                save_checkpoint(cfg, net, epoch=epoch, step=step)
                logger.finish()
                return

        pbar.close()
        save_checkpoint(cfg, net, epoch=epoch, step=step)
        if val_loader is not None:
            run_validation(
                net, val_loader, cfg, device, canonical_over_processed, step,
                logger=logger,
            )

    print(f"[done] total steps={step} elapsed={time.time() - t0:.1f}s")
    logger.finish()


if __name__ == "__main__":
    main()
