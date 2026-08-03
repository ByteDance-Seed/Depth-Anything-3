from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def _zero_loss(ref: torch.Tensor) -> torch.Tensor:
    return ref.sum() * 0.0


def log_l1(
    pred_raw: torch.Tensor,
    gt_canonical: torch.Tensor,
    valid: torch.Tensor,
    bbox_mask: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    mask = valid & (pred_raw > eps) & (gt_canonical > eps)
    if bbox_mask is not None:
        mask = mask & bbox_mask
    if mask.sum() == 0:
        return _zero_loss(pred_raw)
    p = torch.log(pred_raw[mask].clamp_min(eps))
    g = torch.log(gt_canonical[mask].clamp_min(eps))
    return F.l1_loss(p, g)


def silog(
    pred_raw: torch.Tensor,
    gt_canonical: torch.Tensor,
    valid: torch.Tensor,
    bbox_mask: torch.Tensor | None = None,
    lam: float = 0.15,
    eps: float = EPS,
) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al.)."""
    mask = valid & (pred_raw > eps) & (gt_canonical > eps)
    if bbox_mask is not None:
        mask = mask & bbox_mask
    if mask.sum() == 0:
        return _zero_loss(pred_raw)
    d = torch.log(pred_raw[mask].clamp_min(eps)) - torch.log(gt_canonical[mask].clamp_min(eps))
    n = d.numel()
    return (d.pow(2).sum() / n) - lam * (d.sum() / n).pow(2)


@torch.no_grad()
def depth_metrics(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    valid: torch.Tensor,
    eps: float = EPS,
) -> dict[str, float]:
    mask = valid & (pred_m > eps) & (gt_m > eps)
    if mask.sum() == 0:
        return {"absrel": float("nan"), "delta1": float("nan"), "n_valid": 0}
    p = pred_m[mask]
    g = gt_m[mask]
    absrel = torch.mean(torch.abs(p - g) / g).item()
    ratio = torch.maximum(p / g, g / p)
    delta1 = torch.mean((ratio < 1.25).float()).item()
    return {"absrel": absrel, "delta1": delta1, "n_valid": int(mask.sum().item())}
