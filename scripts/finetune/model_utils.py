from __future__ import annotations

import types

import torch

from depth_anything_3.api import DepthAnything3

from .config import FinetuneConfig


def load_model(cfg: FinetuneConfig) -> DepthAnything3:
    return DepthAnything3.from_pretrained(cfg.pretrained_id)


def _has_sky_head(net) -> bool:
    return any("sky" in n for n, _ in net.head.named_parameters())


def freeze_and_report(model: DepthAnything3, cfg: FinetuneConfig) -> DepthAnything3:
    net = model.model

    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.head.parameters():
        p.requires_grad_(True)

    if cfg.freeze_sky_head and _has_sky_head(net):
        for name, p in net.head.named_parameters():
            if "sky" in name:
                p.requires_grad_(False)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    backbone_trainable = sum(
        p.numel() for p in net.backbone.parameters() if p.requires_grad
    )
    print(
        f"[model] trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M "
        f"(backbone trainable={backbone_trainable})"
    )
    assert backbone_trainable == 0, "backbone must be frozen"
    return model


def bypass_sky_postproc(model: DepthAnything3) -> None:
    """Skip _process_mono_sky_estimation while net.training is True.

    Rewires the bound method so the post-processing becomes a no-op during
    training (gradients flow through sky pixels normally and we mask them via
    valid-mask). Eval-time behavior is preserved.
    """
    net = model.model
    orig = net._process_mono_sky_estimation

    def passthrough(self, output):
        if self.training:
            return output
        return orig(output)

    net._process_mono_sky_estimation = types.MethodType(passthrough, net)


def trainable_params(model: DepthAnything3) -> list[torch.nn.Parameter]:
    return [p for p in model.model.parameters() if p.requires_grad]
