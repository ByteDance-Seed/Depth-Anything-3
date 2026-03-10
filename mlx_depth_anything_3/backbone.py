# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""DinoV2 Vision Transformer backbone for MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_depth_anything_3.layers import (
    Attention,
    Block,
    Mlp,
    ModuleList,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFN,
)
from mlx_depth_anything_3.reference_view import (
    THRESH_FOR_REF_SELECTION,
    reorder_by_reference,
    restore_original_order,
    select_reference_view,
)


class DinoVisionTransformer(nn.Module):
    """DinoV2 Vision Transformer with alternating local/global attention."""

    PATCH_SIZE = 14

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        init_values: float = 1.0,
        ffn_layer: str = "mlp",
        num_register_tokens: int = 0,
        interpolate_offset: float = 0.1,
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        rope_freq: float = 100.0,
        cat_token: bool = True,
    ):
        super().__init__()
        self.patch_start_idx = 1
        self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = mx.zeros((1, 1, embed_dim))
        if alt_start != -1:
            self.camera_token = mx.random.normal((1, 2, embed_dim)) * 0.02
        self.pos_embed = mx.zeros((1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            mx.zeros((1, num_register_tokens, embed_dim))
            if num_register_tokens > 0 else None
        )

        # RoPE
        self.rope = None
        self.position_getter = None
        if rope_start != -1:
            if rope_freq > 0:
                self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
                self.position_getter = PositionGetter()

        # Build blocks
        ffn_cls = "swiglu" if ffn_layer in ("swiglufused", "swiglu") else "mlp"
        self.blocks = ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                ffn_layer=ffn_cls,
                qk_norm=(i >= qknorm_start) if qknorm_start != -1 else False,
                rope=self.rope if (i >= rope_start and rope_start != -1) else None,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: mx.array, w: int, h: int) -> mx.array:
        """Interpolate position embeddings for arbitrary resolution."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.astype(mx.float32)
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))

        # Reshape to spatial, interpolate, flatten
        # (1, M, M, dim) -> (1, h0, w0, dim)
        spatial = mx.reshape(patch_pos_embed, (1, M, M, dim))

        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            # Compute target size
            th = int(round(M * sy))
            tw = int(round(M * sx))
        else:
            th, tw = h0, w0

        from mlx_depth_anything_3.layers import bilinear_interpolate
        spatial = bilinear_interpolate(spatial, th, tw, align_corners=False)
        # Crop to exact size if needed
        spatial = spatial[:, :h0, :w0, :]
        patch_pos_embed = mx.reshape(spatial, (1, -1, dim))
        return mx.concatenate([class_pos_embed, patch_pos_embed], axis=1).astype(x.dtype)

    def prepare_tokens(self, x: mx.array) -> mx.array:
        """Prepare tokens from multi-view images.

        Args:
            x: (B, S, H, W, C) in NHWC.

        Returns:
            (B, S, N+1, D) token tensor with cls_token prepended.
        """
        B, S, H, W, C = x.shape
        # Flatten batch and views
        x_flat = mx.reshape(x, (B * S, H, W, C))
        x_flat = self.patch_embed(x_flat)  # (B*S, num_patches, D)

        # Prepend cls token
        cls = mx.broadcast_to(self.cls_token, (B * S, 1, self.embed_dim))
        x_flat = mx.concatenate([cls, x_flat], axis=1)

        # Add position embeddings
        x_flat = x_flat + self.interpolate_pos_encoding(x_flat, W, H)

        # Add register tokens if present
        if self.register_tokens is not None:
            reg = mx.broadcast_to(self.register_tokens, (B * S, self.num_register_tokens, self.embed_dim))
            x_flat = mx.concatenate([x_flat[:, :1], reg, x_flat[:, 1:]], axis=1)

        # Reshape to (B, S, N, D)
        N = x_flat.shape[1]
        D = x_flat.shape[2]
        x_out = mx.reshape(x_flat, (B, S, N, D))
        return x_out

    def _prepare_rope(self, B: int, S: int, H: int, W: int) -> tuple:
        """Prepare RoPE position tensors."""
        if self.rope is None or self.position_getter is None:
            return None, None
        ph = H // self.patch_size
        pw = W // self.patch_size
        pos = self.position_getter(B * S, ph, pw)  # (B*S, ph*pw, 2)
        pos = mx.reshape(pos, (B, S, ph * pw, 2))
        pos_nodiff = mx.zeros_like(pos)
        if self.patch_start_idx > 0:
            pos = pos + 1
            special = mx.zeros((B, S, self.patch_start_idx, 2))
            pos = mx.concatenate([special, pos], axis=2)
            pos_nodiff = pos_nodiff + 1
            pos_nodiff = mx.concatenate([mx.zeros((B, S, self.patch_start_idx, 2)), pos_nodiff], axis=2)
        return pos, pos_nodiff

    def process_attention(self, x: mx.array, block: Block, attn_type: str,
                          pos: mx.array | None = None,
                          attn_mask: mx.array | None = None) -> mx.array:
        """Run one block with local or global attention.

        Args:
            x: (B, S, N, C) token tensor.
            block: transformer block.
            attn_type: "local" or "global".
            pos: optional RoPE positions.
        """
        b, s, n, c = x.shape
        if attn_type == "local":
            x_flat = mx.reshape(x, (b * s, n, c))
            if pos is not None:
                pos_flat = mx.reshape(pos, (b * s, pos.shape[2], pos.shape[3]))
            else:
                pos_flat = None
            x_flat = block(x_flat, pos=pos_flat, attn_mask=attn_mask)
            return mx.reshape(x_flat, (b, s, n, c))
        else:  # global
            x_flat = mx.reshape(x, (b, s * n, c))
            if pos is not None:
                pos_flat = mx.reshape(pos, (b, s * pos.shape[2], pos.shape[3]))
            else:
                pos_flat = None
            x_flat = block(x_flat, pos=pos_flat, attn_mask=attn_mask)
            return mx.reshape(x_flat, (b, s, n, c))

    def get_intermediate_layers(
        self,
        x: mx.array,
        out_layers: list[int],
        cam_token: mx.array | None = None,
        export_feat_layers: list[int] | None = None,
        ref_view_strategy: str = "saddle_balanced",
    ) -> tuple:
        """Extract intermediate features from the backbone.

        Args:
            x: (B, S, H, W, C) multi-view images in NHWC.
            out_layers: list of layer indices to extract.
            cam_token: (B, S, D) optional camera conditioning tokens.
            export_feat_layers: additional layer indices for auxiliary features.
            ref_view_strategy: reference view selection strategy.

        Returns:
            (features, aux_features) tuple.
        """
        if export_feat_layers is None:
            export_feat_layers = []
        B, S, H, W, C = x.shape

        x_tok = self.prepare_tokens(x)  # (B, S, N, D)
        pos, pos_nodiff = self._prepare_rope(B, S, H, W)

        output = []
        aux_output = []
        blocks_to_take = set(out_layers)
        local_x = x_tok
        b_idx = None

        for i, blk in enumerate(self.blocks):
            # RoPE positions
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # Reference view selection (before alt_start).
            # Skipped when external camera tokens are provided, because the
            # camera encoder already supplies per-view conditioning so
            # feature-based reordering is unnecessary.
            if (self.alt_start != -1 and i == self.alt_start - 1
                    and x_tok.shape[1] >= THRESH_FOR_REF_SELECTION
                    and cam_token is None):
                b_idx = select_reference_view(x_tok, strategy=ref_view_strategy)
                x_tok = reorder_by_reference(x_tok, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            # Camera token injection
            if self.alt_start != -1 and i == self.alt_start:
                if cam_token is not None:
                    ct = cam_token
                else:
                    ref_tok = mx.broadcast_to(self.camera_token[:, :1], (B, 1, self.embed_dim))
                    src_tok = mx.broadcast_to(self.camera_token[:, 1:], (B, S - 1, self.embed_dim))
                    ct = mx.concatenate([ref_tok, src_tok], axis=1)
                # Replace cls token position with camera token
                x_tok_list = []
                for bidx in range(B):
                    view_tokens = []
                    for sidx in range(S):
                        tokens = x_tok[bidx, sidx]
                        tokens_new = mx.concatenate(
                            [mx.expand_dims(ct[bidx, sidx], 0), tokens[1:]], axis=0
                        )
                        view_tokens.append(tokens_new)
                    x_tok_list.append(mx.stack(view_tokens))
                x_tok = mx.stack(x_tok_list)

            # Local or global attention
            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x_tok = self.process_attention(x_tok, blk, "global", pos=g_pos)
            else:
                x_tok = self.process_attention(x_tok, blk, "local", pos=l_pos)
                local_x = x_tok

            # Collect output features
            if i in blocks_to_take:
                if self.cat_token:
                    out_x = mx.concatenate([local_x, x_tok], axis=-1)
                else:
                    out_x = x_tok
                # Restore original order if reordering was applied
                if (x_tok.shape[1] >= THRESH_FOR_REF_SELECTION
                        and self.alt_start != -1 and b_idx is not None):
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))

            if i in export_feat_layers:
                aux_output.append(x_tok)

        # Apply final norm
        camera_tokens = [out[0] for out in output]
        if output[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in output]
        elif output[0][1].shape[-1] == self.embed_dim * 2:
            outputs = [
                mx.concatenate(
                    [out[1][..., :self.embed_dim], self.norm(out[1][..., self.embed_dim:])],
                    axis=-1,
                )
                for out in output
            ]
        else:
            raise ValueError(f"Invalid output shape: {output[0][1].shape}")

        aux_output = [self.norm(out) for out in aux_output]

        # Drop cls + register tokens
        skip = 1 + self.num_register_tokens
        outputs = [out[..., skip:, :] for out in outputs]
        aux_output = [out[..., skip:, :] for out in aux_output]

        return list(zip(outputs, camera_tokens)), aux_output


class DinoV2(nn.Module):
    """DinoV2 wrapper matching the PyTorch API."""

    def __init__(
        self,
        name: str,
        out_layers: list[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
    ):
        super().__init__()
        self.name = name
        self.out_layers = out_layers
        encoder_map = {
            "vits": dict(embed_dim=384, depth=12, num_heads=6),
            "vitb": dict(embed_dim=768, depth=12, num_heads=12),
            "vitl": dict(embed_dim=1024, depth=24, num_heads=16),
            "vitg": dict(embed_dim=1536, depth=40, num_heads=24),
        }
        cfg = encoder_map[name]
        ffn = "swiglu" if name == "vitg" else "mlp"
        self.pretrained = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            **cfg,
        )

    def __call__(self, x: mx.array, **kwargs) -> tuple:
        return self.pretrained.get_intermediate_layers(x, self.out_layers, **kwargs)
