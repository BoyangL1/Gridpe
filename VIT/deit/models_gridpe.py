"""
This code is modified based on the repository:
https://github.com/naver-ai/rope-vit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GridAttn.gridAttn import GridPEAttention


class GridPE_Layer_scale_init_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = GridPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, positions):
        attn_out, attn_distance, attn_entropy = self.attn(self.norm1(x), positions)
        x = x + self.drop_path(self.gamma_1 * attn_out)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn_distance, attn_entropy


class gridpe_vit_models(vit_models):
    def __init__(self, use_ape=False, **kwargs):
        super().__init__(**kwargs)

        embed_dim = kwargs["embed_dim"] if "embed_dim" in kwargs else 864

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        # calculate coordinates of each patch
        patch_height, patch_width = self.patch_embed.patch_size
        num_patches_h, num_patches_w = H // patch_height, W // patch_width
        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(num_patches_h),
                    torch.arange(num_patches_w),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .float()
        )
        self.positions = (
            grid.to(x.device).unsqueeze(0).expand(B, -1, -1)
        )  # (B, num_patches, dimension)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1,
                    (img_size[1] // patch_size[1]),
                    (img_size[0] // patch_size[0]),
                    self.embed_dim,
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed,
                    size=(H // patch_size[1], W // patch_size[0]),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        attn_distances = []
        attn_entropies = []
        
        for i, blk in enumerate(self.blocks):
            x, attn_distance, attn_entropy = blk(x, self.positions)
            attn_distances.append(attn_distance)
            attn_entropies.append(attn_entropy)

        # output_dir = "/home/admin/don/Gridpe/eval_results/gridpe_deit_small_patch16_LS_4"
        # if output_dir is not None and dist.get_rank() == 0:
        #     os.makedirs(output_dir, exist_ok=True)

        #     # write attention distance
        #     with open(os.path.join(output_dir, "attn_distance.txt"), "a") as f_dist:
        #         for i, d in enumerate(attn_distances):
        #             f_dist.write(f"Block {i}: {d:.6f}\n")

        #     # write attention entropy
        #     with open(os.path.join(output_dir, "attn_entropy.txt"), "a") as f_ent:
        #         for i, e in enumerate(attn_entropies):
        #             f_ent.write(f"Block {i}: {e:.6f}\n")
                
                
        x = self.norm(x)
        x = x[:, 0]

        return x


# gridpe
@register_model
def gridpe_deit_small_patch16_LS(
    pretrained=False, num_heads = 4,img_size=224, pretrained_21k=False, **kwargs
):
    model = gridpe_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=GridPE_Layer_scale_init_Block,
        Attention_block=GridPEAttention,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def gridpe_deit_base_patch16_LS(
    pretrained=False, num_heads = 6, img_size=224, pretrained_21k=False, **kwargs
):
    model = gridpe_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=864,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=GridPE_Layer_scale_init_Block,
        Attention_block=GridPEAttention,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def gridpe_deit_large_patch16_LS(
    pretrained=False, num_heads = 8, img_size=224, pretrained_21k=False, **kwargs
):
    model = gridpe_vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1056,
        depth=24,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=GridPE_Layer_scale_init_Block,
        Attention_block=GridPEAttention,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model