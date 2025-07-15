# src/network/encoder.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Encoder for LDE: hybrid Vision Transformer and lightweight CNN.
"""
import torch
import torch.nn as nn
from .transformer import TransformerBlock

class PatchEmbed(nn.Module):
    """
    Splits image into patches and projects to embedding space.
    """
    def __init__(self, img_size=224, patch_size=16,
                 in_ch=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1,2)
        return x                       # [B, n_patches, E]

class VisionTransformerEncoder(nn.Module):
    """
    Custom ViT encoder extracting multiscale features.
    """
    def __init__(self, img_size=224, patch_size=16,
                 in_ch=3, embed_dim=384,
                 depth=12, n_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_ch, embed_dim
        )
        n_patches = (img_size // patch_size)**2
        self.cls_token = nn.Parameter(
            torch.zeros(1,1,embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches+1, embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim,
                             heads=n_heads,
                             mlp_dim=embed_dim*4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        feats = []
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i in [2,5,8,11]:
                xn = self.norm(x)
                fmap = xn[:,1:,:].permute(0,2,1)
                ph = H//self.patch_embed.patch_size
                pw = W//self.patch_embed.patch_size
                feats.append(fmap.reshape(B, -1, ph, pw))
        return feats

class FineDetailEncoder(nn.Module):
    """
    Lightweight CNN to extract fine spatial details.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2,
                      kernel_size=3, stride=2,
                      padding=1, padding_mode='replicate',
                      bias=False),
            nn.GroupNorm(8, out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate',
                      bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)