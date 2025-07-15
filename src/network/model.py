# src/network/model.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
LDE_Model: hybrid Transformer-CNN for light depth estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisionTransformerEncoder, FineDetailEncoder
from .decoder import FusionBlock, GatedRefinementUnit, PredictionHead

class LDE_Model(nn.Module):
    """
    Combines ViT global context and CNN local details in a multi-stage decoder.
    """
    def __init__(self,
                 img_size=224,
                 vit_patch_size=16,
                 vit_embed_dim=384,
                 vit_depth=12,
                 vit_n_heads=6,
                 cnn_out_ch=64,
                 decoder_channels=[256,128,64]):
        super().__init__()
        print("[INFO] Initializing LDE_Model from scratch.")

        # encoder branches
        self.vit_encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            in_ch=3,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            n_heads=vit_n_heads
        )
        self.cnn_encoder = FineDetailEncoder(
            in_channels=3, out_channels=cnn_out_ch
        )

        # initial projection of deepest vit feature
        self.initial_proj = nn.Conv2d(
            vit_embed_dim, decoder_channels[0], kernel_size=1
        )

        # fusion blocks for FPN
        self.fusion_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels:
            self.fusion_blocks.append(
                FusionBlock(in_channels=in_ch + vit_embed_dim,
                            out_channels=out_ch)
            )
            in_ch = out_ch

        # hybrid fusion of vit decoder and cnn features
        self.hybrid_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels[-1] + cnn_out_ch,
                      decoder_channels[-1],
                      kernel_size=3, padding=1,
                      padding_mode='replicate'),
            nn.GELU()
        )

        # refinement and prediction
        self.refinement = GatedRefinementUnit(
            in_channels=decoder_channels[-1],
            guidance_channels=3
        )
        self.pred_head = PredictionHead(
            in_channels=decoder_channels[-1],
            target_size=(img_size, img_size)
        )

    def forward(self, x):
        vit_feats = self.vit_encoder(x)
        cnn_feat = self.cnn_encoder(x)
        vit_feats = vit_feats[::-1]

        # vit decoding
        f = self.initial_proj(vit_feats[0])
        for i, fb in enumerate(self.fusion_blocks):
            skip = vit_feats[i+1]
            f = fb(f, skip)

        # hybrid fusion
        f_up = F.interpolate(f,
                             size=cnn_feat.shape[-2:],
                             mode='bilinear',
                             align_corners=False)
        h = torch.cat([f_up, cnn_feat], dim=1)
        fused = self.hybrid_fusion(h)

        # refinement and prediction
        r = self.refinement(fused, x)
        pred = self.pred_head(r)
        return {'pred': pred}