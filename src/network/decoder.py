# src/network/decoder.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
FPN-style decoder, Gated Refinement Unit, and prediction head for LDE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    """
    A lightweight block to fuse an upsampled feature map with a skip connection.
    Uses replication padding to avoid border artifacts.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate', bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate', bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x, skip):
        # upsample then concat with skip
        x = F.interpolate(x,
                          size=skip.shape[-2:],
                          mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GatedRefinementUnit(nn.Module):
    """
    Refines depth features using guidance from the RGB image edges.
    Learns spatial gates to sharpen or smooth depth predictions.
    """
    def __init__(self, in_channels, guidance_channels=3, gate_channels=32):
        super().__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate'),
            nn.GELU()
        )
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(guidance_channels, gate_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate'),
            nn.GELU()
        )
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels + gate_channels,
                      gate_channels,
                      kernel_size=3, padding=1,
                      padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(gate_channels, 1,
                      kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, depth_feat, rgb_image):
        # resize rgb to match depth_feat
        rgb_guidance = F.interpolate(
            rgb_image,
            size=depth_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        refined = self.depth_conv(depth_feat)
        guide = self.guidance_conv(rgb_guidance)
        gate_in = torch.cat([refined, guide], dim=1)
        gate = self.gate_generator(gate_in)
        return depth_feat + refined * gate

class PredictionHead(nn.Module):
    """
    Upsamples to target size and predicts a 1-channel depth map in [0,1].
    """
    def __init__(self, in_channels, target_size=(224,224)):
        super().__init__()
        self.target_size = target_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=3, padding=1,
                      padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1,
                      kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x,
                          size=self.target_size,
                          mode='bilinear',
                          align_corners=False)
        return self.net(x)