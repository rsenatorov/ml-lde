# src/network/losses.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Combined L1 + SSIM loss for LDE:
    total = 0.7 * L1(pred, gt) + 0.3 * (1 - SSIM(pred, gt))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

class LDELoss(nn.Module):
    """
    Simple weighted sum of L1 and SSIM losses.
    """
    def __init__(self, weight_l1=0.7, weight_ssim=0.3):
        super().__init__()
        assert abs(weight_l1 + weight_ssim - 1.0) < 1e-6, \
               "Weights must sum to 1.0"
        self.l1_fn = nn.L1Loss()
        self.w1 = weight_l1
        self.w2 = weight_ssim

    def forward(self, pred_dict, gt, rgb=None):
        pred = pred_dict['pred']
        # L1 loss
        l1_val = self.l1_fn(pred, gt)
        # SSIM loss (1 - SSIM index)
        ssim_index = ssim(pred, gt, data_range=1.0)
        ssim_val = 1.0 - torch.nan_to_num(ssim_index, nan=0.0)
        # combined
        total = self.w1 * l1_val + self.w2 * ssim_val
        return {
            'loss': total,
            'l1': l1_val.detach(),
            'ssim': ssim_val.detach()
        }
