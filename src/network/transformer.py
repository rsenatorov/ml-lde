# src/network/transformer.py
# Copyright (c) 2025, Robert Senatorov
# All rights reserved.
"""
Pre-Normalization Transformer Block for LDE.
"""
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Single pre-norm transformer block:
    LN -> MHSA -> Add -> LN -> MLP -> Add
    """
    def __init__(self, dim=384, heads=6, mlp_dim=1536, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x
