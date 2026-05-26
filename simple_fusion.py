# simple_fusion.py

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleFusionRegressor(nn.Module):
    """
    Small-data-safe multimodal regressor.

    Inputs:
      sentinel_feat: [B, D_img]
      weather_feat:  [B, D_weather]

    Output:
      pred_norm: [B]
    """

    def __init__(
        self,
        img_dim: int,
        weather_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.img_head = MLPBlock(img_dim, hidden_dim, dropout=dropout)
        self.weather_head = MLPBlock(weather_dim, hidden_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sentinel_feat: torch.Tensor, weather_feat: torch.Tensor) -> torch.Tensor:
        z_img = self.img_head(sentinel_feat)
        z_w = self.weather_head(weather_feat)
        z = torch.cat([z_img, z_w], dim=-1)
        return self.fusion(z).squeeze(-1)