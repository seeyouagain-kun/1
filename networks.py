"""
networks.py — 共享编码器网络架构
networks.py — Shared encoder network architectures

从 PPO 训练脚本中提取出来，供 PPO 和 PEARL 共享使用。
Extracted from the PPO training script; shared between PPO and PEARL.

网络结构 / Network architectures:
  - MLPEncoder:    状态向量 → 128 维特征  (state vector → 128-dim features)
  - CNNEncoder:    雷达 19×36 → 128 维特征 (lidar 19×36 → 128-dim features)
  - FusionEncoder: MLP + CNN → 256 维融合特征 (combines MLP + CNN → 256-dim)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    MLP 编码器，用于提取状态特征。
    MLP encoder for extracting state features.

    输入：state_dim 维状态向量
    输出：output_dim 维特征向量（默认 128）
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """
    CNN 编码器，用于提取雷达特征。
    CNN encoder for extracting lidar features.

    输入：(B, H, W) 或 (B, 1, H, W) 的雷达数据
    输出：output_dim 维特征向量（默认 128）

    网络结构（以默认 19×36 为例）：
    Input (B, 19, 36) → unsqueeze → (B, 1, 19, 36)
    Conv(1→32, k=3, p=1)  → (B, 32, 19, 36)
    MaxPool(2)             → (B, 32,  9, 18)
    Conv(32→64, k=3, p=1) → (B, 64,  9, 18)
    MaxPool(2)             → (B, 64,  4,  9)
    Conv(64→128, k=3, p=1)→ (B, 128, 4,  9)
    AdaptiveAvgPool(1)     → (B, 128, 1,  1)
    Flatten                → (B, 128)
    Linear(128→output_dim) + Tanh → (B, output_dim)
    """

    def __init__(self, input_shape: Tuple[int, int], output_dim: int = 128):
        super().__init__()
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)
        return x


class FusionEncoder(nn.Module):
    """
    融合编码器，组合 MLP（状态）和 CNN（雷达）特征。
    Fusion encoder combining MLP (state) and CNN (lidar) features.

    输出维度 = state_feature_dim + lidar_feature_dim（默认 256）
    Output dim = state_feature_dim + lidar_feature_dim (default 256)
    """

    def __init__(
        self,
        state_dim: int,
        lidar_shape: Tuple[int, int],
        state_feature_dim: int = 128,
        lidar_feature_dim: int = 128,
    ):
        super().__init__()
        self.mlp_encoder = MLPEncoder(state_dim, state_feature_dim)
        self.cnn_encoder = CNNEncoder(lidar_shape, lidar_feature_dim)
        self.output_dim: int = state_feature_dim + lidar_feature_dim

    def forward(self, state: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        state_features = self.mlp_encoder(state)
        lidar_features = self.cnn_encoder(lidar)
        return torch.cat([state_features, lidar_features], dim=-1)
