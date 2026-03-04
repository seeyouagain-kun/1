"""
pearl/agent.py — PEARL Agent（基于 SAC + 上下文推断）
pearl/agent.py — PEARL Agent (SAC-based with context inference)

实现 PEARL 算法的核心 Agent：
Implements the core Agent for the PEARL algorithm:

  - Actor π(a|s, z)：将 FusionEncoder 输出与任务隐变量 z 拼接后输出动作分布
    Actor π(a|s, z): concatenates FusionEncoder output with task latent z, outputs action distribution
  - Critic Q(s, a, z)：双 Q 网络，输入状态特征、动作和 z，输出 Q 值
    Critic Q(s, a, z): twin Q-networks, takes state features, action, and z, outputs Q-value
  - 上下文编码器：从 transitions 推断任务隐变量 z 的后验分布
    Context encoder: infers posterior of task latent z from transitions
  - 训练损失 = SAC 损失 + KL 散度（对 z 的正则化）
    Training loss = SAC loss + KL divergence (regularization on z)

参考 / Reference:
  Rakelly, K. et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning
  via Probabilistic Context Variables. ICML 2019.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from navigation3d.networks import FusionEncoder
from navigation3d.context_encoder import ContextEncoder


# ==================== Actor 网络 ====================

class PEARLActor(nn.Module):
    """
    PEARL Actor π(a|s, z)。
    输入：state 特征（FusionEncoder 输出）+ 任务隐变量 z
    输出：高斯策略的均值和对数标准差（用于 SAC 的 squashed Gaussian 策略）

    Input:  state features (FusionEncoder output) + task latent z
    Output: mean and log_std of Gaussian policy (for SAC squashed Gaussian)
    """

    # log_std 的范围 [-20, 2] 对应 std 的范围 [2e-9, 7.39]，足够覆盖大多数环境。
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = feature_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # 初始化最后一层使策略初始探索均匀
        # Initialize last layer for uniform initial exploration
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(
        self, features: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数 / Parameters:
            features: (B, feature_dim)
            z:        (B, latent_dim) or (latent_dim,)

        返回 / Returns:
            mean:    (B, action_dim)
            log_std: (B, action_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(features.shape[0], -1)
        x = torch.cat([features, z], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, features: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重参数化采样动作并计算对数概率（SAC 使用）。
        Reparameterized action sampling with log probability (used by SAC).

        返回 / Returns:
            action:   (B, action_dim) — tanh squashed action in [-1, 1]
            log_prob: (B, 1)          — log probability
        """
        mean, log_std = self.forward(features, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # 根据 mu, sigma 进行采样，使得梯度能够反向传播
        x_t = normal.rsample()  # 重参数化 / reparameterized 
        action = torch.tanh(x_t)
        # SAC 的 squashed Gaussian log prob 修正
        # SAC squashed Gaussian log prob correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act_deterministic(self, features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """确定性动作（评估时使用）。Deterministic action (for evaluation)."""
        mean, _ = self.forward(features, z)
        return torch.tanh(mean)


# ==================== Critic 网络（双 Q 网络）====================

class PEARLCritic(nn.Module):
    """
    PEARL Critic Q(s, a, z)（双 Q 网络）。
    Twin Q-networks for PEARL: Q(s, a, z).

    输入：state 特征 + 动作 + 任务隐变量 z
    输出：两个 Q 值（Q1, Q2）

    Input:  state features + action + task latent z
    Output: two Q values (Q1, Q2)
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = feature_dim + action_dim + latent_dim

        def _build_q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _build_q()
        self.q2 = _build_q()

        for net in [self.q1, self.q2]:
            nn.init.orthogonal_(net[-1].weight, gain=1.0)
            nn.init.zeros_(net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数 / Parameters:
            features: (B, feature_dim)
            action:   (B, action_dim)
            z:        (B, latent_dim) or (latent_dim,)

        返回 / Returns:
            q1: (B, 1)
            q2: (B, 1)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(features.shape[0], -1)
        x = torch.cat([features, action, z], dim=-1)
        return self.q1(x), self.q2(x)


# ==================== PEARL Agent ====================

class PEARLAgent:
    """
    PEARL Agent：SAC 策略 + 上下文推断。
    PEARL Agent: SAC policy + context inference.

    包含：
      - FusionEncoder（state+lidar → 特征）
      - ContextEncoder（transitions → z 后验）
      - Actor π(a|s, z)
      - Critic Q(s, a, z)（双 Q + 目标网络）
      - SAC 可学习温度系数 α

    Contains:
      - FusionEncoder (state+lidar → features)
      - ContextEncoder (transitions → z posterior)
      - Actor π(a|s, z)
      - Critic Q(s, a, z) (twin Q + target networks)
      - SAC learnable temperature α
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lidar_shape: Tuple[int, int],
        state_dim: int = 11,
        latent_dim: int = 5,
        hidden_dim: int = 256,
        context_hidden_dim: int = 256,
        context_num_layers: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_encoder: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        kl_weight: float = 0.1,
        target_entropy: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        参数 / Parameters:
            obs_dim:              拼合后的观测维度（state+lidar 展平后）
                                  Flattened observation dimension (state + flattened lidar)
            action_dim:           动作维度 / Action dimension
            lidar_shape:          雷达数据形状 / Lidar data shape
            state_dim:            状态子向量维度（默认 11）/ State sub-vector dim (default 11)
            latent_dim:           任务隐变量 z 的维度（默认 5）/ Latent task variable z dim (default 5)
            hidden_dim:           Actor/Critic 隐藏层维度（默认 256）
                                  Actor/Critic hidden layer dim (default 256)
            context_hidden_dim:   上下文编码器隐藏层维度（默认 256）
                                  Context encoder hidden layer dim (default 256)
            context_num_layers:   上下文编码器层数（默认 3）
                                  Context encoder number of layers (default 3)
            lr_actor:             Actor 学习率 / Actor learning rate
            lr_critic:            Critic 学习率 / Critic learning rate
            lr_encoder:           上下文编码器学习率 / Context encoder learning rate
            lr_alpha:             温度系数学习率 / Temperature coefficient learning rate
            gamma:                折扣因子 / Discount factor
            tau:                  目标网络软更新系数 / Target network soft update coefficient
            kl_weight:            KL 散度权重 / KL divergence weight
            target_entropy:       SAC 目标熵（默认 -action_dim）
                                  SAC target entropy (default -action_dim)
            device:               计算设备 / Computing device
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lidar_shape = lidar_shape
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.kl_weight = kl_weight

        # ---- 网络构建 / Build networks ----
        self.fusion_encoder = FusionEncoder(
            state_dim=state_dim,
            lidar_shape=lidar_shape,
            state_feature_dim=128,
            lidar_feature_dim=128,
        ).to(device)
        feature_dim = self.fusion_encoder.output_dim  # 256

        self.context_encoder = ContextEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=context_hidden_dim,
            num_layers=context_num_layers,
        ).to(device)

        self.actor = PEARLActor(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.critic = PEARLCritic(
            feature_dim=feature_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # 目标 Critic（软更新）/ Target Critic (soft update)
        self.critic_target = PEARLCritic(
            feature_dim=feature_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # SAC 可学习温度 α / SAC learnable temperature α
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        if target_entropy is None:
            target_entropy = float(-action_dim)
        self.target_entropy = target_entropy

        # ---- 优化器 / Optimizers ----
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.fusion_encoder.parameters()),
            lr=lr_critic,
        )
        self.encoder_optimizer = optim.Adam(self.context_encoder.parameters(), lr=lr_encoder)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    # ------------------------------------------------------------------
    # 辅助方法 / Helper methods
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """SAC 温度系数 α（正数）。SAC temperature α (positive)."""
        return self.log_alpha.exp()

    def _parse_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将拼合观测拆分为 state 向量和 lidar 张量。
        Split flattened observation into state vector and lidar tensor.

        参数 / Parameters:
            obs: (B, obs_dim)

        返回 / Returns:
            state: (B, state_dim)
            lidar: (B, H, W)
        """
        state = obs[:, : self.state_dim]
        lidar_flat = obs[:, self.state_dim :]
        lidar = lidar_flat.view(-1, self.lidar_shape[0], self.lidar_shape[1])
        return state, lidar

    def encode_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        用 FusionEncoder 从观测中提取特征。
        Extract features from observations using FusionEncoder.

        参数 / Parameters:
            obs: (B, obs_dim)

        返回 / Returns:
            features: (B, feature_dim)
        """
        state, lidar = self._parse_obs(obs)
        return self.fusion_encoder(state, lidar)

    def infer_z(
        self,
        context_obs: torch.Tensor,
        context_actions: torch.Tensor,
        context_rewards: torch.Tensor,
        context_next_obs: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从上下文 transitions 推断任务隐变量 z 的后验。
        Infer the posterior of task latent z from context transitions.

        返回 / Returns:
            z:     (latent_dim,) — 采样或均值
            mu:    (latent_dim,)
            sigma: (latent_dim,)
        """
        return self.context_encoder.encode_context(
            context_obs, context_actions, context_rewards, context_next_obs, sample=sample
        )

    def get_prior_z(self, batch_size: int = 1) -> torch.Tensor:
        """
        从先验 N(0, I) 采样 z（用于收集数据时的初始推断）。
        Sample z from prior N(0, I) (used for initial inference when collecting data).

        返回 / Returns:
            z: (batch_size, latent_dim) if batch_size > 1 else (latent_dim,)
        """
        if batch_size > 1:
            return torch.zeros((batch_size, self.latent_dim), device=self.device)
        return torch.zeros(self.latent_dim, device=self.device)

    # ------------------------------------------------------------------
    # 更新步骤 / Update steps
    # ------------------------------------------------------------------

    def update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        更新 Critic（双 Q 网络）。
        Update Critic (twin Q-networks).

        使用目标网络和当前 Actor 计算 TD 目标（SAC 风格）。
        Computes TD targets using target network and current Actor (SAC style).
        """
        with torch.no_grad():
            next_features = self.encode_features(next_obs)
            next_action, next_log_prob = self.actor.sample(next_features, z)
            q1_target, q2_target = self.critic_target(next_features, next_action, z)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            td_target = rewards + self.gamma * (1.0 - dones) * q_target

        features = self.encode_features(obs)
        q1, q2 = self.critic(features, actions, z)

        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.fusion_encoder.parameters()), 1.0
        )
        self.critic_optimizer.step()

        return {"critic_loss": critic_loss.item()}

    def update_actor_and_alpha(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """
        更新 Actor 和温度系数 α。
        Update Actor and temperature coefficient α.
        """
        features = self.encode_features(obs)
        action, log_prob = self.actor.sample(features, z)
        q1, q2 = self.critic(features, action, z)
        q_min = torch.min(q1, q2)

        # Actor 损失：最大化 Q - α * log_prob（即最小化 α * log_prob - Q）
        # Actor loss: maximize Q - α * log_prob (minimize α * log_prob - Q)
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 更新温度系数 α / Update temperature α
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "log_prob": log_prob.mean().item(),
        }

    def update_encoder(
        self,
        context_obs: torch.Tensor,
        context_actions: torch.Tensor,
        context_rewards: torch.Tensor,
        context_next_obs: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        更新上下文编码器（KL 散度 + Critic 辅助损失）。
        Update context encoder (KL divergence + Critic auxiliary loss).

        使用 KL 散度将后验 q(z|c) 拉向先验 N(0, I)，
        同时用 Critic 损失使 z 对任务有区分度。

        Uses KL divergence to pull posterior q(z|c) toward prior N(0, I),
        and Critic loss to make z discriminative for tasks.
        """
        # 推断后验 z（带梯度）
        # Infer posterior z (with gradients)
        z, mu, sigma = self.infer_z(
            context_obs, context_actions, context_rewards, context_next_obs, sample=True
        )

        # KL 散度正则化 / KL divergence regularization
        # Sum over batch/tasks
        kl_loss = self.context_encoder.kl_divergence(mu, sigma)

        # 检查 z 是否需要扩展以匹配 obs 的 batch 大小
        # Check if z needs expansion to match obs batch size
        # z: (num_tasks, latent_dim), obs: (num_tasks * batch_size, obs_dim)
        z_expanded = z
        if z.dim() > 1 and z.shape[0] != obs.shape[0]:
            # Assume obs.shape[0] is multiple of z.shape[0]
            if obs.shape[0] % z.shape[0] == 0:
                full_batch_size = obs.shape[0] // z.shape[0]
                z_expanded = z.repeat_interleave(full_batch_size, dim=0)

        # 用当前 z 计算 Critic 辅助损失（端到端训练编码器）
        # Critic auxiliary loss to train encoder end-to-end
        with torch.no_grad():
            next_features = self.encode_features(next_obs)
            next_action, next_log_prob = self.actor.sample(next_features, z_expanded.detach())
            q1_target, q2_target = self.critic_target(next_features, next_action, z_expanded.detach())
    
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(-1)
            
            # Use sum/mean reduction as needed by MSE, but shape match is key
            # rewards is (TotalBatch, 1), q_target is (TotalBatch, 1)
            td_target = rewards + self.gamma * (1.0 - dones) * q_target

        features = self.encode_features(obs)
        # z 为 (latent_dim,) 或 (NumTasks, latent)
        # 需确保传给 critic 的 z 与 features 第一维度一致
        # Ensure z passed to critic matches first dimension of features
        q1, q2 = self.critic(features, actions, z_expanded)
        
        # Loss accumulation
        encoder_critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        # KL loss needs to be scaled properly if we sum it?
        # encoder_critic_loss is mean over TotalBatch.
        # kl_loss is sum over NumTasks?
        # If we want comparable scales, we might need to normalize KL.
        # But PEARL implementation usually just sums it or takes mean.
        # Here we largely preserve existing logic but enable batching.
        
        encoder_loss = self.kl_weight * kl_loss + encoder_critic_loss

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        nn.utils.clip_grad_norm_(self.context_encoder.parameters(), 1.0)
        self.encoder_optimizer.step()

        return {
            "encoder_loss": encoder_loss.item(),
            "kl_loss": kl_loss.item(),
            "encoder_critic_loss": encoder_critic_loss.item(),
        }

    def soft_update_target(self) -> None:
        """目标 Critic 软更新。Soft update target Critic."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # ------------------------------------------------------------------
    # 动作选择 / Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        选择动作（用于环境交互）。
        Select action (for environment interaction).

        参数 / Parameters:
            obs:          (obs_dim,) 或 (B, obs_dim) / (obs_dim,) or (B, obs_dim)
            z:            (latent_dim,) 任务隐变量 / task latent variable
            deterministic: 是否使用确定性策略 / whether to use deterministic policy

        返回 / Returns:
            action: (action_dim,) — 归一化动作 in [-1, 1]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.encode_features(obs)
        if deterministic:
            action = self.actor.act_deterministic(features, z)
        else:
            action, _ = self.actor.sample(features, z)
        return action.squeeze(0)

    # ------------------------------------------------------------------
    # 模型保存与加载 / Model save & load
    # ------------------------------------------------------------------

    def save(self, path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """保存所有网络权重、优化器状态和额外信息。Save all network weights, optimizer states, and extra info."""
        state = {
            "fusion_encoder": self.fusion_encoder.state_dict(),
            "context_encoder": self.context_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        if extra_info is not None:
            state.update(extra_info)
        torch.save(state, path)

    def load(self, path: str) -> Dict[str, Any]:
        """加载网络权重和优化器状态并返回额外信息。Load network weights and optimizer states, return extra info."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fusion_encoder.load_state_dict(checkpoint["fusion_encoder"])
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.log_alpha = checkpoint["log_alpha"]
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        
        # 返回未知的 key（当作 extra_info）
        # Return unknown keys (as extra_info)
        extra_info = {k: v for k, v in checkpoint.items() if k not in [
            "fusion_encoder", "context_encoder", "actor", "critic", "critic_target",
            "log_alpha", "actor_optimizer", "critic_optimizer", "encoder_optimizer", "alpha_optimizer"
        ]}
        return extra_info

