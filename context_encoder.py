from __future__ import annotations

"""
context_encoder.py — PEARL 上下文编码器
context_encoder.py — PEARL Context Encoder

实现 PEARL 论文（Rakelly et al., 2019）中的概率嵌入上下文编码器：
Implements the probabilistic embedding context encoder from PEARL (Rakelly et al., 2019):
  - 输入：一组 transition {(s, a, r, s')} 作为上下文 c
    Input: a set of transitions {(s, a, r, s')} as context c
  - 独立编码每个 transition → 均值与方差
    Each transition is independently encoded → mean and variance
  - 通过高斯乘积（product of Gaussians）聚合，具有置换不变性
    Aggregation via product of Gaussians (permutation invariant)
  - 输出：后验分布 q(z|c) = N(μ, σ²)（任务隐变量 z 的后验）
    Output: posterior distribution q(z|c) = N(μ, σ²) over latent task variable z
  - 与 N(0, I) 先验之间的 KL 散度用于正则化
    KL divergence against N(0, I) prior used for regularization

参考 / Reference:
  Rakelly, K. et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning
  via Probabilistic Context Variables. ICML 2019.
  https://arxiv.org/abs/1903.08254
"""
"""实现了对上下文：（s,a,r,s'）的批处理：context->通过 MLP 得到每个 context 的 (μ_i, σ²_i）->通过高斯乘积聚合，得到这个 batch 的 context 的后验分布 q(z|c) 的参数（μ, σ）->支持重参数化采样 z ~ N(μ, σ²) 和 KL 散度计算。"""


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """
    PEARL 上下文编码器（概率嵌入）。
    PEARL Context Encoder (probabilistic embedding).

    将一个 transition batch (s, a, r, s') 编码为任务隐变量 z 的后验分布 q(z|c)。
    Encodes a transition batch (s, a, r, s') into a posterior q(z|c) over task latent z.

    内部流程 / Internal flow:
      1. 将每个 transition 拼接为单一向量
         Concatenate each transition into a single vector
      2. 独立地用 MLP 编码每个 transition → (μ_i, log_σ²_i)
         Independently encode each transition with an MLP → (μ_i, log_σ²_i)
      3. 高斯乘积聚合：精度（1/σ²）求和，然后转回均值/方差
         Product of Gaussians aggregation: sum precisions (1/σ²), convert back to mean/var
      4. 返回聚合后的 μ 和 σ（支持重参数化采样）
         Return aggregated μ and σ (supports reparameterized sampling)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_next_obs: bool = True,
    ):
        """
        参数 / Parameters:
            obs_dim:     观测维度（拼合后的 state+lidar 维度）
                         Observation dimension (flattened state+lidar)
            action_dim:  动作维度 / Action dimension
            latent_dim:  任务隐变量 z 的维度（默认 5）
                         Latent task variable z dimension (default 5)
            hidden_dim:  MLP 隐藏层维度（默认 256）
                         MLP hidden layer dimension (default 256)
            num_layers:  MLP 层数（默认 3）/ MLP number of layers (default 3)
            use_next_obs: 是否将 s' 加入 transition 输入（默认 True）
                          Whether to include s' in transition input (default True)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_next_obs = use_next_obs

        # transition 输入维度：s + a + r (+ s')
        transition_dim = obs_dim + action_dim + 1  # +1 for reward
        if use_next_obs:
            transition_dim += obs_dim

        # 构建 MLP 网络：每个 transition → (μ, log_σ²) in R^{2 * latent_dim}
        # Build MLP: each transition → (μ, log_σ²) in R^{2 * latent_dim}
        layers: list[nn.Module] = [] # 这是一个列表，且列表里的每一个元素都必须是 nn.Module 的实例
        in_dim = transition_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        # 最后一层输出 2 * latent_dim（μ 和 log_σ²）
        # Last layer outputs 2 * latent_dim (μ and log_σ²)
        layers.append(nn.Linear(in_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # 前向传播 / Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码一批 transition，返回后验分布的参数。
        Encode a batch of transitions and return posterior distribution parameters.

        参数 / Parameters:
            obs:      (B, obs_dim)    — 当前观测 / current observations
            actions:  (B, action_dim) — 动作 / actions
            rewards:  (B, 1) or (B,) — 奖励 / rewards
            next_obs: (B, obs_dim)    — 下一观测 / next observations

        返回 / Returns:
            mu:    (latent_dim,) — 后验均值 / posterior mean
            sigma: (latent_dim,) — 后验标准差（正数）/ posterior std (positive)
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1) # 确保 rewards 形状为 (B, 1)

        if self.use_next_obs:
            x = torch.cat([obs, actions, rewards, next_obs], dim=-1)
        else:
            x = torch.cat([obs, actions, rewards], dim=-1)

        # 编码每个 transition / Encode each transition
        out = self.encoder(x)  # (..., 2 * latent_dim)
        mu_per = out[..., : self.latent_dim]            # (..., latent_dim)
        log_var_per = out[..., self.latent_dim :]       # (..., latent_dim)

        # 高斯乘积聚合
        # Product of Gaussians aggregation
        mu, sigma = self._product_of_gaussians(mu_per, log_var_per)
        return mu, sigma

    def _product_of_gaussians(
        self, mu_per: torch.Tensor, log_var_per: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算多个高斯分布的乘积（精度加权）。
        Compute the product of multiple Gaussian distributions (precision-weighted).
        Supports (N, C, Latent) input -> Aggregates over dimension 1 (C).
        Supports (C, Latent) input -> Aggregates over dimension 0 (C).

        高斯乘积公式 / Product of Gaussians formula:
            precision_total = Σ precision_i   (precision_i = 1 / var_i)
            mu_total        = (Σ precision_i * mu_i) / precision_total
            var_total       = 1 / precision_total
        """
        # Dim to aggregate over
        agg_dim = 1 if mu_per.dim() == 3 else 0

        # 将 log_var 裁剪到合理范围以提高数值稳定性
        # Clamp log_var for numerical stability
        log_var_per = torch.clamp(log_var_per, min=-10.0, max=10.0)
        var_per = torch.exp(log_var_per)                  
        precision_per = 1.0 / (var_per + 1e-8)           

        precision_total = precision_per.sum(dim=agg_dim)        
        mu_numerator = (precision_per * mu_per).sum(dim=agg_dim)
        
        mu = mu_numerator / (precision_total + 1e-8)
        var = 1.0 / (precision_total + 1e-8)              
        sigma = torch.sqrt(var + 1e-8)
        return mu, sigma

    def sample(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        从后验分布中重参数化采样 z ~ N(μ, σ²)。
        Reparameterized sample z ~ N(μ, σ²) from the posterior.

        参数 / Parameters:
            mu:    (latent_dim,) or (B, latent_dim)
            sigma: (latent_dim,) or (B, latent_dim)

        返回 / Returns:
            z: 与 mu/sigma 相同形状的采样结果 / sampled z with same shape as mu/sigma
        """
        eps = torch.randn_like(mu) # 标准正态噪声
        return mu + sigma * eps

    def kl_divergence(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        计算后验 q(z|c) = N(μ, σ²) 与标准先验 p(z) = N(0, I) 之间的 KL 散度。
        Compute KL divergence between posterior q(z|c) = N(μ, σ²) and prior p(z) = N(0, I).

        KL[N(μ, σ²) || N(0, I)] = 0.5 * Σ (σ² + μ² - 1 - log σ²)

        参数 / Parameters:
            mu:    (latent_dim,) or (B, latent_dim)
            sigma: (latent_dim,) or (B, latent_dim)

        返回 / Returns:
            kl: 标量 KL 散度 / scalar KL divergence
        """
        var = sigma ** 2
        kl = 0.5 * (var + mu ** 2 - 1.0 - torch.log(var + 1e-8))
        return kl.sum()

    def encode_context(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        便捷接口：编码上下文并可选地采样 z。
        Convenience method: encode context and optionally sample z.

        返回 / Returns:
            z:     采样或均值 (latent_dim,) / sampled or mean z
            mu:    (latent_dim,)
            sigma: (latent_dim,)
        """
        mu, sigma = self.forward(obs, actions, rewards, next_obs)
        if sample:
            z = self.sample(mu, sigma)
        else:
            z = mu
        return z, mu, sigma
