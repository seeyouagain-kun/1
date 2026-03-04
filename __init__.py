"""
navigation3d.pearl — PEARL Meta-RL 实现
navigation3d.pearl — PEARL Meta-RL implementation

基于 SAC + 概率上下文编码器的 Meta-RL 算法实现。
Meta-RL algorithm implementation based on SAC + probabilistic context encoder.

参考 / Reference:
  Rakelly, K. et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning
  via Probabilistic Context Variables. ICML 2019.
"""

from navigation3d.pearl.agent import PEARLAgent
from navigation3d.pearl.buffer import MultiTaskReplayBuffer
from navigation3d.pearl.trainer import PEARLTrainer

__all__ = ["PEARLAgent", "MultiTaskReplayBuffer", "PEARLTrainer"]
