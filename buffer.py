"""
pearl/buffer.py — 多任务回放缓冲区
pearl/buffer.py — Multi-task Replay Buffer

维护每个任务独立的回放缓冲区，支持：
  - 上下文采样（用于上下文编码器推断 z）
  - SAC 训练数据采样
  - 每任务独立存储

Maintains separate replay buffers per task, supporting:
  - Context sampling (for context encoder to infer z)
  - SAC training data sampling
  - Per-task independent storage
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class TaskReplayBuffer:
    """
    单任务回放缓冲区（循环缓冲区）。
    Single-task replay buffer (circular buffer).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        """
        参数 / Parameters:
            capacity:   缓冲区容量 / Buffer capacity
            obs_dim:    观测维度 / Observation dimension
            action_dim: 动作维度 / Action dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """添加一个 transition。Add a single transition."""
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = float(reward)
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def bulk_add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """批量添加多个 transition（向量化，比逐条 add 更快）。
        Bulk-add multiple transitions using vectorized numpy indexing.

        参数形状 / Parameter shapes: (T, dim) where T = number of transitions.
        rewards and dones may be 1-D (T,) or 2-D (T, 1).
        """
        T = len(obs)
        if T == 0:
            return
        # 若 T 超过容量，只保留最后 capacity 条 / Keep only last capacity entries if T > capacity
        if T > self.capacity:
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_obs = next_obs[-self.capacity:]
            dones = dones[-self.capacity:]
            T = self.capacity

        indices = np.arange(self._ptr, self._ptr + T) % self.capacity
        self._obs[indices] = obs
        self._actions[indices] = actions
        self._rewards[indices] = rewards.reshape(-1, 1) if rewards.ndim == 1 else rewards
        self._next_obs[indices] = next_obs
        self._dones[indices] = (
            dones.reshape(-1, 1).astype(np.float32) if dones.ndim == 1 else dones.astype(np.float32)
        )

        self._ptr = (self._ptr + T) % self.capacity
        self._size = min(self._size + T, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机采样一个 batch。
        Sample a random batch.

        返回 / Returns:
            obs, actions, rewards, next_obs, dones — 各自的 Tensor
        """
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self._obs[idxs], device=device),
            torch.as_tensor(self._actions[idxs], device=device),
            torch.as_tensor(self._rewards[idxs], device=device),
            torch.as_tensor(self._next_obs[idxs], device=device),
            torch.as_tensor(self._dones[idxs], device=device),
        )

    def sample_context(
        self, context_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样用于上下文编码的 transitions（不含 done）。
        Sample transitions for context encoding (without done flag).

        返回 / Returns:
            obs, actions, rewards, next_obs — Tensors of shape (context_size, dim)
        """
        idxs = np.random.randint(0, self._size, size=context_size)
        return (
            torch.as_tensor(self._obs[idxs], device=device),
            torch.as_tensor(self._actions[idxs], device=device),
            torch.as_tensor(self._rewards[idxs], device=device),
            torch.as_tensor(self._next_obs[idxs], device=device),
        )

    @property
    def size(self) -> int:
        """当前缓冲区中的 transition 数量。Number of transitions currently stored."""
        return self._size

    def __len__(self) -> int:
        return self._size


class MultiTaskReplayBuffer:
    """
    多任务回放缓冲区。
    Multi-task Replay Buffer.

    为每个任务维护独立的 TaskReplayBuffer。
    Maintains independent TaskReplayBuffer instances for each task.
    """

    def __init__(
        self,
        num_tasks: int,
        capacity_per_task: int,
        obs_dim: int,
        action_dim: int,
    ):
        """
        参数 / Parameters:
            num_tasks:          任务数量 / Number of tasks
            capacity_per_task:  每个任务的缓冲区容量 / Buffer capacity per task
            obs_dim:            观测维度 / Observation dimension
            action_dim:         动作维度 / Action dimension
        """
        self.num_tasks = num_tasks
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._buffers: List[TaskReplayBuffer] = [
            TaskReplayBuffer(capacity_per_task, obs_dim, action_dim)
            for _ in range(num_tasks)
        ]

    # ------------------------------------------------------------------
    # 添加数据 / Adding data
    # ------------------------------------------------------------------

    def add(
        self,
        task_idx: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        向指定任务缓冲区添加一个 transition。
        Add a transition to the specified task buffer.
        """
        self._buffers[task_idx].add(obs, action, reward, next_obs, done)

    def add_episode(
        self,
        task_idx: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        批量添加一个 episode 的数据（向量化）。
        Bulk-add an entire episode's data using vectorized numpy insertion.

        参数形状 / Parameter shapes: (T, dim) where T = episode length
        """
        self._buffers[task_idx].bulk_add(
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_observations, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # 采样接口 / Sampling interface
    # ------------------------------------------------------------------

    def sample_context(
        self,
        task_idx: int,
        context_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从指定任务中采样上下文 transitions（用于上下文编码器）。
        Sample context transitions from the specified task (for context encoder).

        返回 / Returns:
            obs, actions, rewards, next_obs — shape (context_size, dim)
        """
        return self._buffers[task_idx].sample_context(context_size, device)

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从指定任务中采样训练 batch（用于 SAC 更新）。
        Sample a training batch from the specified task (for SAC update).

        返回 / Returns:
            obs, actions, rewards, next_obs, dones — shape (batch_size, dim)
        """
        return self._buffers[task_idx].sample(batch_size, device)

    def sample_multi_task(
        self,
        task_indices: List[int],
        batch_size: int,
        context_size: int,
        device: torch.device,
    ) -> Tuple[
        List[Tuple[torch.Tensor, ...]],
        List[Tuple[torch.Tensor, ...]],
    ]:
        """
        从多个任务中同时采样训练 batch 和上下文。
        Sample training batches and contexts from multiple tasks simultaneously.

        参数 / Parameters:
            task_indices: 任务索引列表 / List of task indices
            batch_size:   每任务的训练 batch 大小 / Training batch size per task
            context_size: 每任务的上下文大小 / Context size per task
            device:       目标设备 / Target device

        返回 / Returns:
            contexts: List[(obs, actions, rewards, next_obs)] — 每任务的上下文
            batches:  List[(obs, actions, rewards, next_obs, dones)] — 每任务的 batch
        """
        contexts = [
            self.sample_context(idx, context_size, device) for idx in task_indices
        ]
        batches = [
            self.sample_batch(idx, batch_size, device) for idx in task_indices
        ]
        return contexts, batches

    def sample_multitask_batch_stacked(
        self,
        task_indices: List[int],
        batch_size: int,
        context_size: int,
        device: torch.device,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Stack samples from multiple tasks into a single large batch for efficient GPU processing.
        Returns:
            (stacked_context_tuple, stacked_train_tuple)
            Where each tensor has shape (num_tasks * size, dim) or (num_tasks, size, dim)
        """
        ctx_list = []
        train_list = []
        
        for idx in task_indices:
            ctx_list.append(self.sample_context(idx, context_size, device))
            train_list.append(self.sample_batch(idx, batch_size, device))
            
        # Unzip and stack contexts
        # ctx_list is List[(o, a, r, no)]
        # zip(*ctx_list) -> [(o1, o2...), (a1, a2...), ...]
        c_obs, c_act, c_rew, c_next = zip(*ctx_list)
        # Stack to (num_tasks, context_size, dim)
        stacked_context = (
            torch.stack(c_obs),
            torch.stack(c_act),
            torch.stack(c_rew),
            torch.stack(c_next),
        )

        # Unzip and stack training batches
        # train_list is List[(o, a, r, no, d)]
        t_obs, t_act, t_rew, t_next, t_done = zip(*train_list)
        # Stack and flatten first two dims: (num_tasks * batch_size, dim)
        # We use cat to merge the task dimension into the batch dimension
        stacked_train = (
            torch.cat(t_obs),
            torch.cat(t_act),
            torch.cat(t_rew),
            torch.cat(t_next),
            torch.cat(t_done),
        )

        return stacked_context, stacked_train

    # ------------------------------------------------------------------
    # 查询接口 / Query interface
    # ------------------------------------------------------------------

    def task_size(self, task_idx: int) -> int:
        """返回指定任务缓冲区的当前大小。Return the current size of the specified task buffer."""
        return self._buffers[task_idx].size

    def ready(self, task_idx: int, min_size: int) -> bool:
        """
        检查指定任务缓冲区是否有足够数据（用于开始训练）。
        Check whether the specified task buffer has enough data to start training.
        """
        return self._buffers[task_idx].size >= min_size

    def all_ready(self, task_indices: List[int], min_size: int) -> bool:
        """检查所有指定任务是否都有足够数据。Check all specified tasks have enough data."""
        return all(self.ready(idx, min_size) for idx in task_indices)
