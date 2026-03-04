"""
MetaNavigation3DEnv — 任务分布包装器
MetaNavigation3DEnv — Task distribution wrapper for Meta-RL

将 Navigation3DEnv 包装成支持任务分布采样的 Meta-RL 环境。
提供可配置的任务参数范围（均匀分布），用于 sim-to-real 域随机化。

Wraps Navigation3DEnv to support task distribution sampling for Meta-RL.
Provides configurable task parameter ranges (uniform distribution) for
sim-to-real domain randomization.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navigation3d.env import Navigation3DEnv


# 默认任务参数范围（用于域随机化 / sim-to-real）
# Default task parameter ranges for domain randomization / sim-to-real
TASK_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    # Physics & Dynamics
    "max_velocity": (75.0, 300.0),       # Base 150 * [0.5, 2]
    "max_angular_vel": (0.5, 2.0),       # Base 1.0 * [0.5, 2]
    "dt": (0.05, 0.15),                  # Limited to 0.15s to prevent tunneling (300*0.15 = 45 < 50 min diameter)

    # Rewards & Penalties
    "distance_reward_scale": (0.5, 2.0), # Base 1.0 * [0.5, 2]
    "time_penalty": (0.25, 1.0),         # Base 0.5 * [0.5, 2]
    "obstacle_collision_penalty": (-2000.0, -500.0), # Base -1000 * [2, 0.5] magnitude
    "goal_reward": (500.0, 2000.0),      # Base 1000 * [0.5, 2]
    "boundary_penalty": (-2000.0, -500.0), # Base -1000 * [2, 0.5] magnitude

    # Structure
    "num_obstacles": (5, 20),            # Base 10 * [0.5, 2] (Integer)
    "obstacle_radius": (25.0, 100.0),    # Base 50 * [0.5, 2]
    "bounds": (1500.0, 6000.0),          # Base 3000 * [0.5, 2]
    "cluster_spread": (150.0, 600.0),    # Base 300 * [0.5, 2]
    "agent_scale": (0.5, 2.0),           # Base 1.0 * [0.5, 2]
}


class MetaNavigation3DEnv:
    """
    Meta-RL 任务分布包装器。
    Meta-RL task distribution wrapper.

    提供：
    - 可配置的任务参数分布（均匀范围）
    - sample_task() → 返回任务字典
    - set_task(task) → 应用任务到底层环境
    - reset() 可选地采样新任务
    - meta-train / meta-test 任务划分

    Provides:
    - Configurable task parameter distribution (uniform ranges)
    - sample_task() → returns a task dict
    - set_task(task) → applies task to underlying env
    - reset() that optionally samples a new task
    - meta-train / meta-test task split support
    """

    def __init__(
        self,
        env: Optional[Navigation3DEnv] = None,
        task_param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        sample_task_on_reset: bool = True,
        meta_train: bool = True,
        train_task_fraction: float = 0.8,
        seed: Optional[int] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        参数 / Parameters:
            env: 底层 Navigation3DEnv 实例；若为 None 则用 env_kwargs 创建。
                 Underlying Navigation3DEnv instance; created from env_kwargs if None.
            task_param_ranges: 任务参数范围字典，格式 {param: (low, high)}。
                               Task parameter range dict, format {param: (low, high)}.
            sample_task_on_reset: 是否在每次 reset() 时采样新任务。
                                  Whether to sample a new task on each reset().
            meta_train: True 表示 meta-train 模式，False 表示 meta-test 模式。
                        True for meta-train mode, False for meta-test mode.
            train_task_fraction: 用于 meta-train 的任务参数空间比例（按哈希划分）。
                                 Fraction of task space reserved for meta-train (hash split).
            seed: 随机种子（用于任务采样）。Random seed for task sampling.
            env_kwargs: 传给 Navigation3DEnv 的关键字参数（当 env=None 时使用）。
                        Keyword arguments passed to Navigation3DEnv when env is None.
        """
        if env is None:
            env_kwargs = env_kwargs or {}
            env_kwargs.setdefault("verbose", False)
            env = Navigation3DEnv(**env_kwargs)

        self.env = env
        self.task_param_ranges: Dict[str, Tuple[float, float]] = (
            task_param_ranges if task_param_ranges is not None else copy.deepcopy(TASK_PARAM_RANGES)
        )
        self.sample_task_on_reset = sample_task_on_reset
        self.meta_train = meta_train
        self.train_task_fraction = float(train_task_fraction)

        # 内部 RNG（与 Gymnasium env 的 np_random 分离）
        # Internal RNG separate from the Gymnasium env's np_random
        self._rng = np.random.default_rng(seed)

        self._current_task: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Gymnasium-compatible delegation
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def metadata(self):
        return self.env.metadata

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def sample_task(self) -> Dict[str, Any]:
        """
        从任务参数分布中采样一个任务（均匀分布）。
        持续采样直到采到属于当前 meta-train/meta-test 划分的任务。

        Sample a task from the task parameter distribution (uniform).
        Keeps sampling until a task belonging to the current meta-train/test split is found.
        """
        for _ in range(1000):
            task: Dict[str, Any] = {}
            for param, (low, high) in self.task_param_ranges.items():
                if param == "num_obstacles":
                    # 整数参数 / integer parameter
                    task[param] = int(self._rng.integers(int(low), int(high) + 1))
                else:
                    task[param] = float(self._rng.uniform(low, high))

            # 用任务参数的字符串表示做哈希，决定 train/test 归属（稳定且均匀）
            # Use string repr of task params for hashing to get stable, uniform train/test split
            fingerprint = hash(repr(sorted(task.items()))) & 0x7FFFFFFF
            in_train = (fingerprint % 1000) < int(self.train_task_fraction * 1000)

            if (self.meta_train and in_train) or (not self.meta_train and not in_train):
                return task

        # 兜底：直接返回最后一个采样（避免死循环）
        # Fallback: return the last sampled task to avoid infinite loop
        return task  # type: ignore[return-value]

    def get_task(self) -> Optional[Dict[str, Any]]:
        """
        返回当前任务参数字典（若尚未设置任务则返回 None）。
        Return the current task parameter dict, or None if no task has been set.
        """
        return self._current_task

    def set_task(self, task: Dict[str, Any]) -> None:
        """
        将任务参数应用到底层环境。
        Apply task parameters to the underlying environment.
        """
        self._current_task = task
        self.env.set_task(task)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        new_task: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境，可选地采样新任务。
        Reset the environment, optionally sampling a new task.

        参数 / Parameters:
            seed: Gymnasium reset seed.
            options: 传给底层 env.reset() 的 options（可包含 "task" 键）。
                     Options passed to env.reset() (may include "task" key).
            new_task: 若为 True 或 self.sample_task_on_reset 为 True，则先采样新任务。
                      If True or self.sample_task_on_reset is True, sample a new task first.
        """
        if new_task or self.sample_task_on_reset:
            task = self.sample_task()
            self.set_task(task)

        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        将动作传给底层环境。
        Pass action to the underlying environment.
        """
        return self.env.step(action)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_task_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """返回任务参数范围字典的副本。Return a copy of the task parameter ranges dict."""
        return copy.deepcopy(self.task_param_ranges)

    def set_task_param_ranges(self, ranges: Dict[str, Tuple[float, float]]) -> None:
        """更新任务参数范围。Update task parameter ranges."""
        self.task_param_ranges.update(ranges)

    def __repr__(self) -> str:
        return (
            f"MetaNavigation3DEnv("
            f"meta_train={self.meta_train}, "
            f"sample_task_on_reset={self.sample_task_on_reset}, "
            f"current_task={self._current_task})"
        )
