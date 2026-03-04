"""
pearl/trainer.py — PEARL Meta-Training Loop
pearl/trainer.py — PEARL 元训练循环

实现 PEARL 的元训练程序：
Implements the PEARL meta-training procedure:

  for each iteration:
      1. 从任务分布采样一批任务
         Sample a batch of tasks from task distribution
      2. 对每个任务：
         For each task:
         (a) 用先验 z ~ N(0, I) 收集上下文数据
             Collect context data using prior z ~ N(0, I)
         (b) 从上下文推断后验 z ~ q(z|c)
             Infer posterior z ~ q(z|context)
         (c) 用后验 z 收集训练数据
             Collect training data using posterior z
         (d) 存入每任务回放缓冲区
             Store in per-task replay buffer
      3. 多步梯度更新：
         Multiple gradient update steps:
         - 采样上下文 → 推断 z
           Sample context → infer z
         - 采样训练 batch → SAC + KL 损失
           Sample training batch → SAC + KL loss
         - 更新所有网络
           Update all networks

参考 / Reference:
  Rakelly, K. et al. (2019). Efficient Off-Policy Meta-Reinforcement Learning
  via Probabilistic Context Variables. ICML 2019.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch

from navigation3d.meta_env import MetaNavigation3DEnv
from navigation3d.pearl.agent import PEARLAgent
from navigation3d.pearl.buffer import MultiTaskReplayBuffer


class PEARLTrainer:
    """
    PEARL 元训练器。
    PEARL Meta-Trainer.

    管理元训练循环，包括任务采样、数据收集、网络更新和检查点保存。
    Manages the meta-training loop: task sampling, data collection, network updates,
    and checkpoint saving.
    """

    def __init__(
        self,
        meta_envs: List[MetaNavigation3DEnv],
        agent: PEARLAgent,
        buffer: MultiTaskReplayBuffer,
        # 任务配置 / Task config
        num_train_tasks: int = 100,
        tasks_per_iter: int = 5,
        # 数据收集配置 / Data collection config
        context_collect_steps: int = 200,
        train_collect_steps: int = 200,
        # 训练配置 / Training config
        num_iterations: int = 500,
        gradient_steps_per_iter: int = 200,
        batch_size: int = 256,
        context_size: int = 64,
        min_buffer_size: int = 512,
        # 保存与日志配置 / Save & logging config
        save_dir: str = "runs/pearl",
        save_interval: int = 50,
        log_interval: int = 10,
        seed: int = 42,
    ):
        """
        参数 / Parameters:
            meta_env:              MetaNavigation3DEnv 实例 / MetaNavigation3DEnv instance
            agent:                 PEARLAgent 实例 / PEARLAgent instance
            buffer:                MultiTaskReplayBuffer 实例 / MultiTaskReplayBuffer instance
            num_train_tasks:       元训练任务总数 / Total number of meta-train tasks
            tasks_per_iter:        每次迭代采样的任务数 / Tasks sampled per iteration
            context_collect_steps: 每任务用先验 z 收集的步数 / Steps collected per task with prior z
            train_collect_steps:   每任务用后验 z 收集的步数 / Steps collected per task with posterior z
            num_iterations:        元训练迭代次数 / Number of meta-training iterations
            gradient_steps_per_iter: 每次迭代的梯度更新步数 / Gradient steps per iteration
            batch_size:            SAC 训练 batch 大小 / SAC training batch size
            context_size:          上下文采样大小 / Context sampling size
            min_buffer_size:       开始训练前的最小缓冲区大小 / Min buffer size before training starts
            save_dir:              检查点保存目录 / Checkpoint save directory
            save_interval:         检查点保存间隔（迭代数）/ Checkpoint save interval (iterations)
            log_interval:          日志打印间隔（迭代数）/ Log print interval (iterations)
            seed:                  随机种子 / Random seed
        """
        self.meta_envs = meta_envs
        self.agent = agent
        self.buffer = buffer

        self.num_train_tasks = num_train_tasks
        self.tasks_per_iter = tasks_per_iter
        self.context_collect_steps = context_collect_steps
        self.train_collect_steps = train_collect_steps
        self.num_iterations = num_iterations
        self.gradient_steps_per_iter = gradient_steps_per_iter
        self.batch_size = batch_size
        self.context_size = context_size
        self.min_buffer_size = min_buffer_size
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval

        os.makedirs(save_dir, exist_ok=True)

        np.random.seed(seed)
        torch.manual_seed(seed)

        # 预先生成元训练任务库（固定）
        # Pre-generate a fixed pool of meta-train tasks
        if isinstance(self.meta_envs, list):
            sample_fn = self.meta_envs[0].sample_task
        else:
            # Assume MetaVectorEnv
            sample_fn = self.meta_envs.sample_task
            
        self.task_pool: List[Dict[str, Any]] = [
            sample_fn() for _ in range(num_train_tasks)
        ]

        self._total_env_steps = 0
        self._iteration = 0
        self._start_iteration = 0

    def load_checkpoint(self, path: str) -> None:
        """从检查点恢复 / Resume from checkpoint."""
        import os
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        
        extra_info = self.agent.load(path)
        if "iteration" in extra_info:
            self._start_iteration = extra_info["iteration"] + 1
        if "total_env_steps" in extra_info:
            self._total_env_steps = extra_info["total_env_steps"]
        print(f"Resumed from {path} at iteration {self._start_iteration}")

    # ------------------------------------------------------------------
    # 数据收集 / Data collection
    # ------------------------------------------------------------------

    def _collect_steps_batched(
        self,
        task_indices: List[int],
        tasks: List[Dict[str, Any]],
        z_batch: torch.Tensor,
        num_steps: int,
    ) -> None:
        """
        在给定的多个任务和多个 z 向量下，并行收集 num_steps 步数据存入各自缓冲区。
        Collect num_steps steps of data across multiple environments in parallel.
        """
        # 如果是 MetaVectorEnv，使用向量化收集 / Use vectorized collection if MetaVectorEnv
        if not isinstance(self.meta_envs, list):
            return self._collect_steps_batched_vec(task_indices, tasks, z_batch, num_steps)

        n_envs = len(task_indices)
        assert n_envs <= len(self.meta_envs), "Not enough environments for batch collection."
        
        obs_list = []
        for i in range(n_envs):
            self.meta_envs[i].set_task(tasks[i])
            obs_dict, _ = self.meta_envs[i].reset(new_task=False)
            obs_list.append(self._flatten_obs(obs_dict))

        # (n_envs, obs_dim)
        obs_batch = np.stack(obs_list)
        device = self.agent.device

        for _ in range(num_steps):
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            # z_batch 应该是 (n_envs, latent_dim) / z_batch is (n_envs, latent_dim)
            action_tensor = self.agent.act(obs_tensor, z_batch, deterministic=False)
            action_batch = action_tensor.cpu().numpy()

            next_obs_list = []
            for i in range(n_envs):
                next_obs_dict, reward, terminated, truncated, _ = self.meta_envs[i].step(action_batch[i])
                next_obs = self._flatten_obs(next_obs_dict)
                done = terminated or truncated

                self.buffer.add(task_indices[i], obs_batch[i], action_batch[i], reward, next_obs, done)
                self._total_env_steps += 1

                if done:
                    obs_dict, _ = self.meta_envs[i].reset(new_task=False)
                    next_obs_list.append(self._flatten_obs(obs_dict))
                else:
                    next_obs_list.append(next_obs)
            
            obs_batch = np.stack(next_obs_list)

    def _collect_steps_batched_vec(
        self,
        task_indices: List[int],
        tasks: List[Dict[str, Any]],
        z_batch: torch.Tensor,
        num_steps: int,
    ) -> None:
        """
        使用 MetaVectorEnv 进行并行数据收集。
        Parallel data collection using MetaVectorEnv.
        """
        n_envs = len(task_indices)
        # Check if we align with vec_env capacity
        # Assume self.meta_envs.num_envs >= n_envs
        # But for simplicity, we assume n_envs matches exactly the number of workers created
        # or we just use the first n_envs workers.
        
        # 1. 设置任务 / Set tasks
        self.meta_envs.set_tasks(tasks)

        # 2. Reset (with new_task=False)
        # obs_list_dicts is List[Dict]
        obs_list_dicts, infos = self.meta_envs.reset(new_task=False)
        
        # Flatten observations
        obs_batch = np.stack([self._flatten_obs(o) for o in obs_list_dicts])
        device = self.agent.device

        for _ in range(num_steps):
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            
            # Agent inference
            action_tensor = self.agent.act(obs_tensor, z_batch, deterministic=False)
            action_batch = action_tensor.cpu().numpy()
            
            # Step environments
            next_obs_dicts, rewards, dones, infos = self.meta_envs.step(action_batch)
            
            # Process results
            next_obs_batch = []
            for i in range(n_envs):
                next_obs_flat = self._flatten_obs(next_obs_dicts[i])
                
                # Check if done. If done, next_obs_dicts[i] is the NEW reset observation.
                # The terminal observation is in info.
                real_next_obs = next_obs_flat
                if dones[i]:
                    term_obs = infos[i].get("terminal_observation")
                    if term_obs is not None:
                        real_next_obs = self._flatten_obs(term_obs)
                
                self.buffer.add(task_indices[i], obs_batch[i], action_batch[i], rewards[i], real_next_obs, dones[i])
                self._total_env_steps += 1
                
                # obs_batch for next step is already next_obs_flat (which is reset one if done)
                next_obs_batch.append(next_obs_flat)
            
            obs_batch = np.stack(next_obs_batch)

    @staticmethod
    def _flatten_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """将 dict 观测拼合为单一向量。Flatten dict obs into a single vector."""
        return np.concatenate([obs_dict["state"], obs_dict["lidar"].flatten()]).astype(np.float32)

    # ------------------------------------------------------------------
    # 梯度更新 / Gradient updates
    # ------------------------------------------------------------------

    def _gradient_step(
        self, task_indices: List[int]
    ) -> Dict[str, float]:
        """
        对给定任务列表进行 batched 梯度更新。
        Perform batched gradient update for the given list of tasks.
        """
        if not self.buffer.all_ready(task_indices, self.min_buffer_size):
            return {}

        device = self.agent.device
        
        # 设置最大批处理样本数，防止显存溢出
        # Set max samples per forward pass to prevent OOM
        # 4096 is conservative and safe for most GPUs (approx 1-2GB VRAM usage for buffers)
        MAX_BATCH_SAMPLES = 4096 
        
        num_tasks = len(task_indices)
        # 每个 chunk 包含多少个任务 / How many tasks per chunk
        tasks_per_chunk = max(1, MAX_BATCH_SAMPLES // self.batch_size)
        
        # 将任务索引切分为多个 chunk / Split task indices into chunks
        task_chunks = [task_indices[i:i + tasks_per_chunk] for i in range(0, num_tasks, tasks_per_chunk)]
        
        metrics_accum = {}
        steps_count = 0

        for chunk in task_chunks:
            # 1. Batched sampling for this chunk
            ctx_tuple, train_tuple = self.buffer.sample_multitask_batch_stacked(
                chunk, self.batch_size, self.context_size, device
            )
            
            ctx_obs, ctx_act, ctx_rew, ctx_next = ctx_tuple
            # The buffer method returns 5 items for training batch
            obs, actions, rewards, next_obs, dones = train_tuple
            
            # 2. Infer posterior z
            with torch.no_grad():
                z, _, _ = self.agent.infer_z(ctx_obs, ctx_act, ctx_rew, ctx_next, sample=True)

            # Expand z to match training batch size
            z_expanded = z.repeat_interleave(self.batch_size, dim=0)

            # 3. Update Critic
            critic_info = self.agent.update_critic(obs, actions, rewards, next_obs, dones, z_expanded)
            
            # 4. Update Actor and α
            actor_info = self.agent.update_actor_and_alpha(obs, z_expanded)
            
            # 5. Update Context Encoder
            encoder_info = self.agent.update_encoder(
                ctx_obs, ctx_act, ctx_rew, ctx_next,
                obs, actions, rewards, next_obs, dones,
            )

            # Accumulate metrics
            chunk_metrics = {**critic_info, **actor_info, **encoder_info}
            for k, v in chunk_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            
            # 6. Soft update target Critic (once per chunk update)
            self.agent.soft_update_target()
            
            steps_count += 1

        # Average metrics
        if steps_count > 0:
            for k in metrics_accum:
                metrics_accum[k] /= steps_count
            metrics_accum["alpha"] = self.agent.alpha.item()
            
        return metrics_accum

    # ------------------------------------------------------------------
    # 主训练循环 / Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        运行完整的 PEARL 元训练循环。
        Run the full PEARL meta-training loop.
        """
        print("=" * 60)
        print("PEARL Meta-Training")
        print("=" * 60)
        print(f"  num_train_tasks:    {self.num_train_tasks}")
        print(f"  tasks_per_iter:     {self.tasks_per_iter}")
        print(f"  num_iterations:     {self.num_iterations}")
        print(f"  gradient_steps:     {self.gradient_steps_per_iter}")
        print(f"  batch_size:         {self.batch_size}")
        print(f"  context_size:       {self.context_size}")
        print(f"  device:             {self.agent.device}")
        print("=" * 60)

        start_time = time.time()

        # 使用 tqdm 包装训练循环
        pbar = tqdm(range(self._start_iteration, self.num_iterations), desc="Meta-Training", initial=self._start_iteration, total=self.num_iterations)
        for iteration in pbar:
            self._iteration = iteration

            # ---- 采样任务 / Sample tasks ----
            task_indices = np.random.choice(
                self.num_train_tasks, size=self.tasks_per_iter, replace=False
            ).tolist()
            tasks = [self.task_pool[idx] for idx in task_indices]

            # ---- Batched Data collection phase ----
            
            # (a) 用先验 z 收集上下文数据
            #     Collect context data using prior z
            prior_z_batch = self.agent.get_prior_z(batch_size=len(task_indices))
            self._collect_steps_batched(task_indices, tasks, prior_z_batch, self.context_collect_steps)

            # (b) 从收集的上下文推断后验 z / Infer posterior z from context
            posterior_z_list = []
            for task_idx in task_indices:
                if self.buffer.ready(task_idx, self.context_size):
                    device = self.agent.device
                    ctx_obs, ctx_act, ctx_rew, ctx_next = self.buffer.sample_context(
                        task_idx, self.context_size, device
                    )
                    with torch.no_grad():
                        z, _, _ = self.agent.infer_z(ctx_obs, ctx_act, ctx_rew, ctx_next)
                    posterior_z_list.append(z)
                else:
                    posterior_z_list.append(self.agent.get_prior_z(batch_size=1).squeeze(0))
            
            posterior_z_batch = torch.stack(posterior_z_list)

            # (c) 用后验 z 收集训练数据
            #     Collect training data using posterior z
            self._collect_steps_batched(task_indices, tasks, posterior_z_batch, self.train_collect_steps)

            # ---- 梯度更新阶段 / Gradient update phase ----
            update_metrics: Dict[str, float] = {}
            if self.buffer.all_ready(task_indices, self.min_buffer_size):
                for _ in range(self.gradient_steps_per_iter):
                    step_metrics = self._gradient_step(task_indices)
                    for k, v in step_metrics.items():
                        update_metrics[k] = update_metrics.get(k, 0.0) + v / self.gradient_steps_per_iter
                
                # 如果有更新，可以在进度条显示损失
                # pbar.set_postfix(metrics=update_metrics) # 或者选择几个重要的显示

            # ---- 日志 / Logging ----
            if (iteration + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                pbar.write(
                    f"[Iter {iteration + 1:4d}/{self.num_iterations}] "
                    f"env_steps={self._total_env_steps:,} "
                    f"elapsed={elapsed:.1f}s"
                )
                for k, v in update_metrics.items():
                    pbar.write(f"  {k}: {v:.4f}")

            # ---- 检查点保存 / Checkpoint saving ----
            if (iteration + 1) % self.save_interval == 0:
                ckpt_path = os.path.join(self.save_dir, f"pearl_iter_{iteration + 1}.pt")
                self.agent.save(ckpt_path, extra_info={"iteration": iteration, "total_env_steps": self._total_env_steps})
                pbar.write(f"Checkpoint saved: {ckpt_path}")

        # 最终模型保存 / Final model save
        final_path = os.path.join(self.save_dir, "pearl_final.pt")
        self.agent.save(final_path, extra_info={"iteration": self.num_iterations - 1, "total_env_steps": self._total_env_steps})
        print(f"\nMeta-training complete. Final model saved to: {final_path}")
