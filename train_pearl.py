"""
scripts/train_pearl.py — PEARL 元训练入口点
scripts/train_pearl.py — PEARL meta-training entry point

用法 / Usage:
    python scripts/train_pearl.py [OPTIONS]

示例 / Examples:
    # 默认设置
    python scripts/train_pearl.py

    # 自定义迭代次数和任务数
    python scripts/train_pearl.py --num_iterations 1000 --num_train_tasks 200

    # 快速冒烟测试
    python scripts/train_pearl.py --smoke

选项 / Options:
    --num_iterations INT         元训练迭代次数（默认 500）
                                 Number of meta-training iterations (default 500)
    --num_train_tasks INT        元训练任务数量（默认 100）
                                 Number of meta-train tasks (default 100)
    --tasks_per_iter INT         每次迭代采样的任务数（默认 5）
                                 Tasks sampled per iteration (default 5)
    --latent_dim INT             任务隐变量维度（默认 5）
                                 Task latent variable dimension (default 5)
    --context_collect_steps INT  每任务用先验 z 收集的步数（默认 200）
                                 Steps collected per task with prior z (default 200)
    --train_collect_steps INT    每任务用后验 z 收集的步数（默认 200）
                                 Steps collected per task with posterior z (default 200)
    --gradient_steps INT         每次迭代的梯度更新步数（默认 200）
                                 Gradient steps per iteration (default 200)
    --batch_size INT             SAC 训练 batch 大小（默认 256）
                                 SAC training batch size (default 256)
    --context_size INT           上下文采样大小（默认 64）
                                 Context sampling size (default 64)
    --buffer_capacity INT        每任务缓冲区容量（默认 50000）
                                 Buffer capacity per task (default 50000)
    --save_dir STR               检查点保存目录（默认 runs/pearl）
                                 Checkpoint directory (default runs/pearl)
    --save_interval INT          检查点保存间隔（默认 50）
                                 Checkpoint save interval (default 50)
    --seed INT                   随机种子（默认 42）/ Random seed (default 42)
    --smoke                      快速冒烟测试（减少迭代数和步数）
                                 Quick smoke test (reduced iterations and steps)
    --meta_test                  使用 meta-test 任务划分（默认 meta-train）
                                 Use meta-test task split (default is meta-train)
"""

from __future__ import annotations

import argparse
import os
import sys

# Set threads to 1 to avoid contention in multiprocessing workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing as mp

# 将项目根目录加入 sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from navigation3d.meta_env import MetaNavigation3DEnv
from navigation3d.pearl.agent import PEARLAgent
from navigation3d.pearl.buffer import MultiTaskReplayBuffer
from navigation3d.pearl.trainer import PEARLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PEARL Meta-RL training for 3D Navigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_iterations", type=int, default=500,
                        help="元训练迭代次数 / Number of meta-training iterations")
    parser.add_argument("--num_train_tasks", type=int, default=100,
                        help="元训练任务数量 / Number of meta-train tasks")
    parser.add_argument("--tasks_per_iter", type=int, default=None,
                        help="每次迭代采样的任务数（默认：CPU核心数） / Tasks sampled per iteration")
    parser.add_argument("--latent_dim", type=int, default=5,
                        help="任务隐变量维度 / Task latent variable dimension")
    parser.add_argument("--context_collect_steps", type=int, default=200,
                        help="每任务先验采集步数 / Steps collected per task with prior z")
    parser.add_argument("--train_collect_steps", type=int, default=200,
                        help="每任务后验采集步数 / Steps collected per task with posterior z")
    parser.add_argument("--gradient_steps", type=int, default=200,
                        help="每次迭代梯度更新步数 / Gradient steps per iteration")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="SAC 训练 batch 大小 / SAC training batch size")
    parser.add_argument("--context_size", type=int, default=64,
                        help="上下文采样大小 / Context sampling size")
    parser.add_argument("--buffer_capacity", type=int, default=10_000,
                        help="每任务缓冲区容量 / Buffer capacity per task")
    parser.add_argument("--save_dir", type=str, default="runs/pearl",
                        help="检查点保存目录 / Checkpoint save directory")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="检查点保存间隔（迭代） / Checkpoint save interval (iterations)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志打印间隔（迭代） / Log print interval (iterations)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 / Random seed")
    parser.add_argument("--smoke", action="store_true",
                        help="快速冒烟测试 / Quick smoke test")
    parser.add_argument("--meta_test", action="store_true",
                        help="使用 meta-test 任务划分 / Use meta-test task split")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定检查点文件恢复训练 / Resume training from specified checkpoint")
    return parser.parse_args()


def make_env_factory(rank: int, seed: int, meta_test: bool):
    """Helper to create environment factory function for multiprocessing."""
    def _thunk():
        return MetaNavigation3DEnv(
            meta_train=not meta_test,
            sample_task_on_reset=False,
            seed=seed + rank,
            env_kwargs={
                "verbose": False,
                "use_curriculum": False,
            },
        )
    return _thunk


def main() -> None:
    args = parse_args()

    # Determine tasks_per_iter if not set
    if args.tasks_per_iter is None:
        cpu_count = mp.cpu_count()
        # Use a safe heuristic: cpu_count or slightly fewer
        # For hyperthreading CPUs, cpu_count might be high.
        # But simulation is CPU intensive.
        # Let's cap at 16 or cpu_count
        args.tasks_per_iter = max(1, cpu_count)
        print(f"Auto-configured tasks_per_iter to {args.tasks_per_iter} (CPU count: {cpu_count})")

    # 冒烟测试：减小规模 / Smoke test: reduce scale
    if args.smoke:
        args.num_iterations = 3
        args.num_train_tasks = 10
        args.tasks_per_iter = 2
        args.context_collect_steps = 20
        args.train_collect_steps = 20
        args.gradient_steps = 5
        args.batch_size = 32
        args.context_size = 16
        args.buffer_capacity = 500
        print("[SMOKE] Running in smoke test mode with reduced settings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 批量创建 MetaNavigation3DEnv / Create MetaNavigation3DEnvs ----
    # 使用 MetaVectorEnv 进行多进程加速 / Use MetaVectorEnv for multiprocessing acceleration
    
    # 1. 创建一个 Dummy 环境用于获取 Observation/Action Space
    dummy_env = MetaNavigation3DEnv(meta_train=not args.meta_test, seed=args.seed)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    
    # 2. 创建环境工厂函数列表
    env_fns = [make_env_factory(i, args.seed, args.meta_test) for i in range(args.tasks_per_iter)]
    
    # 3. 初始化向量化环境
    try:
        from navigation3d.pearl.vec_env import MetaVectorEnv
        print(f"Initializing MetaVectorEnv with {len(env_fns)} processes...")
        meta_envs = MetaVectorEnv(env_fns)
    except Exception as e:
        print(f"Warning: Failed to create MetaVectorEnv ({e}). Falling back to serial list.")
        meta_envs = [fn() for fn in env_fns]

    # ---- 计算观测维度和动作维度 / Compute obs_dim and action_dim ----
    # 拼合观测维度（state + lidar 展平）/ Flattened obs dim (state + flattened lidar)
    state_dim = obs_space["state"].shape[0]
    lidar_shape = obs_space["lidar"].shape  # (H, W)
    obs_dim = state_dim + lidar_shape[0] * lidar_shape[1]
    action_dim = act_space.shape[0]

    print(f"State dim: {state_dim}, Lidar shape: {lidar_shape}")
    print(f"Obs dim (flattened): {obs_dim}, Action dim: {action_dim}")

    # ---- 创建 PEARL Agent / Create PEARL Agent ----
    agent = PEARLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        lidar_shape=lidar_shape,
        latent_dim=args.latent_dim,
        device=device,
    )

    # ---- 创建多任务回放缓冲区 / Create multi-task replay buffer ----
    buffer = MultiTaskReplayBuffer(
        num_tasks=args.num_train_tasks,
        capacity_per_task=args.buffer_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # ---- 创建 Trainer / Create Trainer ----
    trainer = PEARLTrainer(
        meta_envs=meta_envs,
        agent=agent,
        buffer=buffer,
        num_train_tasks=args.num_train_tasks,
        tasks_per_iter=args.tasks_per_iter,
        context_collect_steps=args.context_collect_steps,
        train_collect_steps=args.train_collect_steps,
        num_iterations=args.num_iterations,
        gradient_steps_per_iter=args.gradient_steps,
        batch_size=args.batch_size,
        context_size=args.context_size,
        save_dir=args.save_dir,
        min_buffer_size=args.batch_size, # Ensure we have at least one batch
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()

    # 关闭环境资源 / Close environment resources
    print("Closing environments...")
    if hasattr(meta_envs, "close") and callable(meta_envs.close):
        meta_envs.close()
    else:
        for env in meta_envs:
            env.close()

    if 'dummy_env' in locals():
        dummy_env.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) # Ensure consistency for multiprocessing
    main()
