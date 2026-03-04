"""
使用 skrl 框架和 PPO 算法训练三维导航环境
支持 Dict 观测空间(状态 + 雷达数据)
使用 MLP 提取状态特征, CNN 提取雷达特征, 融合后输入 Actor 和 Critic
动作空间：[yaw_rate, pitch_rate, linear_velocity]（速度控制）

编码器已迁移至 navigation3d.networks，供 PPO 和 PEARL 共享。
Encoders have been moved to navigation3d.networks, shared between PPO and PEARL.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import atexit
import signal
import threading
from contextlib import contextmanager
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, Tuple

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from navigation3d.env import Navigation3DEnv
from navigation3d.networks import FusionEncoder


# ==================== 环境配置 ====================
ENV_CONFIG = {
    "bounds": 3000.0,
    "agent_size": (47.0, 6.0, 17.0),
    "max_velocity": 150.0,
    "max_angular_vel": 1.0,
    "reach_threshold": 50.0,
    "num_obstacles": 30,
    "obstacle_radius": 50.0,
    "obstacle_collision_penalty": -1000.0,
    "goal_reward": 1000.0,
    "boundary_penalty": -1000.0,
    "cluster_spread": 300.0,
    "dt": 0.1,
    "lidar_resolution": 10.0,
    "distance_reward_scale": 1.0,
    "time_penalty": 0.1,
    "use_curriculum": True,
    "initial_curriculum":  0.0,
}


# ==================== 策略网络（Actor）====================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum",
                 state_dim=11, lidar_shape=(19, 36)):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        
        self.state_dim = state_dim
        self.lidar_shape = lidar_shape
        
        self.encoder = FusionEncoder(
            state_dim=self.state_dim,
            lidar_shape=self.lidar_shape,
            state_feature_dim=128,
            lidar_feature_dim=128
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.num_actions)
        )
        
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)

        # Encourage initial exploration with non-trivial forward velocity.
        # action[2] maps [-1, 1] -> [0, max_velocity], so a small positive bias helps
        # avoid the early local optimum of near-zero speed.
        if self.num_actions >= 3:
            with torch.no_grad():
                self.policy_head[-1].bias[2] = 0.5
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    
    def compute(self, inputs: Dict[str, Any], role: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]: 
        states = inputs["states"]
        
        state = states[:, :self.state_dim]
        lidar_flat_size = self.lidar_shape[0] * self.lidar_shape[1]
        lidar = states[:, self.state_dim:self.state_dim + lidar_flat_size]
        lidar = lidar.view(-1, self.lidar_shape[0], self.lidar_shape[1])
        
        features = self.encoder(state, lidar)
        mean = self.policy_head(features)
        
        return mean, self.log_std_parameter, {}


# ==================== 价值网络（Critic）====================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 state_dim=11, lidar_shape=(19, 36)):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.state_dim = state_dim
        self.lidar_shape = lidar_shape
        
        self.encoder = FusionEncoder(
            state_dim=self.state_dim,
            lidar_shape=self.lidar_shape,
            state_feature_dim=128,
            lidar_feature_dim=128
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)
    
    def compute(self, inputs: Dict[str, Any], role: str) -> Tuple[torch.Tensor, Dict]:
        states = inputs["states"]
        
        state = states[:, :self.state_dim]
        lidar_flat_size = self.lidar_shape[0] * self.lidar_shape[1]
        lidar = states[:, self.state_dim:self.state_dim + lidar_flat_size]
        lidar = lidar.view(-1, self.lidar_shape[0], self.lidar_shape[1])
        
        features = self.encoder(state, lidar)
        value = self.value_head(features)
        
        return value, {}


# ==================== 观测预处理Wrapper ====================
class FlattenDictObservation(gym.ObservationWrapper):
    """将 Dict 观测扁平化为单个向量"""
    def __init__(self, env):
        super().__init__(env)
        
        state_dim = env.observation_space["state"].shape[0]
        lidar_shape = env.observation_space["lidar"].shape
        lidar_size = lidar_shape[0] * lidar_shape[1]
        total_size = state_dim + lidar_size
        
        self.state_dim = state_dim
        self.lidar_shape = lidar_shape
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
    
    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        state = obs["state"]
        lidar = obs["lidar"].flatten()
        return np.concatenate([state, lidar]).astype(np.float32)


# ==================== 环境工厂函数 ====================
def make_env(env_config: dict, render_mode: str = None):
    """创建并包装环境"""
    def _make():
        env = Navigation3DEnv(
            bounds=env_config["bounds"],
            render_mode=render_mode,
            verbose=False,
            agent_size=env_config["agent_size"],
            max_velocity=env_config["max_velocity"],
            max_angular_vel=env_config["max_angular_vel"],
            reach_threshold=env_config["reach_threshold"],
            num_obstacles=env_config["num_obstacles"],
            obstacle_radius=env_config["obstacle_radius"],
            obstacle_collision_penalty=env_config["obstacle_collision_penalty"],
            # 新增配置
            goal_reward=env_config["goal_reward"],
            boundary_penalty=env_config["boundary_penalty"],
            cluster_spread=env_config["cluster_spread"],
            dt=env_config["dt"],
            # 
            lidar_resolution=env_config["lidar_resolution"],
            distance_reward_scale=env_config["distance_reward_scale"],
            time_penalty=env_config["time_penalty"],
            use_curriculum=env_config["use_curriculum"],
            initial_curriculum=env_config.get("initial_curriculum", 0.0),
        )
        
        env = FlattenDictObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -5.0, 5.0).astype(np.float32),
        )# observation_space=env.observation_space,
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        # env = gym.wrappers.TransformReward(
        #     env,
        #     lambda r: np.clip(r, -10.0, 10.0)
        # )
        
        return env
    
    return _make


def make_eval_env(env_config: dict, render_mode: str = None):
    """创建评估环境"""
    eval_config = env_config.copy()
    eval_config["use_curriculum"] = False
    
    def _make():
        env = Navigation3DEnv(
            bounds=eval_config["bounds"],
            render_mode=render_mode,
            verbose=False,
            agent_size=eval_config["agent_size"],
            max_velocity=eval_config["max_velocity"],
            max_angular_vel=eval_config["max_angular_vel"],
            reach_threshold=eval_config["reach_threshold"],
            # 新增配置
            goal_reward=eval_config["goal_reward"],
            boundary_penalty=eval_config["boundary_penalty"],
            cluster_spread=eval_config["cluster_spread"],
            dt=eval_config["dt"],
            # 
            num_obstacles=eval_config["num_obstacles"],
            obstacle_radius=eval_config["obstacle_radius"],
            obstacle_collision_penalty=eval_config["obstacle_collision_penalty"],
            lidar_resolution=eval_config["lidar_resolution"],
            distance_reward_scale=eval_config["distance_reward_scale"],
            time_penalty=eval_config["time_penalty"],
            use_curriculum=eval_config["use_curriculum"],
        )
        env = FlattenDictObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -5.0, 5.0).astype(np.float32),
        )# observation_space=env.observation_space,
        return env
    
    return _make


def get_lidar_shape(resolution: float) -> Tuple[int, int]:
    """根据分辨率计算雷达数据形状"""
    n_azimuth = int(360 / resolution)
    n_elevation = int(180 / resolution) + 1
    return (n_elevation, n_azimuth)


def _save_checkpoint(agent: PPO, experiment_dir: str, filename: str):
    os.makedirs(experiment_dir, exist_ok=True)
    path = os.path.join(experiment_dir, filename)
    tmp_path = f"{path}.tmp"
    # Write to a temp file first, then atomically replace the target.
    # This reduces the chance of ending up with a missing/partial checkpoint
    # if the process is interrupted during the write.
    agent.save(tmp_path)
    os.replace(tmp_path, path)
    print(f"Checkpoint saved: {path}")


@contextmanager
def _ignore_signals(signals_to_ignore):
    old_handlers = {}
    try:
        for sig in signals_to_ignore:
            try:
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, signal.SIG_IGN)
            except Exception:
                # Some signals may not be available on all platforms
                pass
        yield
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass


def _start_stop_listener(stop_event: threading.Event):
    """监听用户输入，输入 stop/q/quit/exit 触发中断"""
    if not sys.stdin or not sys.stdin.isatty():
        return None

    def _listen():
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd in {"stop", "q", "quit", "exit"}:
                    print("\nStop command received. Stopping training...")
                    stop_event.set()
                    # os.kill(os.getpid(), signal.SIGINT) # Windows下可能行为不一致，改为使用 trainer 的 stop 机制或抛出异常
                    # 由于 skrl 的 trainer.train() 没有公开的 stop 方法，我们通过 KeyboardInterrupt 来模拟
                    import _thread
                    _thread.interrupt_main()
                    break
            except RuntimeError:
                break

    thread = threading.Thread(target=_listen, daemon=True)
    thread.start()
    return thread


# ==================== 训练函数 ====================
def train(
    num_envs: int = 8,
    total_timesteps: int = 10_000_000,
    seed: int = 42,
    experiment_name: str = None,
    mode: str = "train",
    fast: bool = False,
    resume_path: str = None,
):
    set_seed(seed)
    
    if experiment_name is None:
        if resume_path:
            experiment_dir = os.path.dirname(os.path.abspath(resume_path))
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{mode}_{timestamp}"
            experiment_dir = os.path.join("runs", experiment_name)
    else:
        experiment_dir = os.path.join("runs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    env_config = ENV_CONFIG.copy()
    if fast:
        # 仅修改训练迭代参数，不修改环境参数
        total_timesteps = min(total_timesteps, 2_000_000)
        print("[FAST] Using reduced PPO rollouts/epochs for faster iteration (Env config unchanged)")

    print("=" * 60)
    print(f"Training Navigation3D with PPO (skrl)")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Number of environments: {num_envs}")
    print(f"Total timesteps: {total_timesteps: ,}")
    print(f"Seed: {seed}")
    print(f"Action Space: [yaw_rate, pitch_rate, linear_velocity]")
    print("=" * 60)

    vector_env = gym.vector.AsyncVectorEnv(
        [make_env(env_config) for _ in range(num_envs)]
    )
    env = wrap_env(vector_env)
    
    device = env.device
    print(f"Using device: {device}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    lidar_shape = get_lidar_shape(env_config["lidar_resolution"])
    state_dim = 11
    
    print(f"State dim: {state_dim}")
    print(f"Lidar shape: {lidar_shape}")
    
    models = {
        "policy": Policy(env.observation_space, env.action_space, device,
                        state_dim=state_dim, lidar_shape=lidar_shape),
        "value": Value(env.observation_space, env.action_space, device,
                      state_dim=state_dim, lidar_shape=lidar_shape),
    }
    
    policy_params = sum(p.numel() for p in models["policy"].parameters())
    value_params = sum(p.numel() for p in models["value"].parameters())
    print(f"Policy parameters: {policy_params:,}")
    print(f"Value parameters: {value_params:,}")
    
    rollouts = 4096
    memory = RandomMemory(
        memory_size=rollouts,
        num_envs=env.num_envs,
        device=device
    )
    
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 64
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["grad_norm_clip"] = 0.5
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 0.5
    cfg["timesteps"] = total_timesteps
    cfg["experiment"] = {
        "directory": experiment_dir,
        "experiment_name": "",
        "write_interval": 1000,
        "checkpoint_interval": 50000,
        "store_separately": False,
        "wandb": False,
    }
    
    if fast:
        rollouts = 2048
        memory = RandomMemory(
            memory_size=rollouts,
            num_envs=env.num_envs,
            device=device
        )
        cfg["rollouts"] = rollouts
        cfg["learning_epochs"] = 4
        cfg["mini_batches"] = 32

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        agent.load(resume_path)
    
    trainer = SequentialTrainer(
        cfg=cfg,
        env=env,
        agents=[agent]
    )

    # skrl registers an atexit handler that closes the environment.
    # After Ctrl+C (SIGINT), AsyncVectorEnv workers may already be gone and a second
    # close at interpreter shutdown can raise BrokenPipeError. We manage closing
    # explicitly in the finally block, so unregister the atexit hook here.
    try:
        _atexit_close_env = getattr(trainer, "close_env", None)
        if callable(_atexit_close_env):
            atexit.unregister(_atexit_close_env)
    except Exception:
        pass
    
    stop_event = threading.Event()
    _start_stop_listener(stop_event)
    if sys.stdin and sys.stdin.isatty():
        print("Type 'stop' (or 'q') then Enter to stop and save.")

    interrupted = False
    print("\nStarting training...")
    try:
        trainer.train()
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        print("\nTraining interrupted by user. Saving checkpoint...")
    except Exception:
        # Training crashed unexpectedly; still try to save what we have.
        print("\nTraining crashed. Saving checkpoint before re-raising...")
        raise
    finally:
        # Save first (env.close can hang with AsyncVectorEnv after SIGINT).
        # Also ignore SIGINT briefly so a second Ctrl+C doesn't abort the save.
        try:
            with _ignore_signals([signal.SIGINT, getattr(signal, "SIGTERM", None)]):
                _save_checkpoint(agent, experiment_dir, "last_model.pt")  # 确保保存
        except Exception as e:
            print(f"[WARN] Failed to save last_model.pt: {e}")

        # Close the underlying AsyncVectorEnv directly.
        # The skrl wrapper's close() may not accept kwargs and can hang if a worker crashed
        # while a step call is pending.
        try:
            close_attempts = [
                {"terminate": True, "timeout": 0},
                {"terminate": True},
                {"timeout": 0},
                {},
            ]
            closed = False
            for kwargs in close_attempts:
                try:
                    vector_env.close(**kwargs)
                    closed = True
                    break
                except TypeError:
                    continue
            if not closed:
                vector_env.close()
        except Exception as e:
            print(f"[WARN] Failed to close vector env cleanly: {e}")

    if not interrupted:
        final_model_path = os.path.join(experiment_dir, "final_model.pt")
        agent.save(final_model_path)
        print(f"\nTraining completed!  Final model saved to: {final_model_path}")

    return agent, experiment_dir


# ==================== 评估函数 ====================
def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render:  bool = True,
    deterministic: bool = True,
):
    print("=" * 60)
    print("Evaluating trained model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("=" * 60)
    
    render_mode = "human" if render else None
    env = make_eval_env(ENV_CONFIG, render_mode)()
    
    env_for_wrap = gym.wrappers.ClipAction(
        Navigation3DEnv(
            bounds=ENV_CONFIG["bounds"],
            render_mode=None,
            agent_size=ENV_CONFIG["agent_size"],
            verbose=False,
            lidar_resolution=ENV_CONFIG["lidar_resolution"],
            # 补全配置以确保 Observation Space 一致
            num_obstacles=ENV_CONFIG["num_obstacles"],
            dt=ENV_CONFIG["dt"],
        )
    )
    env_for_wrap = FlattenDictObservation(env_for_wrap)
    env_wrapped = wrap_env(env_for_wrap)
    device = env_wrapped.device
    env_for_wrap.close()
    
    lidar_shape = get_lidar_shape(ENV_CONFIG["lidar_resolution"])
    state_dim = 11
    
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(state_dim + lidar_shape[0] * lidar_shape[1],),
        dtype=np.float32
    )
    act_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    
    models = {
        "policy": Policy(obs_space, act_space, device,
                        state_dim=state_dim, lidar_shape=lidar_shape),
        "value": Value(obs_space, act_space, device,
                      state_dim=state_dim, lidar_shape=lidar_shape),
    }
    
    memory = RandomMemory(memory_size=1, num_envs=1, device=device)
    eval_cfg = PPO_DEFAULT_CONFIG.copy()
    # 评估阶段关闭日志/检查点写入，保留结构避免 None 报错
    eval_cfg["experiment"] = {
        "directory": None,
        "experiment_name": "",
        "write_interval": 0,
        "checkpoint_interval": 0,
        "store_separately": False,
        "wandb": False,
    }
    eval_agent = PPO(
        models=models,
        memory=memory,
        cfg=eval_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device
    )
    eval_agent.load(model_path)
    # 进入评估模式：冻结 dropout/bn 等（若存在）
    for m in eval_agent.models.values():
        if hasattr(m, "eval"):
            m.eval()

    for ep in range(n_episodes):
        obs, info = env.reset()
        start_distance = float(info.get("distance", 0.0))
        if render:
            env.render()
        terminated = False
        truncated = False
        ep_reward = 0.0
        velocities = []
        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, log_std, _ = models["policy"].compute({"states": obs_tensor}, role="policy")
                if log_std.ndim < mean.ndim:
                    log_std = log_std.expand_as(mean)
                if deterministic:
                    action = torch.tanh(mean)
                else:
                    std = torch.exp(log_std)
                    action = torch.tanh(mean + std * torch.randn_like(mean))
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            if render:
                env.render()
            ep_reward += float(reward)
            if "velocity" in info:
                velocities.append(float(info["velocity"]))
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        end_distance = float(info.get("distance", 0.0))
        print(
            f"Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, "
            f"success={info.get('success', False)}, "
            f"avg_vel={avg_velocity:.2f}, "
            f"dist={start_distance:.1f}->{end_distance:.1f}"
        )
    env.close()
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Navigation3D with PPO (skrl)")
    parser.add_argument("mode", choices=["train", "eval", "smoke"], help="run mode")
    parser.add_argument("--model_path", type=str, default=None, help="path to model for eval")
    parser.add_argument("--resume_path", type=str, default=None, help="path to checkpoint for resume training")
    parser.add_argument("--experiment_name", type=str, default=None, help="experiment name for run directory")
    parser.add_argument("--num_envs", type=int, default=8, help="number of vector envs for training")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="total timesteps for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_episodes", type=int, default=10, help="episodes for eval")
    parser.add_argument("--no_render", action="store_true", help="disable rendering during eval")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic actions during eval")
    parser.add_argument("--fast", action="store_true", help="use faster training preset")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(
            num_envs=args.num_envs,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            mode="train",
            fast=args.fast,
            resume_path=args.resume_path,
            experiment_name=args.experiment_name,
        )
    elif args.mode == "smoke":
        # 快速冒烟：单环境、步数很小，用于验证代码路径/依赖
        train(num_envs=1, total_timesteps=2_000, seed=args.seed, mode="smoke",
              resume_path=args.resume_path, experiment_name=args.experiment_name)
    elif args.mode == "eval":
        if not args.model_path:
            raise ValueError("--model_path is required for eval mode")
        evaluate(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            deterministic=args.deterministic,
        )


if __name__ == "__main__":
    main()
