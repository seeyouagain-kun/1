"""
scripts/train_ppo.py — PPO 训练入口点
scripts/train_ppo.py — PPO training entry point

向后兼容的 PPO 训练入口，委托给 navigation3d.ppo.train_and_eval。
Backward-compatible PPO training entry point delegating to navigation3d.ppo.train_and_eval.

用法 / Usage:
    python scripts/train_ppo.py train [--num_envs N] [--total_timesteps T] [--seed S]
    python scripts/train_ppo.py eval  --model_path PATH [--n_episodes N]
    python scripts/train_ppo.py smoke
"""

import os
import sys

# 将项目根目录加入 sys.path，以便以任意工作目录运行
# Add project root to sys.path so the script can be run from any working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from navigation3d.ppo.train_and_eval import main

if __name__ == "__main__":
    main()
