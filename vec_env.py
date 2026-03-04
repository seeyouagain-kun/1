"""
Vectorized environment wrapper for PEARL Meta-RL.
Supports multiprocessing for parallel data collection.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

# Try to use cloudpickle for better function serialization
try:
    import cloudpickle
    _pickle = cloudpickle
except ImportError:
    import pickle
    _pickle = pickle


class CloudpickleWrapper:
    """Wrapper to make lambda functions picklable."""
    def __init__(self, x: Any):
        self.x = x

    def __getstate__(self):
        return _pickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = _pickle.loads(ob)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """
    Worker process for running an environment.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                # data is action
                action = data
                obs_dict, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated
                if done:
                    # Auto-reset if done, to prevent worker from blocking/waiting
                    # We save the terminal observation in info
                    info["terminal_observation"] = obs_dict
                    # Reset with new_task=False by default for PEARL rollout within same task
                    # If we need to change task, we should have called set_task before.
                    obs_dict, reset_info = env.reset(new_task=False)
                    # Merge reset_info into info if needed, or just ignore for now
                
                remote.send((obs_dict, reward, done, info))

            elif cmd == "reset":
                # data is kwargs for reset
                kwargs = data or {}
                obs_dict, info = env.reset(**kwargs)
                remote.send((obs_dict, info))

            elif cmd == "set_task":
                # data is task dict
                env.set_task(data)
                remote.send(None)

            elif cmd == "sample_task":
                task = env.sample_task()
                remote.send(task)

            elif cmd == "close":
                env.close()
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        print("Worker KeyboardInterrupt")
    except Exception as e:
        # Send error object back to parent
        remote.send(e)
    finally:
        env.close()


class MetaVectorEnv:
    """
    A vectorized wrapper for Meta-RL environments.
    Runs multiple environments in separate processes.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        Args:
            env_fns: List of callables that create environments.
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        ctx = mp.get_context("spawn") # Force spawn for compatibility (Windows/Linux)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Assuming all envs have same space, we can query one if needed?
        # But we don't strictly need spaces if we just pass through data.

    def step_async(self, actions: np.ndarray) -> None:
        """Send step command to all workers."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """Wait for Step results."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        # Unpack results: (obs_dict, reward, done, info)
        obs_list, rews, dones, infos = zip(*results)
        
        return (
            obs_list, # List of Dict
            np.stack(rews),    # Array
            np.stack(dones),   # Array
            infos     # List of Dict
        )

    def step(self, actions: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """Synchronous step."""
        self.step_async(actions)
        return self.step_wait()

    def reset(self, **kwargs) -> Tuple[List[Dict], List[Dict]]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(("reset", kwargs))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list, infos = zip(*results)
        return obs_list, infos

    def set_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Set task for each environment."""
        assert len(tasks) == self.num_envs
        for remote, task in zip(self.remotes, tasks):
            remote.send(("set_task", task))
        
        # Wait for confirmation
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, Exception):
                raise res

    def sample_task(self) -> Dict[str, Any]:
        """Sample a task from the first environment."""
        self.remotes[0].send(("sample_task", None))
        return self.remotes[0].recv()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True
