"""
使用 RL 算法进行三维空间中无人机（用长方体表示）点到点导航的训练
假设无人机能实现的基本运动为偏航、俯仰、沿其头部朝向前进，与磁控仿生鱼类似
假设偏航、俯仰、前进可以同时进行

三维空间大小：边长为 3000 的立方体
Agent 尺寸大小: (47,6,17) ,与 (4.7,0.6,0.17)mm 对应
障碍物设置: 3 * num_obstacles 个球形障碍物；障碍物半径为 obstacles_radius, 障碍物分布为：在起点和终点附近各生成 num_obstacles 个障碍物（集群），在整个空间内随机生成 num_obstacles 个障碍物（随机）
体积碰撞检测：将 Agent 看成一个长方体（OBB），计算长方体到碰撞物的距离，判断是否碰撞
雷达数据: 球形雷达，分辨率可调，默认为 10°，最大范围为 1000，输出为二维数组（elevation x azimuth）

状态空间: navigation state (11维) + 二维雷达
    1. yaw (当前偏航角)
    2. pitch (当前俯仰角)
    3. x position (x坐标)
    4. y position (y坐标)
    5. z position (z坐标)
    6. velocity (当前线速度)
    7. distance to goal (到目标的距离)
    8. relative yaw to goal (目标相对偏航角)
    9. relative pitch to goal (目标相对俯仰角)
    10. yaw rate (当前偏航角速度)
    11. pitch rate (当前俯仰角速度)
动作空间: yaw_rate, pitch_rate, linear_velocity(速度控制)
渲染：使用 VPython 进行可交互 3D 可视化
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from navigation3d.core import Obstacle
from navigation3d.utils import normalize_angle, compute_relative_angles
from navigation3d.sensors import LidarSensor, LIDAR_MAX_RANGE
from navigation3d.renderer import VPythonRenderer
from navigation3d.collision_obb import get_obb_axes

class Navigation3DEnv(gym.Env):
    """
    三维空间无人机点对点导航环境
    包含球形障碍物和多线雷达感知
    动作空间：[yaw_rate, pitch_rate, linear_velocity]（速度控制）

    支持任务参数化（用于 Meta-RL / 域随机化）。
    Supports task parameterization for Meta-RL and sim-to-real domain randomization.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        render_mode:  Optional[str] = None,
        reach_threshold: float = 2.0,       
        lidar_resolution: float = 10.0,
        
        # 在 0.5 和 2 之间随机缩放？缩放因子之间的关系？奖励的数值关系？
        bounds: float = 3000.0,
        agent_size: tuple = (47.0, 6.0, 17.0), 
        max_velocity: float = 150.0,
        max_angular_vel: float = 1.0,
        distance_reward_scale: float = 1.0,
        time_penalty: float = 0.5,
        num_obstacles: int = 10,
        obstacle_radius: float = 50.0,
        obstacle_collision_penalty: float = -1000.0,
        goal_reward: float = 1000.0,
        boundary_penalty: float = -1000.0,
        cluster_spread: float = 300.0,
        dt: float = 0.1,

        use_curriculum: bool = True,
        initial_curriculum:  float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()
        
        self.bounds = bounds
        
        # 保存基准尺寸，用于 set_task 中的缩放
        # Keep base size for scaling in set_task
        self.base_agent_size = {
            'length': float(agent_size[0]),
            'width': float(agent_size[1]),
            'height': float(agent_size[2])
        }
        self.agent_size = self.base_agent_size.copy()
        self.agent_scale = 1.0 # Default scale
        
        self.max_velocity = max_velocity
        self.max_angular_vel = max_angular_vel
        
        # 长方体参数
        self._update_agent_geometry()
        
        self.max_distance_observation = float(self.bounds * np.sqrt(3))
        
        self.reach_threshold = reach_threshold
        
        self.distance_reward_scale = distance_reward_scale
        self.time_penalty = time_penalty
        
        self.num_obstacles_base = int(num_obstacles)

        # 随课程学习难度的变化而变化
        self.num_obstacles = int(num_obstacles)
        
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.goal_reward = float(goal_reward)
        self.boundary_penalty = float(boundary_penalty)

        # Internal cluster params - controlled by main obstacle params
        self.cluster_obstacles_per_center = self.num_obstacles_base
        self.cluster_spread = float(cluster_spread)
        
        # 优化：不再存储 Obstacle 对象列表，直接使用 numpy 数组管理
        # Optimized: Use numpy arrays directly instead of list of objects
        self._obstacle_centers = np.empty((0, 3), dtype=np.float32)
        self._obstacle_radii = np.empty((0,), dtype=np.float32)
        
        self.lidar_resolution = lidar_resolution
        self.lidar = LidarSensor(resolution=lidar_resolution)
        self.lidar_data:  Optional[np.ndarray] = None
        
        self.use_curriculum = use_curriculum
        # 课程学习，难度逐渐提高，即目标距离 Agent 的距离越来越远
        self.curriculum_progress = initial_curriculum if use_curriculum else 1.0
        # 记录课程学习每个阶段的成功率，如果成功率超过某一阈值，就进入下一阶段
        self.success_history = []
        self.curriculum_window = 100
        self.curriculum_threshold = 0.8
        
        self.dt = float(dt)
        
        # 始终根据 bounds 动态计算 max_steps
        # Always calculate max_steps dynamically based on bounds
        step_dist_init = self.max_velocity * self.dt
        if step_dist_init > 1e-6:
             self.max_steps = int((self.max_distance_observation / step_dist_init) * 1.5)
        else:
             self.max_steps = 1000 # Fallback if physics invalid

        lidar_shape = self.lidar.get_data_shape()
        state_dim = 11
        
        # 强制使用归一化观测空间
        # Force normalized observation space
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.ones(state_dim, dtype=np.float32),
                high=np.ones(state_dim, dtype=np.float32),
                dtype=np.float32
            ),
            "lidar": spaces.Box(
                low=np.zeros(lidar_shape, dtype=np.float32),
                high=np.ones(lidar_shape, dtype=np.float32),
                dtype=np.float32
            )
        })
        
        self.action_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32
        )
        
        self.position = None
        self.yaw = None
        self.pitch = None
        self.velocity = None
        self.yaw_rate = None
        self.pitch_rate = None
        self.goal = None
        self.step_count = None
        self.trajectory = None
        self.prev_distance = None
        self.distance = None
        self.relative_yaw = None
        self.relative_pitch = None
        
        # 缓存当前的旋转矩阵
        self.rotation_matrix = None
        
        self.render_mode = render_mode
        self.renderer = None

        self.verbose = bool(verbose)

        # 缓存障碍物数组（每个 episode 固定），用于向量化计算与减少 Python 侧开销
        self._obstacle_centers = np.empty((0, 3), dtype=np.float32)
        self._obstacle_radii = np.empty((0,), dtype=np.float32)
        
        if self.verbose:
            self._print_config()

    def _update_agent_geometry(self):
        """Update derived geometry parameters from agent_size"""
        self.half_length = self.agent_size['length'] / 2
        self.half_width = self.agent_size['width'] / 2
        self.half_height = self.agent_size['height'] / 2
        
        # Conservative bounding radius
        self.bounding_radius = np.sqrt(
            self.half_length**2 + self.half_width**2 + self.half_height**2
        )

    # ==================== 任务参数化接口 (Meta-RL) ====================
    # Task parameterization interface for Meta-RL and domain randomization

    def get_task(self) -> Dict[str, Any]:
        """
        返回当前任务参数字典。
        Return the current task parameter dictionary.
        """
        return {
            "max_velocity": float(self.max_velocity),
            "max_angular_vel": float(self.max_angular_vel),
            "dt": float(self.dt),
            "distance_reward_scale": float(self.distance_reward_scale),
            "time_penalty": float(self.time_penalty),
            "obstacle_collision_penalty": float(self.obstacle_collision_penalty),
            "goal_reward": float(self.goal_reward),
            "boundary_penalty": float(self.boundary_penalty),
            "reach_threshold": float(self.reach_threshold),
            "num_obstacles": int(self.num_obstacles_base),
            "cluster_spread": float(self.cluster_spread),
            "obstacle_radius": float(self.obstacle_radius),
            "bounds": float(self.bounds),
            "agent_scale": float(self.agent_scale),
        }

    def set_task(self, task_dict: Dict[str, Any]) -> None:
        """
        从字典应用任务参数（向后兼容——未提供的键保持不变）。
        Apply task parameters from a dictionary. Keys not present are left unchanged.
        """
        if "max_velocity" in task_dict:
            self.max_velocity = float(task_dict["max_velocity"])
        if "max_angular_vel" in task_dict:
            self.max_angular_vel = float(task_dict["max_angular_vel"])
        if "dt" in task_dict:
            self.dt = float(task_dict["dt"])
        if "distance_reward_scale" in task_dict:
            self.distance_reward_scale = float(task_dict["distance_reward_scale"])
        if "time_penalty" in task_dict:
            self.time_penalty = float(task_dict["time_penalty"])
        if "obstacle_collision_penalty" in task_dict:
            self.obstacle_collision_penalty = float(task_dict["obstacle_collision_penalty"])
        if "goal_reward" in task_dict:
            self.goal_reward = float(task_dict["goal_reward"])
        if "boundary_penalty" in task_dict:
            self.boundary_penalty = float(task_dict["boundary_penalty"])
        if "reach_threshold" in task_dict:
            self.reach_threshold = float(task_dict["reach_threshold"])
        if "num_obstacles" in task_dict:
            n = int(task_dict["num_obstacles"])
            self.num_obstacles_base = n
            self.num_obstacles = n
            self.cluster_obstacles_per_center = n
        if "cluster_spread" in task_dict:
            self.cluster_spread = float(task_dict["cluster_spread"])
        if "obstacle_radius" in task_dict:
            self.obstacle_radius = float(task_dict["obstacle_radius"])
        if "bounds" in task_dict:
            self.bounds = float(task_dict["bounds"])
            self.max_distance_observation = float(self.bounds * np.sqrt(3))
        
        if "agent_scale" in task_dict:
            self.agent_scale = float(task_dict["agent_scale"])
            self.agent_size = {
                k: v * self.agent_scale for k, v in self.base_agent_size.items()
            }
            self._update_agent_geometry()

        # [Meta-RL Update] 根据物理参数动态调整 max_steps 
        # Dynamically adjust max_steps based on physics parameters
        # 逻辑: 对角线距离 / (最大速度 * dt) * 1.5 (安全余量)
        if any(k in task_dict for k in ["bounds", "max_velocity", "dt"]):
            step_dist = self.max_velocity * self.dt
            if step_dist > 1e-6:
                self.max_steps = int((self.max_distance_observation / step_dist) * 1.5)

    # ==================== 配置打印 ====================

    def _print_config(self):
        """打印环境配置"""
        print("=" * 60)
        print("Navigation3DEnv Configuration (Velocity Control)")
        print("=" * 60)
        print(f"  bounds: {self.bounds}")
        print(f"  max_steps: {self.max_steps}")
        print(f"  max_velocity: {self.max_velocity}")
        print(f"  max_angular_vel: {self.max_angular_vel}")
        print(f"  reach_threshold: {self.reach_threshold}")
        print(f"  max_distance:  {self.max_distance_observation:.1f}")
        print(f"  use_curriculum: {self.use_curriculum}")
        if self.use_curriculum:
            print(f"  initial_curriculum: {self.curriculum_progress}")
        print("-" * 60)
        print("Action Space:  [yaw_rate, pitch_rate, linear_velocity]")
        print(f"  yaw_rate range: [-{self.max_angular_vel}, {self.max_angular_vel}] rad/s")
        print(f"  pitch_rate range:  [-{self.max_angular_vel}, {self.max_angular_vel}] rad/s")
        print(f"  velocity range: [0, {self.max_velocity}]")
        print("-" * 60)
        print("Obstacle Configuration:")
        print(f"  layout: clustered(fixed) + random")
        total_obs = self.num_obstacles_base + self.cluster_obstacles_per_center * 2
        print(f"  num_obstacles (total): {total_obs} = {self.num_obstacles_base} (random) + {self.cluster_obstacles_per_center}x2 (clusters)")
        print(f"  cluster_spread: {self.cluster_spread}")
        print(f"  obstacle_radius: {self.obstacle_radius}")
        print(f"  collision_penalty: {self.obstacle_collision_penalty}")
        print(f"  goal_reward: {self.goal_reward}")
        print(f"  boundary_penalty: {self.boundary_penalty}")
        print("-" * 60)
        print("Lidar Configuration:")
        print(f"  resolution: {self.lidar_resolution} deg")
        print(f"  max_range: {LIDAR_MAX_RANGE} (fixed)")
        print(f"  data_shape: {self.lidar.get_data_shape()}")
        print("-" * 60)
        
        max_travel_per_step = self.max_velocity * self.dt
        max_travel_total = max_travel_per_step * self.max_steps
        print(f"  max travel per step: {max_travel_per_step:.1f}")
        print(f"  max total travel: {max_travel_total:.1f}")
        print(f"  max target distance: {self.max_distance_observation:.1f}")
        
        if max_travel_total < self.max_distance_observation:
            print(f"  Warning: may not reach farthest target!")
        else:
            print(f"  Config OK, can reach all targets")
        print("=" * 60)
    
    def _generate_obstacles(self):
        """生成随机球形障碍物 (Vectorized)"""
        # 初始化为空数组
        self._obstacle_centers = np.empty((0, 3), dtype=np.float32)
        self._obstacle_radii = np.empty((0,), dtype=np.float32)
        
        # 始终生成集群障碍物
        self._generate_clustered_obstacles()
        # 始终生成随机障碍物
        self._generate_random_obstacles()

    def _filter_and_add_candidates(self, candidates: np.ndarray, max_add: int, min_safe_dist: float) -> int:
        """
        过滤候选点并添加到障碍物列表。
        Filters candidates and adds valid ones to obstacle storage.
        """
        radius = self.obstacle_radius
        
        # 1. Check Agent & Goal safety distance
        d_agent = np.linalg.norm(candidates - self.position, axis=1)
        d_goal = np.linalg.norm(candidates - self.goal, axis=1)
        
        valid_mask = (d_agent > min_safe_dist) & (d_goal > min_safe_dist)
        candidates = candidates[valid_mask]
        
        if candidates.shape[0] == 0:
            return 0
        
        # 2. Check against EXISTING obstacles (Vectorized)
        if self._obstacle_centers.shape[0] > 0:
             dists = np.linalg.norm(candidates[:, None, :] - self._obstacle_centers[None, :, :], axis=2)
             overlap = dists < (radius + self._obstacle_radii[None, :])
             no_overlap = ~np.any(overlap, axis=1)
             candidates = candidates[no_overlap]
        
        if candidates.shape[0] == 0:
            return 0
        
        # 3. Check Internal Collisions (Greedy)
        kept_candidates = []
        for c in candidates:
            if len(kept_candidates) >= max_add:
                break
            
            overlap_internal = False
            for kept in kept_candidates:
                if np.linalg.norm(c - kept) < (radius * 2):
                     overlap_internal = True
                     break
            if not overlap_internal:
                kept_candidates.append(c)
        
        count = len(kept_candidates)
        if count > 0:
            new_pts = np.array(kept_candidates, dtype=np.float32)
            new_radii = np.full((count,), radius, dtype=np.float32)
            
            self._obstacle_centers = np.concatenate([self._obstacle_centers, new_pts], axis=0)
            self._obstacle_radii = np.concatenate([self._obstacle_radii, new_radii], axis=0)
            
        return count

    def _generate_clustered_obstacles(self):
        """在起点和终点周围紧凑生成障碍物 (Vectorized)"""
        centers_to_cluster = [self.position, self.goal]
        
        # Determine number of obstacles based on curriculum
        if self.use_curriculum:
            target_count = int(round(self.cluster_obstacles_per_center * self.curriculum_progress))
            target_count = max(0, min(self.cluster_obstacles_per_center, target_count))
        else:
            target_count = self.cluster_obstacles_per_center
            
        if target_count <= 0:
            return

        min_safe = self.obstacle_radius + self.bounding_radius + 0.1

        for cluster_center in centers_to_cluster:
            # 尝试次数
            attempts = 0
            max_attempts_batch_iters = 20
            generated_so_far = 0
            
            while generated_so_far < target_count and attempts < max_attempts_batch_iters:
                attempts += 1
                needed = target_count - generated_so_far
                
                # 生成一批候选点 (Batch Generation)
                batch_size = needed * 5 + 10 
                
                offsets = self.np_random.uniform(
                    low=-self.cluster_spread, high=self.cluster_spread, size=(batch_size, 3)
                ).astype(np.float32)
                
                candidates = cluster_center + offsets
                
                added = self._filter_and_add_candidates(candidates, needed, min_safe)
                generated_so_far += added

    def _generate_random_obstacles(self):
        """生成全局随机分布的障碍物 (Vectorized)"""
        limit = (self.bounds / 2) # 直接使用边界一半作为球心生成的极限
        
        # 计算需要额外添加的随机障碍物数量
        if self.use_curriculum:
            desired_random_count = int(round(self.num_obstacles_base * self.curriculum_progress))
            desired_random_count = max(0, min(self.num_obstacles_base, desired_random_count))
        else:
            desired_random_count = self.num_obstacles_base

        current_total = self._obstacle_centers.shape[0]
        target_total_obstacles = current_total + desired_random_count
        self.num_obstacles = target_total_obstacles
        
        min_safe = self.obstacle_radius + self.bounding_radius + 0.1
        attempts = 0
        
        while self._obstacle_centers.shape[0] < target_total_obstacles and attempts < 20:
             attempts += 1
             needed = target_total_obstacles - self._obstacle_centers.shape[0]
             batch_size = needed * 5 + 20
             
             candidates = self.np_random.uniform(
                low=-limit, high=limit, size=(batch_size, 3)
             ).astype(np.float32)
             
             self._filter_and_add_candidates(candidates, needed, min_safe)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境。
        Reset the environment.

        options 可选键 / Optional keys in options:
            "task" (dict): 在重置前调用 set_task(task)，用于 Meta-RL episode 切换。
                           If provided, set_task(task) is called before the episode starts.
        """
        super().reset(seed=seed)

        # 支持通过 options["task"] 在 reset 时切换任务
        # Support task switching at reset time via options["task"]
        if options is not None and "task" in options:
            self.set_task(options["task"])
        
        # 强制随机初始化位置 [Forced random start]
        # [Meta-RL Update] 扩大初始位置随机范围，以覆盖更多状态空间
        # Increase randomization range to cover more state space
        # 使用 `half_bounds - bounding_radius` 确保 Agent 整体都在边界内，且最大化利用空间
        # Use `half_bounds - bounding_radius` to ensure the full agent body is within bounds while maximizing space usage
        limit = (self.bounds / 2) - self.bounding_radius
        if limit < 0: limit = 0.0 # Safety check
        
        self.position = self.np_random.uniform(
            low=[-limit, -limit, -limit],
            high=[limit, limit, limit]
        ).astype(np.float32)
        
        self.yaw = self.np_random.uniform(-np.pi, np.pi)
        self.pitch = self.np_random.uniform(-np.pi/6, np.pi/6)
        
        # 初始速度和角速度设为0，以便更可控地开始任务
        self.velocity = 0.0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0
        
        self.goal = self._sample_goal()
        self._generate_obstacles()
        
        self.step_count = 0
        self.trajectory = [self.position.copy()]
        
        # 预先更新一次旋转矩阵（Lidar和碰撞检测都需要）
        self._update_rotation_matrix()
        
        self._update_relative_info()
        self.prev_distance = self.distance
        collision = self._compute_obstacle_metrics()

        self.lidar_data = self.lidar.scan_arrays(
            self.position, self.yaw, self.pitch, self._obstacle_centers, self._obstacle_radii,
            rotation_matrix=self.rotation_matrix
        )
        
        if self.render_mode == 'human' and self.renderer is not None:
            self.renderer.reset_trajectory()
            self.renderer.update_obstacles(self._obstacle_centers, self._obstacle_radii)
        
        obs = self._get_obs()
        info = self._get_info(collision=collision)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        actual_action = self._denormalize_action(action)
        self._update_state(actual_action)
        
        # 状态更新后，同步更新旋转矩阵
        self._update_rotation_matrix()

        self.lidar_data = self.lidar.scan_arrays(
            self.position, self.yaw, self.pitch, self._obstacle_centers, self._obstacle_radii,
            rotation_matrix=self.rotation_matrix
        )

        collision = self._compute_obstacle_metrics()

        obs = self._get_obs()
        reward = self._compute_reward(collision=collision)

        terminated = self._is_terminated(collision=collision)
        truncated = self.step_count >= self.max_steps

        info = self._get_info(collision=collision)
        
        self.step_count += 1
        self.trajectory.append(self.position.copy())
        self.prev_distance = self.distance
        
        if self.use_curriculum and (terminated or truncated):
            self._update_curriculum(info['success'])
        
        return obs, reward, terminated, truncated, info
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        将归一化动作转换为实际值
        action[0]:  yaw_rate, [-1, 1] -> [-max_angular_vel, max_angular_vel]
        action[1]:  pitch_rate, [-1, 1] -> [-max_angular_vel, max_angular_vel]
        action[2]: linear_velocity, [-1, 1] -> [0, max_velocity]
        """
        yaw_rate = action[0] * self.max_angular_vel
        pitch_rate = action[1] * self.max_angular_vel
        linear_velocity = (action[2] + 1.0) / 2.0 * self.max_velocity
        return np.array([yaw_rate, pitch_rate, linear_velocity], dtype=np.float32)
    
    def _sample_goal(self) -> np.ndarray:
        """采样目标点（支持课程学习）"""
        limit = (self.bounds / 2) - self.bounding_radius
        
        if self.use_curriculum:
            # 课程学习范围：从 (bounding_radius + reach_threshold) 逐渐扩展到 limit
            # 确保初始阶段仍然能采样到有效目标，同时基础范围更宽
            min_limit = self.bounding_radius + self.reach_threshold
            actual_limit = min_limit + (limit - min_limit) * self.curriculum_progress
            
            goal = self.np_random.uniform(
                low=[-actual_limit, -actual_limit, -actual_limit],
                high=[actual_limit, actual_limit, actual_limit]
            )
            return goal.astype(np.float32)
        
        goal = self.np_random.uniform(
            low=[-limit, -limit, -limit],
            high=[limit, limit, limit]
        )
        return goal.astype(np.float32)
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """获取观测值"""
        # 强制归一化 Forced normalization
        state = np.array([
            self.yaw / np.pi,
            self.pitch / (np.pi / 2),
            self.position[0] / (self.bounds / 2),
            self.position[1] / (self.bounds / 2),
            self.position[2] / (self.bounds / 2),
            #  速度归一化后在 [0, 1]，调整到 [-1, 1] 区间以匹配动作空间的对称性
            (self.velocity / self.max_velocity) * 2 - 1,
            (self.distance / self.max_distance_observation) * 2 - 1,
            self.relative_yaw / np.pi,
            self.relative_pitch / np.pi,
            self.yaw_rate / self.max_angular_vel,
            self.pitch_rate / self.max_angular_vel
        ], dtype=np.float32)
        state = np.clip(state, -1.0, 1.0)
        
        lidar = self.lidar_data / LIDAR_MAX_RANGE
        lidar = np.clip(lidar, 0.0, 1.0)
        
        return {"state": state, "lidar":  lidar}
    
    def _get_info(
        self,
        collision: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """获取额外信息"""
        is_close = self.distance < self.reach_threshold
        success = is_close
        if collision is None:
            collision = self._compute_obstacle_metrics()
        
        return {
            'distance': float(self.distance),
            'velocity': float(self.velocity),
            'position': self.position.copy(),
            'goal': self.goal.copy(),
            'step_count': self.step_count,
            'success': bool(success),
            'is_close': bool(is_close),
            'out_of_bounds': self._check_out_of_bounds(),
            'obstacle_collision': bool(collision),
            'relative_yaw': float(self.relative_yaw),
            'relative_pitch': float(self.relative_pitch),
            'curriculum_progress': float(self.curriculum_progress) if self.use_curriculum else 1.0,
            'num_obstacles': self._obstacle_centers.shape[0]
        }
    
    def _update_state(self, action: np.ndarray):
        """
        根据动作更新状态（速度控制模式）
        action:  [yaw_rate, pitch_rate, linear_velocity]
        """
        yaw_rate, pitch_rate, linear_velocity = action
        
        # 更新角速度记录
        self.yaw_rate = yaw_rate
        self.pitch_rate = pitch_rate
        
        self.yaw = normalize_angle(self.yaw + yaw_rate * self.dt)
        self.pitch = np.clip(self.pitch + pitch_rate * self.dt, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        self.velocity = np.clip(linear_velocity, 0.0, self.max_velocity)
        
        # 使用预先计算的旋转矩阵将速度变换到世界坐标系会更快，但这里需要的是更新前的矩阵，还是更新后的？
        # 更新位置是用更新后的姿态来算的，所以要在 self.yaw/pitch 更新后计算。
        # 简单起见，这里先保留原逻辑，或者用新的 self._update_rotation_matrix 算完再用。
        # 这里只有3个分量，直接算可能比调用矩阵乘法快（Python开销）。保持原样。
        vx = self.velocity * np.cos(self.pitch) * np.cos(self.yaw)
        vy = self.velocity * np.cos(self.pitch) * np.sin(self.yaw)
        vz = self.velocity * np.sin(self.pitch)
        
        velocity_vector = np.array([vx, vy, vz], dtype=np.float32)
        self.position = self.position + velocity_vector * self.dt
        
        self._update_relative_info()
    
    def _update_rotation_matrix(self):
        """更新机体旋转矩阵"""
        x_axis, y_axis, z_axis = get_obb_axes(yaw=self.yaw, pitch=self.pitch)
        self.rotation_matrix = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)

    def _update_relative_info(self):
        """更新相对距离和角度"""
        relative_vector = self.goal - self.position
        self.distance = float(np.linalg.norm(relative_vector))
        
        # 当距离小于判定阈值时，认为已经到达，相对角度直接置零
        if self.distance > self.reach_threshold:
            self.relative_yaw, self.relative_pitch = compute_relative_angles(
                relative_vector, self.yaw, self.pitch
            )
        else:
            self.relative_yaw = 0.0
            self.relative_pitch = 0.0

    # ---------- 修改障碍物交互方法 ----------
    def _get_min_obstacle_distance(self) -> float:
        """获取到最近障碍物表面的有符号距离（负表示穿透）"""
        if self._obstacle_centers.size == 0:
            return float("inf")

        # 使用缓存的旋转矩阵
        rotation = self.rotation_matrix

        diff = (self._obstacle_centers - self.position[None, :]).astype(np.float32)
        p_local = diff @ rotation

        hx, hy, hz = (float(self.half_length), float(self.half_width), float(self.half_height))
        ax = np.abs(p_local[:, 0]).astype(np.float32)
        ay = np.abs(p_local[:, 1]).astype(np.float32)
        az = np.abs(p_local[:, 2]).astype(np.float32)

        inside = (ax <= hx) & (ay <= hy) & (az <= hz)

        dist = np.empty((p_local.shape[0],), dtype=np.float32)
        if np.any(inside):
            dx_in = (hx - ax[inside]).astype(np.float32)
            dy_in = (hy - ay[inside]).astype(np.float32)
            dz_in = (hz - az[inside]).astype(np.float32)
            dist[inside] = -np.minimum(np.minimum(dx_in, dy_in), dz_in)

        if np.any(~inside):
            dx = np.maximum(0.0, ax[~inside] - hx).astype(np.float32)
            dy = np.maximum(0.0, ay[~inside] - hy).astype(np.float32)
            dz = np.maximum(0.0, az[~inside] - hz).astype(np.float32)
            dist[~inside] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

        surface_dist = dist - self._obstacle_radii
        return float(np.min(surface_dist))

    def _check_obstacle_collision(self) -> bool:
        """检查是否与障碍物碰撞（球心到长方体的有符号距离 ≤ 球半径）"""
        return self._compute_obstacle_metrics()

    def _compute_obstacle_metrics(self) -> bool:
        """一次性计算碰撞（向量化）。"""

        if self._obstacle_centers.size == 0:
            return False

        # 使用缓存的旋转矩阵
        rotation = self.rotation_matrix  # (3, 3)

        # 将所有障碍物球心转换到机体局部坐标系。
        # 旋转矩阵 'rotation' 的每一列代表机体坐标系的轴在世界坐标系下的表示 (x_axis, y_axis, z_axis)。
        # 这里的 diff 是世界坐标系下的相对位置向量 (N, 3)。
        # 为了转换到局部坐标系，我们需要将这些向量投影到局部坐标系的轴上。
        # 计算 diff @ rotation，结果的第 i 行第 j 列即为第 i 个向量在第 j 个局部轴上的投影分量。
        # 这等价于列向量形式的 p_local = R.T @ p_world。

        diff = (self._obstacle_centers - self.position[None, :]).astype(np.float32)
        p_local = diff @ rotation  # (N, 3)

        hx, hy, hz = (float(self.half_length), float(self.half_width), float(self.half_height))
        ax = np.abs(p_local[:, 0]).astype(np.float32)
        ay = np.abs(p_local[:, 1]).astype(np.float32)
        az = np.abs(p_local[:, 2]).astype(np.float32)

        inside = (ax <= hx) & (ay <= hy) & (az <= hz)

        # signed distance point->OBB
        dist = np.empty((p_local.shape[0],), dtype=np.float32)
        if np.any(inside):
            dx_in = (hx - ax[inside]).astype(np.float32)
            dy_in = (hy - ay[inside]).astype(np.float32)
            dz_in = (hz - az[inside]).astype(np.float32)
            dist[inside] = -np.minimum(np.minimum(dx_in, dy_in), dz_in)

        if np.any(~inside):
            dx = np.maximum(0.0, ax[~inside] - hx).astype(np.float32)
            dy = np.maximum(0.0, ay[~inside] - hy).astype(np.float32)
            dz = np.maximum(0.0, az[~inside] - hz).astype(np.float32)
            dist[~inside] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

        # sphere surface distance and collision
        # collision if dist <= radius (surface_dist <= 0)
        # Using np.any is generally faster than finding min then checking <= 0
        collision = bool(np.any(dist <= self._obstacle_radii))
        return collision
    
    def _compute_reward(self, collision: Optional[bool] = None) -> float:
        """计算奖励"""
        is_close = self.distance < self.reach_threshold
        if is_close: return float(self.goal_reward)
        if self._check_out_of_bounds(): return float(self.boundary_penalty)
        if collision is None: collision = self._compute_obstacle_metrics()
        if collision: return self.obstacle_collision_penalty
        
        reward = 0.0
        
        distance_change = self.prev_distance - self.distance
        max_change = self.max_velocity * self.dt
        normalized_distance_change = distance_change / max_change
        reward += normalized_distance_change * self.distance_reward_scale
        
        reward -= self.time_penalty
        
        return float(reward)
    
    def _is_terminated(self, collision: Optional[bool] = None) -> bool:
        """判断是否终止"""
        is_close = self.distance < self.reach_threshold
        success = is_close
        
        if collision is None: collision = self._compute_obstacle_metrics()
        
        return success or self._check_out_of_bounds() or collision
    
    def _check_out_of_bounds(self) -> bool:
        """检查是否出界"""
        return bool(np.any(np.abs(self.position) > self.bounds / 2))
    
    def _update_curriculum(self, success: bool):
        """更新课程学习进度"""
        self.success_history.append(success)
        
        if len(self.success_history) >= self.curriculum_window:
            recent_success_rate = sum(self.success_history[-self.curriculum_window:]) / self.curriculum_window
            
            if recent_success_rate > self.curriculum_threshold:
                old_progress = self.curriculum_progress
                self.curriculum_progress = min(1.0, self.curriculum_progress + 0.1)
                if self.curriculum_progress > old_progress:
                      print(f"[Curriculum] Progress:  {old_progress:.1%} -> {self.curriculum_progress:.1%} "
                          f"(success rate: {recent_success_rate:.1%})")
            
            if len(self.success_history) > self.curriculum_window * 2:
                self.success_history = self.success_history[-self.curriculum_window:]
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return None
        
        if self.renderer is None:
            self.renderer = VPythonRenderer(self.bounds, self.agent_size)
            self.renderer.initialize()
            if self.renderer.initialized:
                self.renderer.update_obstacles(self._obstacle_centers, self._obstacle_radii)
        
        if self.renderer.initialized:
            min_obs_dist = self._get_min_obstacle_distance()
            self.renderer.update(
                position=self.position,
                yaw=self.yaw,
                pitch=self.pitch,
                goal=self.goal,
                velocity=self.velocity,
                distance=self.distance,
                step_count=self.step_count,
                min_obstacle_dist=min_obs_dist,
                curriculum_progress=self.curriculum_progress
            )
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
