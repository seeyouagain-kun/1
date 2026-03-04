from __future__ import annotations

from typing import List

import numpy as np

from navigation3d.core import Obstacle


class VPythonRenderer:
    """VPython 渲染器（可交互 3D 可视化）。"""

    def __init__(self, bounds: float, agent_size: dict):
        self.bounds = float(bounds)
        self.agent_size = agent_size
        self.initialized = False

        self.vp = None
        self.canvas = None
        self.agent_box = None
        self.agent_arrow = None
        self.goal_sphere = None
        self.obstacle_spheres = []
        self.trajectory_curve = None
        self.info_label = None

    def initialize(self) -> None:
        """初始化 VPython 场景。"""

        if self.initialized:
            return

        try:
            import vpython as vp

            self.vp = vp
        except ImportError:
            print("Warning: vpython not installed. Rendering disabled.")
            return

        vp = self.vp

        # 基本画面设置
        self.canvas = vp.canvas(
            title="3D Navigation with Obstacles",
            width=1200,
            height=800,
            center=vp.vector(0, 0, 0),
            background=vp.color.gray(0.2),
            userpan=True,
            userzoom=True,
            userspin=True,
        )

        # 允许用户交互 (平移、缩放、旋转)
        self.canvas.userpan = True
        self.canvas.userzoom = True
        self.canvas.userspin = True
        
        # 禁用自动缩放，防止物体移动时视角乱跳
        self.canvas.autoscale = False

        self.canvas.camera.follow = None # 确保没有自动跟随目标，允许自由视角

        self.canvas.camera.pos = vp.vector(self.bounds * 0.8, self.bounds * 0.8, self.bounds * 0.8)
        self.canvas.camera.axis = vp.vector(-self.bounds * 0.8, -self.bounds * 0.8, -self.bounds * 0.8)

        self._draw_bounds()

        # Agent 长方体与箭头 
        self.agent_box = vp.box(
            pos=vp.vector(0, 0, 0),
            size=vp.vector(self.agent_size["length"], self.agent_size["width"], self.agent_size["height"]),
            color=vp.color.cyan,
            opacity=0.8,
        )

        self.agent_arrow = vp.arrow(
            pos=vp.vector(0, 0, 0),
            axis=vp.vector(100, 0, 0),
            color=vp.color.green,
            shaftwidth=10,
        )

        # 目标球体
        self.goal_sphere = vp.sphere(
            pos=vp.vector(0, 0, 0),
            radius=80,
            color=vp.color.red,
            emissive=True,
        )

        # 轨迹曲线
        self.trajectory_curve = vp.curve(color=vp.color.yellow, radius=5)

        self.info_label = vp.label(
            pos=vp.vector(0, self.bounds * 0.4, 0),
            text="",
            height=16,
            color=vp.color.white,
            background=vp.color.black,
            opacity=0.7,
            box=True,
        )

        self.initialized = True

    def _draw_bounds(self) -> None:
        """绘制边界线框。"""

        vp = self.vp
        b = self.bounds / 2
        
        # 使用 points 方式一次性绘制，减少对象数量
        # 这种连线方式不会画出所有边，但视觉上足够看出边界框
        points = [
            (-b, -b, -b), (b, -b, -b), (b, b, -b), (-b, b, -b), (-b, -b, -b), # 底面
            (-b, -b, b), (b, -b, b), (b, b, b), (-b, b, b), (-b, -b, b),      # 顶面
            (-b, b, b), (-b, b, -b),                                          # 立柱
            (b, b, -b), (b, b, b),                                            # 立柱
            (b, -b, b), (b, -b, -b)                                           # 立柱
        ]
        
        vp.curve(pos=[vp.vector(*p) for p in points], color=vp.color.white, radius=3)

    def update_obstacles(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """更新障碍物显示 (Vectorized Input)。"""

        if not self.initialized:
            return

        vp = self.vp

        for sphere in self.obstacle_spheres:
            sphere.visible = False
            # remove from scene if possible or just hide
            sphere.delete()
        self.obstacle_spheres = []

        # centers shape: (N, 3), radii shape: (N,)
        for i in range(centers.shape[0]):
            pos = centers[i]
            r = radii[i]
            sphere = vp.sphere(
                pos=vp.vector(float(pos[0]), float(pos[1]), float(pos[2])),
                radius=float(r),
                color=vp.color.orange,
                opacity=0.4,
            )
            self.obstacle_spheres.append(sphere)

    def update(
        self,
        position: np.ndarray,
        yaw: float,
        pitch: float,
        goal: np.ndarray,
        velocity: float,
        distance: float,
        step_count: int,
        min_obstacle_dist: float,
        curriculum_progress: float,
    ) -> None:
        """更新渲染。"""

        if not self.initialized:
            return

        vp = self.vp

        self.agent_box.pos = vp.vector(float(position[0]), float(position[1]), float(position[2]))

        direction = vp.vector(
            float(np.cos(pitch) * np.cos(yaw)),
            float(np.cos(pitch) * np.sin(yaw)),
            float(np.sin(pitch)),
        )

        self.agent_box.axis = direction * float(self.agent_size["length"])
        self.agent_box.up = vp.vector(
            float(-np.sin(pitch) * np.cos(yaw)),
            float(-np.sin(pitch) * np.sin(yaw)),
            float(np.cos(pitch)),
        )

        self.agent_arrow.pos = self.agent_box.pos
        self.agent_arrow.axis = direction * 150

        self.goal_sphere.pos = vp.vector(float(goal[0]), float(goal[1]), float(goal[2]))
        self.trajectory_curve.append(self.agent_box.pos)

        self.info_label.text = (
            f"Step: {step_count} | Distance: {distance:.1f} | Velocity: {velocity:.1f}\n"
            f"Min Obstacle Dist: {min_obstacle_dist:.1f} | Curriculum: {curriculum_progress:.1%}"
        )

        vp.rate(60)

    def reset_trajectory(self) -> None:
        """重置轨迹。"""

        if self.initialized and self.trajectory_curve:
            self.trajectory_curve.clear()

    def close(self) -> None:
        """关闭渲染器。"""

        if self.initialized and self.canvas:
            self.canvas.delete()
            self.initialized = False
