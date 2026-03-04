from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from navigation3d.collision_obb import get_obb_axes
from navigation3d.core import Obstacle


LIDAR_MAX_RANGE = 2000.0


class LidarSensor:
    """多线球形雷达"""

    def __init__(self, resolution: float = 10.0):
        self.resolution = float(resolution)
        # 角度转为弧度
        self.azimuth_resolution = float(np.radians(resolution))
        self.elevation_resolution = float(np.radians(resolution))
        self.max_range = float(LIDAR_MAX_RANGE)

        self.n_azimuth = int(360 / resolution)
        self.n_elevation = int(180 / resolution) + 1

        self.azimuths = np.linspace(-np.pi, np.pi, self.n_azimuth, endpoint=False)
        self.elevations = np.linspace(-np.pi / 2, np.pi / 2, self.n_elevation)

        # 预计算 Agent 坐标系下的单位方向向量：x 前、y 左、z 上
        az = self.azimuths[None, :]  # 经度 (1, A)
        el = self.elevations[:, None]  # 纬度 (E, 1)
        cos_el = np.cos(el)
        local_dirs = np.stack([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el) * np.ones_like(az), ], axis=-1, ).astype(np.float32)  # (E, A, 3)

        self._local_dirs_flat = local_dirs.reshape(-1, 3)  # (R, 3)

    def scan_arrays(self, position: np.ndarray, yaw: float, pitch: float, centers: np.ndarray, radii: np.ndarray, rotation_matrix: Optional[np.ndarray] = None, ) -> np.ndarray:
        """向量化雷达扫描（射线方向随 yaw/pitch 转动）。"""

        position = np.asarray(position, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)

        if centers.size == 0:
            return np.full((self.n_elevation, self.n_azimuth), self.max_range, dtype=np.float32)

        if rotation_matrix is None:
            # 由 yaw/pitch 得到机体系在世界系下的轴（单位向量）
            x_axis, y_axis, z_axis = get_obb_axes(yaw=float(yaw), pitch=float(pitch))
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)  # (3, 3)

        # local_dir -> world_dir: world_dir = local_dir @ rotation.T
        # 每条射线的方向向量在世界坐标系下的表示，shape (R, 3)
        world_dirs_flat = self._local_dirs_flat @ rotation_matrix.T  # (R, 3)
        world_dirs_flat_T = world_dirs_flat.T  # (3, R)

        # oc = ray_origin - sphere_center, shape (n, 3)
        oc = position[None, :] - centers # (1, 3) - (n, 3) -> (n, 3)
        c = np.einsum("ij,ij->i", oc, oc) - radii * radii  # (n,)

        # b = 2 * dot(oc, d), shape (O, R)
        b = 2.0 * (oc @ world_dirs_flat_T)

        # discriminant = b^2 - 4ac, with a=1 (rotation preserves unit length)
        disc = b * b - 4.0 * c[:, None]
        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

        t1 = (-b - sqrt_disc) * 0.5
        t2 = (-b + sqrt_disc) * 0.5

        inf = np.float32(np.inf)
        t = np.where(t1 > 0.0, t1, np.where(t2 > 0.0, t2, inf))
        t = np.where(valid, t, inf)

        min_t = np.min(t, axis=0)  # (R,)
        min_t = np.minimum(min_t, self.max_range).astype(np.float32)
        return min_t.reshape(self.n_elevation, self.n_azimuth)

    def scan(self, position: np.ndarray, yaw: float, pitch: float, obstacles: List[Obstacle]) -> np.ndarray:
        """list[Obstacle] 版本的扫描包装。"""

        if not obstacles:
            return np.full((self.n_elevation, self.n_azimuth), self.max_range, dtype=np.float32)

        centers = np.stack([o.center for o in obstacles], axis=0).astype(np.float32)
        radii = np.asarray([o.radius for o in obstacles], dtype=np.float32)
        return self.scan_arrays(position=position, yaw=yaw, pitch=pitch, centers=centers, radii=radii)

    def get_data_shape(self) -> Tuple[int, int]:
        return (self.n_elevation, self.n_azimuth)
