"""并没有实现完全的向量化，主要采用的是针对单个对象的计算方式，适用于当前环境中障碍物数量较少的情况。对于大量障碍物的情况，可以考虑进一步优化和向量化计算以提高性能。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def get_obb_axes(yaw: float, pitch: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 Agent 坐标系的坐标轴向量。"""

    yaw = float(yaw)
    pitch = float(pitch)

    x_axis = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch),], dtype=np.float32,)
    z_axis = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch),], dtype=np.float32,)
    y_axis = np.cross(z_axis, x_axis).astype(np.float32)

    return x_axis, y_axis, z_axis


def world_to_local(point: np.ndarray, position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """将世界坐标点（指障碍物球心）转换到 Agent 坐标系。"""

    # 障碍物球心
    point = np.asarray(point, dtype=np.float32)
    # Agent 位置
    position = np.asarray(position, dtype=np.float32)
    x_axis, y_axis, z_axis = get_obb_axes(yaw=yaw, pitch=pitch)

    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation.T @ (point - position)


def signed_distance_point_to_obb(point: np.ndarray, position: np.ndarray, yaw: float, pitch: float, half_extents: Tuple[float, float, float],) -> float:
    """
    点到定向长方体（OBB）的有符号距离：计算 Agent 的坐标系 -> 点在 Agent 坐标系下的坐标 -> 判断点在 OBB 内外并计算距离。

    Returns:
        - 正数：点在 OBB 外部，到表面的最短距离
        - 负数：点在 OBB 内部，返回负的“到最近表面距离”
    """

    hx, hy, hz = (float(half_extents[0]), float(half_extents[1]), float(half_extents[2]))
    # 障碍物球心在 Agent 坐标系下的坐标
    p_local = world_to_local(point=point, position=position, yaw=yaw, pitch=pitch)

    ax, ay, az = (abs(float(p_local[0])), abs(float(p_local[1])), abs(float(p_local[2])))

    inside = (ax <= hx) and (ay <= hy) and (az <= hz)
    if inside:
        dx = hx - ax
        dy = hy - ay
        dz = hz - az
        return -float(min(dx, dy, dz))

    dx = max(0.0, ax - hx)
    dy = max(0.0, ay - hy)
    dz = max(0.0, az - hz)
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))
