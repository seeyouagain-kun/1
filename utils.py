from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_angle(angle: float) -> float:
    """将角度截断到 [-π, π] 范围（用于 yaw）。"""

    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def compute_relative_angles(relative_vector: np.ndarray, yaw: float, pitch: float) -> Tuple[float, float]:
    """计算目标相对 Agent 朝向的偏差角。

    Returns:
        (relative_yaw, relative_pitch)
        - relative_yaw: [-π, π]
        - relative_pitch: [-π, π]（差值可能达到 π，不做 2π 环绕）
    """

    # 阈值待确认
    distance = float(np.linalg.norm(relative_vector))
    if distance < 1e-6:
        return 0.0, 0.0

    direction = relative_vector / distance
    target_yaw = float(np.arctan2(direction[1], direction[0]))
    target_pitch = float(np.arcsin(np.clip(direction[2], -1.0, 1.0)))

    relative_yaw = normalize_angle(target_yaw - float(yaw))
    relative_pitch = target_pitch - float(pitch)
    return relative_yaw, relative_pitch
