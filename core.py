from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
    """球形障碍物"""

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32)
        self.radius = float(self.radius)
