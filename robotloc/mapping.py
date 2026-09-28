"""地图匹配接口占位。

本轮只融合轮速与惯性观测，地图匹配以后再加。这里定义好接口，
未来实现后可在 pipeline 中作为第三个观测源接入，无需改动 EKF 主干。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class MapMatcher(Protocol):
    """地图匹配器协议：输入位姿，返回地图约束观测。"""

    def match(
        self, t: float, pose_xy: np.ndarray, theta: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """返回 (约束量 z, 信息/协方差)，无匹配时返回 None。"""
        ...
