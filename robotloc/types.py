"""系统中流转的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class WheelObservation:
    """虚构轮速观测：单轮编码器读数与本次读数的时间间隔。

    Parameters
    ----------
    t:
        观测对应的测量时刻（秒，单调时间基）。
    received_at:
        消息到达估计器的时刻；``received_at > t`` 即表示迟到。
    left_counts, right_counts:
        左右轮累计计数（tick，可为浮点）。
    dt_counts:
        距上一次读数的时间间隔（秒）。
    wheel_radius, wheel_base:
        该消息时刻的运动学参数（米 / 米）。
    counts_per_revolution:
        每圈计数数。
    seq:
        发送端序列号，用于区分同时刻消息与断流。
    """

    t: float
    received_at: float
    left_counts: float
    right_counts: float
    dt_counts: float
    wheel_radius: float = 0.1
    wheel_base: float = 0.5
    counts_per_revolution: float = 1024.0
    seq: int = 0

@dataclass(frozen=True, slots=True)
class ImuObservation:
    """虚构惯性观测：偏航角速率 + 航向绝对参考。

    yaw_rate 来自陀螺（持续可用）；yaw 为低频、带漂移的航向参考，
    无法提供航向时传 ``None``。
    """

    t: float
    received_at: float
    yaw_rate: float
    yaw: float | None = None
    seq: int = 0


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """某时刻的定位输出。

    covariance 为 5x5（x, y, theta, v, omega），其中不确定性还会在
    断流期间随时间增长，调用方可据此判断当前精度是否可信。
    ``fresh`` 表示该输出是否由实际观测更新（False = 纯推演）。
    """

    t: float
    x: float
    y: float
    theta: float
    covariance: np.ndarray = field(compare=False)
    fresh: bool = True

    @property
    def position_sigma(self) -> float:
        """位置综合 1σ（米）。"""
        return float(np.sqrt(self.covariance[0, 0] + self.covariance[1, 1]))

    @property
    def heading_sigma(self) -> float:
        """航向 1σ（弧度）。"""
        return float(np.sqrt(self.covariance[2, 2]))

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)
