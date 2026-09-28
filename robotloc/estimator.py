"""状态估计核心：轮速 + 惯性的误差状态 EKF（5 维状态）。

状态 ``[x, y, theta, v, omega]``：平面位置、航向、本体线速度、偏航角速率。
所有更新只允许使用时刻 ``t`` 的观测；调用方（pipeline）负责因果性，
本模块不保存任何“未来”缓冲，因此可以安全地从检查点回滚重放。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import ImuObservation, PoseEstimate, WheelObservation


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EkfConfig:
    """过程噪声与观测噪声参数（均为连续/物理量纲）。"""

    # 速度随机游走 (m/s^2 / sqrt(s)) 与角速率随机游走 (rad/s^2 / sqrt(s))
    q_v: float = 0.35
    q_omega: float = 0.8
    # 陀螺角速率观测噪声 std (rad/s)
    r_gyro: float = 0.01
    # 轮速线速度观测噪声 std (m/s)；打滑时被自适应放大
    r_wheel: float = 0.05
    # 绝对航向参考噪声 std (rad)
    r_yaw: float = 0.08
    # 卡方门限（1 维，约 95%），超过则认为观测异常（打滑/异常陀螺）
    nis_gate: float = 3.84
    # 超过该 NIS（1 维，约 99.99%）的轮速观测视为打滑，直接拒绝
    nis_reject: float = 15.0
    # 异常观测时 R 的最大放大倍数
    r_inflation_max: float = 200.0


@dataclass
class _WheelCarry:
    """上一条轮速读数，用于把累计计数转换成区间平均速度。"""

    t: float
    left_counts: float
    right_counts: float


@dataclass
class EkfLocalizer:
    """误差状态 EKF 定位器。

    用法::

        loc = EkfLocalizer(initial_pose=(0, 0, 0))
        loc.predict(t)
        loc.update_wheel(msg)   # 观测时刻必须 >= 当前时间
        out = loc.estimate(t, fresh=True)
    """

    config: EkfConfig = field(default_factory=EkfConfig)
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_heading_std: float = 0.01
    initial_velocity_std: float = 0.1

    def __post_init__(self) -> None:
        x0, y0, th0 = self.initial_pose
        self._x = np.array([x0, y0, _wrap(th0), 0.0, 0.0], dtype=float)
        self._P = np.diag(
            [1e-8, 1e-8, self.initial_heading_std**2,
             self.initial_velocity_std**2, self.initial_velocity_std**2]
        )
        self._t: float | None = None
        self._last_wheel: _WheelCarry | None = None
        self.last_update_kind: str = "init"

    # ------------------------------------------------------------------
    # 纯推演（dead reckoning）。断流时由 pipeline 反复调用，
    # 协方差按过程噪声持续增长，因此不会“声称精度不变”。
    # ------------------------------------------------------------------
    def predict(self, t: float) -> None:
        if self._t is None:
            self._t = t
            return
        dt = t - self._t
        if dt < -1e-9:
            raise ValueError(
                f"观测时间倒流: {t} < {self._t}；迟到观测应由 pipeline 回滚处理"
            )
        if dt <= 0.0:
            return

        _, _, theta, v, omega = self._x
        # 中点航向积分，降低转弯时的一阶离散误差
        theta_mid = theta + 0.5 * omega * dt
        F = np.eye(5)
        F[0, 2] = -v * math.sin(theta_mid) * dt
        F[0, 3] = math.cos(theta_mid) * dt
        F[0, 4] = -0.5 * v * math.sin(theta_mid) * dt * dt
        F[1, 2] = v * math.cos(theta_mid) * dt
        F[1, 3] = math.sin(theta_mid) * dt
        F[1, 4] = 0.5 * v * math.cos(theta_mid) * dt * dt
        F[2, 4] = dt

        self._x[0] += v * math.cos(theta_mid) * dt
        self._x[1] += v * math.sin(theta_mid) * dt
        self._x[2] = _wrap(self._x[2] + omega * dt)

        # 过程噪声：v 与 omega 为随机游走；位置/航向噪声通过雅可比传播
        Q = np.zeros((5, 5))
        Q[3, 3] = self.config.q_v**2 * dt
        Q[4, 4] = self.config.q_omega**2 * dt
        # 额外的位置直接扩散项：体现里程计/地面接触的未建模误差
        Q[0, 0] = (0.02 * max(abs(v), 0.2) * dt) ** 2
        Q[1, 1] = Q[0, 0]
        self._P = F @ self._P @ F.T + Q
        self._t = t

    # ------------------------------------------------------------------
    # 观测更新
    # ------------------------------------------------------------------
    def update_wheel(self, msg: WheelObservation) -> bool:
        """轮速消息：由累计计数得到区间平均 v/omega 并更新。

        返回是否接受该观测；打滑导致残差超门限时会自适应放大噪声
        （降权而非丢弃），严重异常才拒绝。
        """
        self.predict(msg.t)
        accepted = True
        if self._last_wheel is not None and msg.t > self._last_wheel.t:
            prev = self._last_wheel
            dt = msg.t - prev.t
            k = 2.0 * math.pi / msg.counts_per_revolution
            v_l = (msg.left_counts - prev.left_counts) * k * msg.wheel_radius / dt
            v_r = (msg.right_counts - prev.right_counts) * k * msg.wheel_radius / dt
            v_meas = 0.5 * (v_l + v_r)
            w_meas = (v_r - v_l) / msg.wheel_base

            nis_v, reject_v = self._scalar_update(
                3, v_meas, self.config.r_wheel,
                allow_inflate=True, allow_reject=True,
            )
            # 打滑时差速角速率极不可信，放大其观测噪声
            nis_w, reject_w = self._scalar_update(
                4, w_meas, self.config.r_wheel / msg.wheel_base,
                allow_inflate=True, allow_reject=True,
            )
            self.last_update_kind = "wheel_rejected" if (reject_v or reject_w) else "wheel"
        else:
            # 第一条读数：测量值是 [t-dt, t] 的区间平均速度，
            # 直接把该段位移并入状态，避免启动时丢半个区间。
            if self._last_wheel is None and msg.dt_counts > 0.0:
                k = 2.0 * math.pi / msg.counts_per_revolution
                v_l = msg.left_counts * k * msg.wheel_radius / msg.dt_counts
                v_r = msg.right_counts * k * msg.wheel_radius / msg.dt_counts
                v0 = 0.5 * (v_l + v_r)
                w0 = (v_r - v_l) / msg.wheel_base
                dt0 = msg.dt_counts
                theta_mid = self._x[2] + 0.5 * w0 * dt0
                self._x[0] += v0 * math.cos(theta_mid) * dt0
                self._x[1] += v0 * math.sin(theta_mid) * dt0
                self._x[2] = _wrap(self._x[2] + w0 * dt0)
                self._x[3] = v0
                self._x[4] = w0
                self.last_update_kind = "wheel_init"
            else:
                accepted = False
        self._last_wheel = _WheelCarry(msg.t, msg.left_counts, msg.right_counts)
        return accepted

    def update_imu(self, msg: ImuObservation) -> bool:
        """惯性消息：陀螺更新 omega；若有绝对航向参考再更新 theta。"""
        self.predict(msg.t)
        self._scalar_update(4, msg.yaw_rate, self.config.r_gyro, allow_inflate=True)
        if msg.yaw is not None:
            # 残差做角度包裹，避免 ±π 附近误判
            innov = _wrap(msg.yaw - self._x[2])
            H = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            S = float(H @ self._P @ H) + self.config.r_yaw**2
            nis = innov * innov / S
            if nis <= self.config.nis_gate:
                K = self._P @ H / S
                self._x += K * innov
                self._x[2] = _wrap(self._x[2])
                self._P -= np.outer(K, S * K)
                self._P = 0.5 * (self._P + self._P.T)
        self.last_update_kind = "imu"
        return True

    def _scalar_update(
        self, index: int, z: float, r_std: float, *,
        allow_inflate: bool, allow_reject: bool = False,
    ) -> tuple[float, bool]:
        H = np.zeros(5)
        H[index] = 1.0
        innov = z - self._x[index]
        S0 = float(H @ self._P @ H) + r_std * r_std
        r_var = r_std * r_std
        nis = innov * innov / S0
        rejected = False
        if allow_reject and nis > self.config.nis_reject:
            return nis, True
        if allow_inflate and nis > self.config.nis_gate:
            # 自适应：按残差需要的方差放大 R（封顶），等价于给打滑观测降权
            needed = innov * innov / self.config.nis_gate - float(H @ self._P @ H)
            r_var = min(max(r_var, needed),
                        r_var * self.config.r_inflation_max)
        S = float(H @ self._P @ H) + r_var
        K = self._P @ H / S
        self._x += K * innov
        self._P -= np.outer(K, S * K)
        self._P = 0.5 * (self._P + self._P.T)
        return nis, rejected

    # ------------------------------------------------------------------
    # 输出与检查点
    # ------------------------------------------------------------------
    def estimate(self, t: float | None = None, *, fresh: bool = True) -> PoseEstimate:
        if t is not None:
            self.predict(t)
        assert self._t is not None
        return PoseEstimate(
            t=self._t,
            x=float(self._x[0]),
            y=float(self._x[1]),
            theta=float(self._x[2]),
            covariance=self._P.copy(),
            fresh=fresh,
        )

    def checkpoint(self) -> dict:
        """序列化为纯内置类型（JSON 友好），恢复后逐位等价。"""
        return {
            "state": self._x.tolist(),
            "covariance": self._P.tolist(),
            "t": self._t,
            "last_wheel": (
                None
                if self._last_wheel is None
                else {
                    "t": self._last_wheel.t,
                    "left_counts": self._last_wheel.left_counts,
                    "right_counts": self._last_wheel.right_counts,
                }
            ),
            "last_update_kind": self.last_update_kind,
        }

    def restore(self, data: dict) -> None:
        self._x = np.array(data["state"], dtype=float)
        self._P = np.array(data["covariance"], dtype=float)
        self._t = data["t"]
        lw = data["last_wheel"]
        self._last_wheel = (
            None if lw is None
            else _WheelCarry(lw["t"], lw["left_counts"], lw["right_counts"])
        )
        self.last_update_kind = data["last_update_kind"]
