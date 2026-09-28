"""虚构样例数据生成：已知真值路径 + 转弯 + 轮速打滑 + 迟到/断流。

不读取真实设备。真值用于核对误差，观测中叠加固定种子的噪声。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import ImuObservation, WheelObservation


@dataclass(frozen=True)
class GroundTruth:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray

    def at(self, t: float) -> tuple[float, float, float]:
        x = float(np.interp(t, self.t, self.x))
        y = float(np.interp(t, self.t, self.y))
        theta = float(np.interp(np.unwrap(self.theta), self.t, np.unwrap(self.theta)))
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        return x, y, theta


def make_ground_truth(duration: float = 20.0, dt: float = 0.01) -> GroundTruth:
    """三段式路径：直行加速 -> 90° 转弯 -> 直行减速 -> 反向转弯回正。"""
    t = np.arange(0.0, duration + dt / 2, dt)
    v = np.zeros_like(t)
    omega = np.zeros_like(t)
    # 0-4s 加速到 1 m/s；4-7s 恒速；7-10s 原地段以 0.5 m/s、0.35 rad/s 转弯
    v[(t >= 0) & (t < 4)] = np.linspace(0.0, 1.0, ((t >= 0) & (t < 4)).sum())
    v[(t >= 4) & (t < 7)] = 1.0
    v[(t >= 7) & (t < 11.5)] = 0.5
    omega[(t >= 7) & (t < 11.5)] = 0.35          # 约 0.5 圈
    v[(t >= 11.5) & (t < 15.5)] = 0.8
    v[(t >= 15.5) & (t < 17.0)] = 0.6
    omega[(t >= 15.5) & (t < 20.0)] = -0.22      # 回旋
    v[(t >= 17.0)] = np.linspace(0.6, 0.0, (t >= 17.0).sum())

    theta = np.zeros_like(t)
    theta[1:] = np.cumsum(0.5 * (omega[:-1] + omega[1:]) * dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    x[1:] = np.cumsum(0.5 * (v[:-1] + v[1:]) * np.cos(theta[1:]) * dt)
    y[1:] = np.cumsum(0.5 * (v[:-1] + v[1:]) * np.sin(theta[1:]) * dt)
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    return GroundTruth(t=t, x=x, y=y, theta=theta, v=v, omega=omega)


@dataclass
class ScenarioConfig:
    wheel_hz: float = 20.0
    imu_hz: float = 50.0
    wheel_radius: float = 0.1
    wheel_base: float = 0.5
    counts_per_revolution: float = 1024.0
    gyro_noise: float = 0.008         # rad/s
    yaw_noise: float = 0.03           # rad（低频航向参考）
    yaw_every: float = 1.0            # 每 1s 给一次航向参考
    slip_start: float = 8.5           # 打滑区间（转弯内）
    slip_end: float = 10.5
    slip_gain: float = 1.35           # 轮速读数虚高 35%
    dropout_start: float = 13.0       # 消息短暂缺失
    dropout_end: float = 14.2
    late_delay: float = 0.8           # 某条 IMU 迟到 0.8s
    late_target_t: float = 5.2
    seed: int = 20260929


def make_messages(
    gt: GroundTruth, cfg: ScenarioConfig | None = None
) -> list[WheelObservation | ImuObservation]:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    msgs: list[WheelObservation | ImuObservation] = []

    # ---- 轮速：按累计计数生成，打滑区间人为虚增计数 ----
    step = 1.0 / cfg.wheel_hz
    times = np.arange(step, gt.t[-1] - 1e-9, step)
    left_counts = right_counts = 0.0
    seq_w = 0
    k_counts = cfg.counts_per_revolution / (2 * math.pi * cfg.wheel_radius)
    for t in times:
        v = float(np.interp(t, gt.t, gt.v))
        w = _omega_at(t, gt)
        gain = (
            cfg.slip_gain
            if cfg.slip_start <= t <= cfg.slip_end
            else 1.0
        )
        v_l = (v - 0.5 * w * cfg.wheel_base) * gain
        v_r = (v + 0.5 * w * cfg.wheel_base) * gain
        v_l += rng.normal(0.0, cfg.gyro_noise * 0.5)
        v_r += rng.normal(0.0, cfg.gyro_noise * 0.5)
        left_counts += v_l * step * k_counts
        right_counts += v_r * step * k_counts
        received = t
        if cfg.dropout_start <= t < cfg.dropout_end:
            continue  # 消息缺失（既不发送也不补发）
        msgs.append(
            WheelObservation(
                t=float(t), received_at=float(received),
                left_counts=left_counts, right_counts=right_counts,
                dt_counts=step, wheel_radius=cfg.wheel_radius,
                wheel_base=cfg.wheel_base,
                counts_per_revolution=cfg.counts_per_revolution,
                seq=seq_w,
            )
        )
        seq_w += 1

    # ---- IMU：50Hz 陀螺 + 1Hz 航向参考 ----
    step_i = 1.0 / cfg.imu_hz
    times_i = np.arange(step_i, gt.t[-1] - 1e-9, step_i)
    seq_i = 0
    theta_unwrap = np.unwrap(gt.theta)
    for i, t in enumerate(times_i):
        if cfg.dropout_start <= t < cfg.dropout_end:
            continue
        w = _omega_at(float(t), gt) + float(rng.normal(0, cfg.gyro_noise))
        yaw = None
        if abs(t % cfg.yaw_every) < step_i / 2:
            yaw_val = float(np.interp(float(t), gt.t, theta_unwrap))
            yaw_val += float(rng.normal(0, cfg.yaw_noise))
            yaw = (yaw_val + math.pi) % (2 * math.pi) - math.pi
        received = float(t)
        if abs(t - cfg.late_target_t) < step_i / 2:
            received += cfg.late_delay
        msgs.append(
            ImuObservation(
                t=float(t), received_at=received, yaw_rate=w, yaw=yaw, seq=seq_i
            )
        )
        seq_i += 1

    msgs.sort(key=lambda m: (m.received_at, m.seq))
    return msgs


def _omega_at(t: float, gt: GroundTruth) -> float:
    # 真值角速率：直接对 unwrap(theta) 中心差分后插值
    th = np.unwrap(gt.theta)
    dt = gt.t[1] - gt.t[0]
    dth = np.gradient(th, dt)
    return float(np.interp(t, gt.t, dth))
