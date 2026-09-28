"""用已知路径核对定位误差与误差范围（协方差）的合理性。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scenario import GroundTruth
from .types import PoseEstimate


@dataclass
class Metrics:
    rms_xy: float
    max_xy: float
    rms_theta: float
    max_theta: float
    # 误差落在 1σ / 2σ 椭圆内的比例（粗略一致性检验）
    inside_1sigma: float
    inside_2sigma: float
    drift_xy_at_end: float

    def as_dict(self) -> dict:
        return {
            "rms_xy_m": self.rms_xy,
            "max_xy_m": self.max_xy,
            "rms_theta_deg": math.degrees(self.rms_theta),
            "max_theta_deg": math.degrees(self.max_theta),
            "inside_1sigma": self.inside_1sigma,
            "inside_2sigma": self.inside_2sigma,
            "drift_end_m": self.drift_xy_at_end,
        }


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def evaluate(
    estimates: list[PoseEstimate], gt: GroundTruth
) -> Metrics:
    th_unwrap = np.unwrap(gt.theta)
    errs_xy: list[float] = []
    errs_th: list[float] = []
    n1 = n2 = 0
    n = 0
    for e in estimates:
        if e.t < gt.t[0] or e.t > gt.t[-1]:
            continue
        gx = float(np.interp(e.t, gt.t, gt.x))
        gy = float(np.interp(e.t, gt.t, gt.y))
        gth = float(np.interp(e.t, gt.t, th_unwrap))
        dxy = math.hypot(e.x - gx, e.y - gy)
        dth = abs(_wrap(e.theta - gth))
        errs_xy.append(dxy)
        errs_th.append(dth)
        sx = math.sqrt(max(e.covariance[0, 0], 1e-12))
        sy = math.sqrt(max(e.covariance[1, 1], 1e-12))
        st = math.sqrt(max(e.covariance[2, 2], 1e-12))
        if dxy <= math.sqrt(e.covariance[0, 0] + e.covariance[1, 1]) and dth <= st:
            n1 += 1
        if dxy <= 2 * math.sqrt(e.covariance[0, 0] + e.covariance[1, 1]) and dth <= 2 * st:
            n2 += 1
        n += 1
    exy = np.array(errs_xy)
    eth = np.array(errs_th)
    return Metrics(
        rms_xy=float(np.sqrt(np.mean(exy**2))),
        max_xy=float(np.max(exy)),
        rms_theta=float(np.sqrt(np.mean(eth**2))),
        max_theta=float(np.max(eth)),
        inside_1sigma=n1 / n,
        inside_2sigma=n2 / n,
        drift_xy_at_end=float(exy[-1]) if len(exy) else float("nan"),
    )
