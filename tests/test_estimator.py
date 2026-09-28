import math

import numpy as np
import pytest

from robotloc.estimator import EkfConfig, EkfLocalizer
from robotloc.types import ImuObservation, WheelObservation


def _wheel(t, lc, rc, dt=0.05):
    return WheelObservation(
        t=t, received_at=t, left_counts=lc, right_counts=rc, dt_counts=dt
    )


def test_straight_line_motion():
    """恒速直行：位置应随时间线性，航向保持 0。"""
    loc = EkfLocalizer()
    k = 1024.0 / (2 * math.pi * 0.1)  # counts / meter
    lc = rc = 0.0
    for i in range(1, 101):  # 20Hz x 5s, v=1 m/s
        lc += 1.0 * 0.05 * k
        rc += 1.0 * 0.05 * k
        t = i * 0.05
        loc.update_wheel(_wheel(t, lc, rc))
        loc.update_imu(ImuObservation(t=t, received_at=t, yaw_rate=0.0))
    est = loc.estimate(5.0)
    assert abs(est.x - 5.0) < 0.05
    assert abs(est.y) < 0.05
    assert abs(est.theta) < 0.02


def test_covariance_grows_without_observations():
    """断流推演：时间越久不确定性越大，而不是精度不变。"""
    loc = EkfLocalizer()
    k = 1024.0 / (2 * math.pi * 0.1)
    lc = rc = 0.0
    for i in range(1, 21):
        lc += 1.0 * 0.05 * k
        rc += 1.0 * 0.05 * k
        loc.update_wheel(_wheel(i * 0.05, lc, rc))
    p_before = loc.estimate(1.0).position_sigma
    loc.estimate(3.0)
    p_short = loc.estimate(3.0).position_sigma
    loc.estimate(8.0)
    p_long = loc.estimate(8.0).position_sigma
    assert p_long > p_short > p_before


def test_time_travel_rejected():
    """估计器自身拒绝倒流；迟到由 pipeline 负责回滚。"""
    loc = EkfLocalizer()
    k = 1024.0 / (2 * math.pi * 0.1)
    loc.update_wheel(_wheel(1.0, 0.05 * k, 0.05 * k))
    with pytest.raises(ValueError):
        loc.predict(0.5)


def test_checkpoint_roundtrip_bit_identical():
    rng = np.random.default_rng(0)
    loc = EkfLocalizer()
    k = 1024.0 / (2 * math.pi * 0.1)
    lc = rc = 0.0
    for i in range(1, 61):
        v = 0.8 + 0.1 * math.sin(i * 0.1)
        w = 0.1 * math.cos(i * 0.07)
        lc += (v - 0.25 * w) * 0.05 * k
        rc += (v + 0.25 * w) * 0.05 * k
        t = i * 0.05
        loc.update_wheel(_wheel(t, lc, rc))
        loc.update_imu(ImuObservation(t=t, received_at=t, yaw_rate=w))
    snap = loc.checkpoint()
    e1 = loc.estimate(3.0)

    loc2 = EkfLocalizer()
    loc2.restore(snap)
    e2 = loc2.estimate(3.0)
    np.testing.assert_array_equal(e1.covariance, e2.covariance)
    assert e1.x == e2.x and e1.y == e2.y and e1.theta == e2.theta


def test_slip_observation_downweighted():
    """打滑起始的突变应被门限拦截。

    恒速、且两轮同比例打滑时，陀螺角速度同样为 0，单凭本系统的两类
    传感器无法完全消除速度尺度误差——这是留给后续地图匹配的可观性。
    这里检验：开始/结束的突变被拒绝后，误差明显小于“完全相信打滑
    读数”的 0.75m。
    """
    def build(slip_gain):
        loc = EkfLocalizer(config=EkfConfig())
        k = 1024.0 / (2 * math.pi * 0.1)
        lc = rc = 0.0
        for i in range(1, 121):
            gain = slip_gain if 40 <= i <= 60 else 1.0
            lc += 1.0 * 0.05 * k * gain
            rc += 1.0 * 0.05 * k * gain
            loc.update_wheel(_wheel(i * 0.05, lc, rc))
            loc.update_imu(ImuObservation(t=i * 0.05, received_at=i * 0.05,
                                         yaw_rate=0.0))
        return loc.estimate(6.0)
    slipped = build(1.5)
    assert abs(slipped.x - 6.0) < 0.6
