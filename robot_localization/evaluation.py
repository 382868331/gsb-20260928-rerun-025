from __future__ import annotations

import math
from dataclasses import dataclass

from .estimator import StateEstimate
from .sources import GroundTruthPoint


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    samples: int
    rmse_position: float
    mean_abs_heading_error: float
    uncertainty_coverage: float
    mean_position_95_radius: float
    degraded_samples: int


def compare_with_truth(
    estimates: list[StateEstimate],
    truth: list[GroundTruthPoint],
) -> TrajectoryMetrics:
    squared_errors: list[float] = []
    heading_errors: list[float] = []
    covered = 0
    radii: list[float] = []
    degraded_samples = 0
    for estimate in estimates:
        point = _interpolate_truth(truth, estimate.time)
        if point is None:
            continue
        error = math.hypot(estimate.x - point.x, estimate.y - point.y)
        squared_errors.append(error**2)
        heading_errors.append(abs(_angle_diff(estimate.heading, point.heading)))
        radii.append(estimate.position_95_radius)
        if error <= estimate.position_95_radius:
            covered += 1
        if estimate.degraded_reasons:
            degraded_samples += 1
    samples = len(squared_errors)
    if samples == 0:
        raise ValueError("no overlapping trajectory samples")
    return TrajectoryMetrics(
        samples=samples,
        rmse_position=math.sqrt(sum(squared_errors) / samples),
        mean_abs_heading_error=sum(heading_errors) / samples,
        uncertainty_coverage=covered / samples,
        mean_position_95_radius=sum(radii) / samples,
        degraded_samples=degraded_samples,
    )

def _interpolate_truth(
    truth: list[GroundTruthPoint],
    time: float,
) -> GroundTruthPoint | None:
    if not truth or time < truth[0].time - 1e-9 or time > truth[-1].time + 1e-9:
        return None
    for index, point in enumerate(truth[1:], start=1):
        previous = truth[index - 1]
        if previous.time <= time <= point.time:
            span = point.time - previous.time
            ratio = 0.0 if span <= 1e-12 else (time - previous.time) / span
            return GroundTruthPoint(
                time=time,
                x=_linear(previous.x, point.x, ratio),
                y=_linear(previous.y, point.y, ratio),
                heading=_angle_linear(previous.heading, point.heading, ratio),
                linear_velocity=_linear(
                    previous.linear_velocity, point.linear_velocity, ratio
                ),
                yaw_rate=_linear(previous.yaw_rate, point.yaw_rate, ratio),
                forward_acceleration=_linear(
                    previous.forward_acceleration,
                    point.forward_acceleration,
                    ratio,
                ),
            )
    return truth[-1]


def _linear(left: float, right: float, ratio: float) -> float:
    return left + (right - left) * ratio


def _angle_linear(left: float, right: float, ratio: float) -> float:
    return _normalize_angle(left + _angle_diff(right, left) * ratio)


def _angle_diff(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
