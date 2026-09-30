from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .types import ImuObservation, Observation, WheelObservation


class ObservationSource(Protocol):
    def observations(self) -> list[Observation]: ...


@dataclass(frozen=True, slots=True)
class GroundTruthPoint:
    time: float
    x: float
    y: float
    heading: float
    linear_velocity: float
    yaw_rate: float
    forward_acceleration: float


class SyntheticScenarioSource:
    """Deterministic fictional sensor data: acceleration, a 90-degree turn, slip, gaps."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def ground_truth(self, dt: float = 0.02) -> list[GroundTruthPoint]:
        points: list[GroundTruthPoint] = []
        x = y = heading = velocity = 0.0
        previous_velocity = 0.0
        steps = int(10.0 / dt)
        for step in range(steps + 1):
            time = step * dt
            acceleration = self._commanded_acceleration(time)
            yaw_rate = self._commanded_yaw_rate(time)
            if step > 0:
                midpoint_heading = heading + 0.5 * yaw_rate * dt
                x += velocity * math.cos(midpoint_heading) * dt
                y += velocity * math.sin(midpoint_heading) * dt
                heading += yaw_rate * dt
                velocity += acceleration * dt
                velocity = max(0.0, velocity)
            points.append(
                GroundTruthPoint(
                    time=time,
                    x=x,
                    y=y,
                    heading=_normalize_angle(heading),
                    linear_velocity=velocity,
                    yaw_rate=yaw_rate,
                    forward_acceleration=acceleration,
                )
            )
            previous_velocity = velocity
        return points

    def observations(self) -> list[Observation]:
        rng = np.random.default_rng(self.seed)
        truth = self.ground_truth()
        messages: list[Observation] = []
        wheel_sequence = imu_sequence = 0
        for point in truth:
            time = point.time
            if abs((time * 10.0) - round(time * 10.0)) < 1e-9 and time > 0.0:
                if not (6.8 < time < 7.4):  # short wheel outage during/after slip
                    slip_offset = 0.65 if 6.0 <= time < 8.0 else 0.0
                    wheel_velocity = (
                        point.linear_velocity
                        + slip_offset
                        + float(rng.normal(0.0, 0.035))
                    )
                    messages.append(
                        WheelObservation(
                            time=time,
                            sensor="rear_wheels",
                            sequence=wheel_sequence,
                            linear_velocity=wheel_velocity,
                            yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.015)),
                            linear_velocity_variance=0.01,
                            yaw_rate_variance=0.004,
                        )
                    )
                    wheel_sequence += 1
            if abs((time * 20.0) - round(time * 20.0)) < 1e-9 and time > 0.0:
                if not (7.1 < time < 7.7):  # short IMU outage
                    messages.append(
                        ImuObservation(
                            time=time,
                            sensor="imu",
                            sequence=imu_sequence,
                            yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.018)),
                            forward_acceleration=point.forward_acceleration
                            + float(rng.normal(0.0, 0.04)),
                            yaw_rate_variance=0.002,
                            acceleration_variance=0.004,
                        )
                    )
                    imu_sequence += 1
        # A wheel sample from the turn arrives after some later messages. It must
        # be withheld until arrival, then trigger rollback rather than being
        # silently ignored or applied to the newest timestamp.
        late = next(item for item in messages if isinstance(item, WheelObservation) and 4.8 <= item.time <= 5.2)
        messages.remove(late)
        insertion = min(len(messages), 70)
        messages.insert(insertion, late)
        return messages

    @staticmethod
    def _commanded_acceleration(time: float) -> float:
        if time < 2.0:
            return 0.5
        return 0.0

    @staticmethod
    def _commanded_yaw_rate(time: float) -> float:
        if 4.0 <= time < 4.0 + math.pi / (2.0 * 0.7853981633974483):
            return 0.7853981633974483
        return 0.0


class TimedDelayedScenarioSource:
    """Known path with delayed wheel timestamps, an outage, burst replay, and slip."""

    wheel_delay = 0.5

    def __init__(self, seed: int = 17) -> None:
        self.base = SyntheticScenarioSource(seed)

    def ground_truth(self, dt: float = 0.02) -> list[GroundTruthPoint]:
        return self.base.ground_truth(dt)

    def observations(self) -> list[Observation]:
        rng = np.random.default_rng(self.base.seed)
        messages: list[Observation] = []
        wheel_sequence = imu_sequence = 0
        burst: list[Observation] = []
        for point in self.ground_truth():
            if point.time <= 0.0:
                continue
            if abs(point.time * 10.0 - round(point.time * 10.0)) < 1e-9:
                observed_time = point.time + self.wheel_delay
                in_outage = 2.0 <= point.time < 4.0
                arrival = 4.2 if in_outage else observed_time
                slip_offset = 0.7 if 6.0 <= point.time < 7.0 else 0.0
                wheel = WheelObservation(
                    time=observed_time,
                    sensor="rear_wheels",
                    sequence=wheel_sequence,
                    arrival_time=arrival,
                    linear_velocity=point.linear_velocity
                    + slip_offset
                    + float(rng.normal(0.0, 0.025)),
                    yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.012)),
                    linear_velocity_variance=0.01,
                    yaw_rate_variance=0.004,
                )
                wheel_sequence += 1
                if in_outage:
                    burst.append(wheel)
                elif abs(point.time - 5.0) < 1e-9:
                    object.__setattr__(wheel, "arrival_time", 5.7)
                    messages.append(wheel)
                else:
                    messages.append(wheel)
            if abs(point.time * 20.0 - round(point.time * 20.0)) < 1e-9:
                messages.append(
                    ImuObservation(
                        time=point.time,
                        sensor="imu",
                        sequence=imu_sequence,
                        arrival_time=point.time,
                        yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.015)),
                        forward_acceleration=point.forward_acceleration
                        + float(rng.normal(0.0, 0.035)),
                        yaw_rate_variance=0.002,
                        acceleration_variance=0.004,
                    )
                )
                imu_sequence += 1
        messages.sort(key=lambda item: (item.arrival_time or 0.0, item.sensor, item.sequence))
        insert_at = next(
            index
            for index, item in enumerate(messages)
            if item.arrival_time is not None and item.arrival_time >= 4.2
        )
        return messages[:insert_at] + burst + messages[insert_at:]

def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
