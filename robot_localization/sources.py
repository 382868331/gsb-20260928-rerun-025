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
    """Deterministic fictional sensor data: acceleration, a turn, slip, gaps.

    With ``fixed_delays`` unset the source is the original raw stream (no
    arrival timestamps).  With delays provided, every message keeps its stamp
    and gains a separate arrival time; the wheel stream suffers a sustained
    outage whose samples arrive later as an ordered burst (backfill).
    """

    WHEEL_OUTAGE = (6.8, 7.7)
    WHEEL_BACKFILL_ARRIVAL = 7.95

    def __init__(
        self,
        seed: int = 7,
        fixed_delays: dict[str, float] | None = None,
        arrival_jitter: float = 0.02,
    ) -> None:
        self.seed = seed
        self.fixed_delays = fixed_delays
        self.arrival_jitter = arrival_jitter

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
        physical_times: dict[tuple[str, int], float] = {}
        wheel_sequence = imu_sequence = 0
        delayed = self.fixed_delays is not None
        for point in truth:
            physical_time = point.time
            time = point.time
            if abs((time * 10.0) - round(time * 10.0)) < 1e-9 and time > 0.0:
                outage_start, outage_end = self.WHEEL_OUTAGE
                legacy_gap = not delayed and outage_start < time < 7.4
                if not legacy_gap:
                    slip_offset = 0.65 if 6.0 <= time < 8.0 else 0.0
                    wheel_velocity = (
                        point.linear_velocity
                        + slip_offset
                        + float(rng.normal(0.0, 0.035))
                    )
                    arrival = None
                    stamped_time = time
                    if delayed:
                        delay = self.fixed_delays.get("rear_wheels", 0.0)
                        jitter = float(rng.uniform(-self.arrival_jitter, self.arrival_jitter))
                        arrival = time + delay + abs(jitter)
                        if outage_start <= time < outage_end:
                            arrival = self.WHEEL_BACKFILL_ARRIVAL + wheel_sequence * 0.002
                        # Stamp is assigned on receipt; corrected time = stamp - delay.
                        stamped_time = arrival - abs(jitter)
                    messages.append(
                        WheelObservation(
                            time=stamped_time,
                            sensor="rear_wheels",
                            sequence=wheel_sequence,
                            arrival_time=arrival,
                            linear_velocity=wheel_velocity,
                            yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.015)),
                            linear_velocity_variance=0.01,
                            yaw_rate_variance=0.004,
                        )
                    )
                    physical_times[("rear_wheels", wheel_sequence)] = physical_time
                    wheel_sequence += 1
            if abs((time * 20.0) - round(time * 20.0)) < 1e-9 and time > 0.0:
                if delayed or not (7.1 < time < 7.7):  # short IMU outage
                    arrival = None
                    stamped_time = time
                    if delayed:
                        delay = self.fixed_delays.get("imu", 0.0)
                        jitter = float(rng.uniform(-self.arrival_jitter, self.arrival_jitter))
                        arrival = time + delay + abs(jitter)
                        stamped_time = arrival - abs(jitter)
                    messages.append(
                        ImuObservation(
                            time=stamped_time,
                            sensor="imu",
                            sequence=imu_sequence,
                            arrival_time=arrival,
                            yaw_rate=point.yaw_rate + float(rng.normal(0.0, 0.018)),
                            forward_acceleration=point.forward_acceleration
                            + float(rng.normal(0.0, 0.04)),
                            yaw_rate_variance=0.002,
                            acceleration_variance=0.004,
                        )
                    )
                    physical_times[("imu", imu_sequence)] = physical_time
                    imu_sequence += 1
        # One wheel sample from the turn arrives badly out of order. It must be
        # withheld until arrival, then revise history rather than being stamped
        # with the newest time.
        late = next(
            item
            for item in messages
            if isinstance(item, WheelObservation)
            and 4.8 <= physical_times[(item.sensor, item.sequence)] <= 5.2
        )
        messages.remove(late)
        if delayed:
            delay = self.fixed_delays.get("rear_wheels", 0.0)
            late = WheelObservation(
                time=late.time,
                sensor=late.sensor,
                sequence=late.sequence,
                arrival_time=late.time + 1.6,
                linear_velocity=late.linear_velocity,
                yaw_rate=late.yaw_rate,
                linear_velocity_variance=late.linear_velocity_variance,
                yaw_rate_variance=late.yaw_rate_variance,
            )
            messages.sort(key=lambda item: item.observed_at())
        else:
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


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
