from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Observation:
    time: float
    sensor: str
    sequence: int
    arrival_time: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class WheelObservation(Observation):
    linear_velocity: float
    yaw_rate: float
    linear_velocity_variance: float = 0.01
    yaw_rate_variance: float = 0.01

    def __post_init__(self) -> None:
        if self.time < 0.0:
            raise ValueError("observation time must be non-negative")
        if self.linear_velocity_variance <= 0.0 or self.yaw_rate_variance <= 0.0:
            raise ValueError("measurement variance must be positive")


@dataclass(frozen=True, slots=True)
class ImuObservation(Observation):
    yaw_rate: float
    forward_acceleration: float
    yaw_rate_variance: float = 0.0025
    acceleration_variance: float = 0.01

    def __post_init__(self) -> None:
        if self.time < 0.0:
            raise ValueError("observation time must be non-negative")
        if self.yaw_rate_variance <= 0.0 or self.acceleration_variance <= 0.0:
            raise ValueError("measurement variance must be positive")
