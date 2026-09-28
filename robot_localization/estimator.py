from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import ImuObservation, WheelObservation


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(slots=True)
class StateEstimate:
    time: float
    x: float
    y: float
    heading: float
    linear_velocity: float
    yaw_rate: float
    forward_acceleration: float
    covariance: np.ndarray
    last_wheel_time: float | None = None
    last_imu_time: float | None = None
    degraded_reasons: set[str] = field(default_factory=set)

    @property
    def position_95_radius(self) -> float:
        eigenvalues = np.linalg.eigvalsh(self.covariance[:2, :2])
        return float(1.96 * math.sqrt(max(0.0, eigenvalues.max())))

    @property
    def heading_95(self) -> float:
        return float(1.96 * math.sqrt(max(0.0, self.covariance[2, 2])))

    def quality(self) -> dict[str, object]:
        return {
            "time": self.time,
            "position_95_radius": self.position_95_radius,
            "heading_95": self.heading_95,
            "degraded_reasons": sorted(self.degraded_reasons),
        }


class LocalizationEstimator:
    """Causal error-state EKF for replaceable wheel and IMU observations.

    State is [x, y, heading, forward speed, yaw rate, forward acceleration].
    Map matching intentionally has no hook here yet; it can later be added as
    another measurement source without changing sensor ingestion or replay.
    """

    STATE_DIM = 6

    def __init__(
        self,
        *,
        acceleration_process_noise: float = 0.08,
        yaw_acceleration_process_noise: float = 0.25,
        wheel_timeout: float = 0.35,
        imu_timeout: float = 0.25,
        slip_gate_sigmas: float = 4.0,
    ) -> None:
        self.acceleration_process_noise = acceleration_process_noise
        self.yaw_acceleration_process_noise = yaw_acceleration_process_noise
        self.wheel_timeout = wheel_timeout
        self.imu_timeout = imu_timeout
        self.slip_gate_sigmas = slip_gate_sigmas
        self.state = np.zeros(self.STATE_DIM, dtype=float)
        self.covariance = np.diag([1e-9, 1e-9, 1e-9, 1.0, 0.25, 1.0])
        self.time = 0.0
        self.last_wheel_time: float | None = None
        self.last_imu_time: float | None = None
        self.degraded_reasons: set[str] = set()

    def snapshot(self) -> StateEstimate:
        self._refresh_quality()
        return StateEstimate(
            time=self.time,
            x=float(self.state[0]),
            y=float(self.state[1]),
            heading=float(_normalize_angle(float(self.state[2]))),
            linear_velocity=float(self.state[3]),
            yaw_rate=float(self.state[4]),
            forward_acceleration=float(self.state[5]),
            covariance=self.covariance.copy(),
            last_wheel_time=self.last_wheel_time,
            last_imu_time=self.last_imu_time,
            degraded_reasons=set(self.degraded_reasons),
        )

    def restore(self, snapshot: StateEstimate) -> None:
        self.time = snapshot.time
        self.state = np.array(
            [
                snapshot.x,
                snapshot.y,
                snapshot.heading,
                snapshot.linear_velocity,
                snapshot.yaw_rate,
                snapshot.forward_acceleration,
            ],
            dtype=float,
        )
        self.covariance = snapshot.covariance.copy()
        self.last_wheel_time = snapshot.last_wheel_time
        self.last_imu_time = snapshot.last_imu_time
        self.degraded_reasons = set(snapshot.degraded_reasons)

    def predict(self, time: float) -> StateEstimate:
        if time < self.time - 1e-9:
            raise ValueError("predictions cannot move backwards in time")
        dt = max(0.0, time - self.time)
        if dt > 0.0:
            _, _, heading, speed, yaw_rate, acceleration = self.state
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            transition = np.eye(self.STATE_DIM)
            transition[0, 3] = cos_h * dt
            transition[0, 5] = 0.5 * cos_h * dt * dt
            transition[1, 3] = sin_h * dt
            transition[1, 5] = 0.5 * sin_h * dt * dt
            transition[2, 4] = dt
            transition[3, 5] = dt

            q = np.zeros((self.STATE_DIM, self.STATE_DIM))
            qa = self.acceleration_process_noise * dt
            qw = self.yaw_acceleration_process_noise * dt
            q[0, 0] = q[1, 1] = 0.25 * qa * dt**4 + 0.01 * speed * speed * dt * dt
            q[0, 3] = q[3, 0] = 0.5 * qa * dt**2
            q[1, 3] = q[3, 1] = 0.5 * qa * dt**2
            q[2, 2] = qw * dt * dt
            q[2, 4] = q[4, 2] = qw * dt
            q[3, 3] = qa * dt * dt
            q[4, 4] = qw
            q[5, 5] = qa

            self.state = transition @ self.state
            self.state[2] = _normalize_angle(float(self.state[2]))
            self.covariance = transition @ self.covariance @ transition.T + q
            self.time = time
        return self.snapshot()

    def step(self, observation: WheelObservation | ImuObservation) -> StateEstimate:
        if observation.time < self.time - 1e-9:
            raise ValueError("observations must be applied causally")
        self.predict(observation.time)
        if isinstance(observation, WheelObservation):
            self._update_wheel(observation)
            self.last_wheel_time = observation.time
        elif isinstance(observation, ImuObservation):
            self._update_imu(observation)
            self.last_imu_time = observation.time
        else:
            raise TypeError(f"unsupported observation type: {type(observation)!r}")
        return self.snapshot()

    def _update_wheel(self, observation: WheelObservation) -> None:
        self.degraded_reasons.discard("wheel_slip_suspected")
        h = np.zeros((2, self.STATE_DIM))
        h[0, 3] = 1.0
        h[1, 4] = 1.0
        measured = np.array(
            [observation.linear_velocity, observation.yaw_rate], dtype=float
        )
        noise = np.diag(
            [
                observation.linear_velocity_variance,
                observation.yaw_rate_variance,
            ]
        )
        predicted = h @ self.state
        innovation = measured - predicted
        innovation[1] = _normalize_angle(float(innovation[1]))
        innovation_covariance = h @ self.covariance @ h.T + noise
        standard_deviations = np.sqrt(np.maximum(np.diag(innovation_covariance), 1e-12))
        ratios = np.abs(innovation) / standard_deviations

        # Wheel slip appears as a velocity innovation unsupported by IMU
        # acceleration. Gate the affected field instead of trusting wheel speed.
        for index in range(2):
            if ratios[index] > self.slip_gate_sigmas:
                noise[index, index] *= 1.0 + ratios[index] ** 2
                self.degraded_reasons.add("wheel_slip_suspected")
        innovation_covariance = h @ self.covariance @ h.T + noise
        self._apply_measurement(h, innovation, innovation_covariance, noise)

    def _update_imu(self, observation: ImuObservation) -> None:
        h = np.zeros((2, self.STATE_DIM))
        h[0, 4] = 1.0
        h[1, 5] = 1.0
        innovation = np.array(
            [
                _normalize_angle(observation.yaw_rate - float(self.state[4])),
                observation.forward_acceleration - float(self.state[5]),
            ]
        )
        noise = np.diag(
            [observation.yaw_rate_variance, observation.acceleration_variance]
        )
        innovation_covariance = h @ self.covariance @ h.T + noise
        self._apply_measurement(h, innovation, innovation_covariance, noise)

    def _apply_measurement(
        self,
        h: np.ndarray,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
        noise: np.ndarray,
    ) -> None:
        kalman_gain = self.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        self.state += kalman_gain @ innovation
        self.state[2] = _normalize_angle(float(self.state[2]))
        identity = np.eye(self.STATE_DIM)
        joseph_form = identity - kalman_gain @ h
        self.covariance = (
            joseph_form @ self.covariance @ joseph_form.T
            + kalman_gain @ noise @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _refresh_quality(self) -> None:
        wheel_age = None if self.last_wheel_time is None else self.time - self.last_wheel_time
        imu_age = None if self.last_imu_time is None else self.time - self.last_imu_time
        if wheel_age is not None and wheel_age > self.wheel_timeout:
            self.degraded_reasons.add("wheel_stream_gap")
        else:
            self.degraded_reasons.discard("wheel_stream_gap")
        if imu_age is not None and imu_age > self.imu_timeout:
            self.degraded_reasons.add("imu_stream_gap")
        else:
            self.degraded_reasons.discard("imu_stream_gap")
