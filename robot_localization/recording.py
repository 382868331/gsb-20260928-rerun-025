from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .estimator import StateEstimate
from .types import ImuObservation, Observation, WheelObservation


class JsonlRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write_many(self, observations: Iterable[Observation]) -> None:
        with self.path.open("w", encoding="utf-8") as stream:
            for observation in observations:
                stream.write(json.dumps(serialize_observation(observation), allow_nan=False))
                stream.write("\n")


class JsonlMessageSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[Observation]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield deserialize_observation(json.loads(line))


@dataclass(slots=True)
class CheckpointStore:
    path: str | Path

    def save(
        self,
        estimate: StateEstimate,
        observations: Iterable[Observation] | None = None,
        calibration: dict[str, float] | None = None,
        arrival_horizon: float | None = None,
        seal_time: float | None = None,
    ) -> None:
        payload = {
            "time": estimate.time,
            "mean": [
                estimate.x,
                estimate.y,
                estimate.heading,
                estimate.linear_velocity,
                estimate.yaw_rate,
                estimate.forward_acceleration,
            ],
            "covariance": estimate.covariance.tolist(),
            "last_wheel_time": estimate.last_wheel_time,
            "last_imu_time": estimate.last_imu_time,
            "degraded_reasons": sorted(estimate.degraded_reasons),
            "observations": [
                serialize_observation(item)
                for item in (observations or [])
            ],
            "arrival_horizon": arrival_horizon,
            "seal_time": seal_time,
            "fixed_delays": calibration or {},
        }
        Path(self.path).write_text(json.dumps(payload), encoding="utf-8")

    def save_pipeline_checkpoint(self, checkpoint) -> None:
        payload = {
            "state": {
                "time": checkpoint.state.time,
                "mean": [
                    checkpoint.state.x,
                    checkpoint.state.y,
                    checkpoint.state.heading,
                    checkpoint.state.linear_velocity,
                    checkpoint.state.yaw_rate,
                    checkpoint.state.forward_acceleration,
                ],
                "covariance": checkpoint.state.covariance.tolist(),
                "last_wheel_time": checkpoint.state.last_wheel_time,
                "last_imu_time": checkpoint.state.last_imu_time,
                "degraded_reasons": sorted(checkpoint.state.degraded_reasons),
            },
            "event_count": checkpoint.event_count,
            "arrival_horizon": checkpoint.arrival_horizon,
            "seal_time": checkpoint.seal_time,
            "sealed_at_arrival": checkpoint.sealed_at_arrival,
            "fixed_delays": (checkpoint.calibration or DelayCalibration()).as_dict(),
            "observations": [
                serialize_observation(item)
                for item in checkpoint.visible_observations
            ],
            "pending_corrections": [
                list(item) for item in checkpoint.pending_corrections
            ],
            "rejected_observations": [
                self._serialize_impact(item) for item in checkpoint.rejected_observations
            ],
            "held_observation_keys": [
                list(item) for item in checkpoint.held_observation_keys
            ],
            "sealed_outputs": [
                self._serialize_state(item) for item in checkpoint.sealed_outputs
            ],
        }
        Path(self.path).write_text(json.dumps(payload), encoding="utf-8")

    def load_pipeline_checkpoint(self):
        from .calibration import DelayCalibration
        from .pipeline import Checkpoint, ObservationImpact

        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        state_payload = payload.get("state", payload)
        mean = state_payload["mean"]
        covariance = state_payload.get("covariance")
        if covariance is None:
            covariance = np.eye(6) * 1e-6
        state = StateEstimate(
            time=float(state_payload["time"]),
            x=float(mean[0]),
            y=float(mean[1]),
            heading=float(mean[2]),
            linear_velocity=float(mean[3]),
            yaw_rate=float(mean[4]),
            forward_acceleration=float(mean[5]),
            covariance=np.asarray(covariance, dtype=float),
            last_wheel_time=state_payload.get("last_wheel_time"),
            last_imu_time=state_payload.get("last_imu_time"),
            degraded_reasons=set(state_payload.get("degraded_reasons", [])),
        )
        return Checkpoint(
            state=state,
            event_count=int(payload.get("event_count", 0)),
            arrival_horizon=payload.get("arrival_horizon"),
            seal_time=payload.get("seal_time"),
            calibration=DelayCalibration(payload.get("fixed_delays", {})),
            visible_observations=[
                deserialize_observation(item)
                for item in payload.get("observations", [])
            ],
            sealed_at_arrival=payload.get("sealed_at_arrival"),
            pending_corrections=[
                tuple(item) for item in payload.get("pending_corrections", [])
            ],
            rejected_observations=[
                ObservationImpact(**item)
                for item in payload.get("rejected_observations", [])
            ],
            held_observation_keys=[
                tuple(item) for item in payload.get("held_observation_keys", [])
            ],
            sealed_outputs=[
                self._deserialize_state(item)
                for item in payload.get("sealed_outputs", [])
            ],
        )

    @staticmethod
    def _serialize_impact(impact) -> dict[str, object]:
        return {
            "sensor": impact.sensor,
            "sequence": impact.sequence,
            "stamp_time": impact.stamp_time,
            "arrival_time": impact.arrival_time,
            "old_corrected_time": impact.old_corrected_time,
            "new_corrected_time": impact.new_corrected_time,
            "action": impact.action,
            "reason": impact.reason,
        }

    @staticmethod
    def _serialize_state(state: StateEstimate) -> dict[str, object]:
        return {
            "time": state.time,
            "mean": [
                state.x,
                state.y,
                state.heading,
                state.linear_velocity,
                state.yaw_rate,
                state.forward_acceleration,
            ],
            "covariance": state.covariance.tolist(),
            "last_wheel_time": state.last_wheel_time,
            "last_imu_time": state.last_imu_time,
            "degraded_reasons": sorted(state.degraded_reasons),
        }

    @staticmethod
    def _deserialize_state(payload: dict[str, object]) -> StateEstimate:
        mean = payload["mean"]
        return StateEstimate(
            time=float(payload["time"]),
            x=float(mean[0]),
            y=float(mean[1]),
            heading=float(mean[2]),
            linear_velocity=float(mean[3]),
            yaw_rate=float(mean[4]),
            forward_acceleration=float(mean[5]),
            covariance=np.asarray(payload["covariance"], dtype=float),
            last_wheel_time=payload.get("last_wheel_time"),
            last_imu_time=payload.get("last_imu_time"),
            degraded_reasons=set(payload.get("degraded_reasons", [])),
        )

    def load(self) -> tuple[StateEstimate, list[Observation]]:
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        mean = payload["mean"]
        estimate = StateEstimate(
            time=float(payload["time"]),
            x=float(mean[0]),
            y=float(mean[1]),
            heading=float(mean[2]),
            linear_velocity=float(mean[3]),
            yaw_rate=float(mean[4]),
            forward_acceleration=float(mean[5]),
            covariance=np.asarray(payload["covariance"], dtype=float),
            last_wheel_time=payload.get("last_wheel_time"),
            last_imu_time=payload.get("last_imu_time"),
            degraded_reasons=set(payload.get("degraded_reasons", [])),
        )
        observations = [
            deserialize_observation(item) for item in payload.get("observations", [])
        ]
        return estimate, observations


def serialize_observation(observation: Observation) -> dict[str, object]:
    payload = asdict(observation)
    payload["type"] = type(observation).__name__
    return payload


def deserialize_observation(payload: dict[str, object]) -> Observation:
    kind = payload.pop("type")
    if kind == "WheelObservation":
        return WheelObservation(**payload)  # type: ignore[arg-type]
    if kind == "ImuObservation":
        return ImuObservation(**payload)  # type: ignore[arg-type]
    raise ValueError(f"unknown observation type: {kind}")
