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
        calibration: dict[str, object] | None = None,
        arrival_horizon: float | None = None,
        seal_time: float | None = None,
        sealed_outputs: list[StateEstimate] | None = None,
        rejected_observations: dict[tuple[str, int], str] | None = None,
        staged_calibration: dict[str, object] | None = None,
        staged_sensor: str | None = None,
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
            "clock_models": calibration or {},
            "fixed_delays": _legacy_fixed_delays(calibration or {}),
            "staged_calibration": staged_calibration,
            "staged_sensor": staged_sensor,
            "sealed_outputs": [
                serialize_state(item)
                for item in (sealed_outputs or [])
            ],
            "rejected_observations": [
                {
                    "sensor": sensor,
                    "sequence": sequence,
                    "reason": reason,
                }
                for (sensor, sequence), reason in (rejected_observations or {}).items()
            ],
        }
        Path(self.path).write_text(json.dumps(payload), encoding="utf-8")

    def load_with_metadata(
        self,
    ) -> tuple[StateEstimate, list[Observation], dict[str, object]]:
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
        metadata = {
            "arrival_horizon": payload.get("arrival_horizon"),
            "seal_time": payload.get("seal_time"),
            "clock_models": payload.get(
                "clock_models", payload.get("fixed_delays", {})
            ),
            "fixed_delays": payload.get("fixed_delays", {}),
            "sealed_outputs": [
                deserialize_state(item) for item in payload.get("sealed_outputs", [])
            ],
            "rejected_observations": {
                (item["sensor"], item["sequence"]): item["reason"]
                for item in payload.get("rejected_observations", [])
            },
            "staged_calibration": payload.get("staged_calibration"),
            "staged_sensor": payload.get("staged_sensor"),
        }
        return estimate, observations, metadata

    def load(self) -> tuple[StateEstimate, list[Observation]]:
        estimate, observations, _ = self.load_with_metadata()
        return estimate, observations


def _legacy_fixed_delays(calibration: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for sensor, model in calibration.items():
        if isinstance(model, (int, float)):
            result[sensor] = float(model)
        elif isinstance(model, dict) and model.get("type", "fixed") == "fixed":
            result[sensor] = float(model.get("delay", 0.0))
    return result


def serialize_state(estimate: StateEstimate) -> dict[str, object]:
    return {
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
    }


def deserialize_state(payload: dict[str, object]) -> StateEstimate:
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
