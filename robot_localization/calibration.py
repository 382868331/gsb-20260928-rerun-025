from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .types import Observation

REBOOT_CLOCK_TYPE = "reboot_segments"


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """A known correspondence between a raw stamp and trajectory time."""

    stamp: float
    corrected_time: float

    @property
    def offset(self) -> float:
        return self.stamp - self.corrected_time


@dataclass(frozen=True, slots=True)
class SensorDelayModel:
    """Backwards-compatible fixed per-sensor transport delay estimate."""

    sensor: str
    delay: float = 0.0

    def __post_init__(self) -> None:
        if self.delay < 0.0:
            raise ValueError("fixed delay estimate must be non-negative")

    def corrected_time_at(self, stamp: float) -> float:
        return stamp - self.delay

    def offset_at(self, stamp: float) -> float:
        return self.delay


@dataclass(frozen=True, slots=True)
class SensorClockModel:
    """Sensor clock correction as a piecewise-linear offset.

    ``corrected = stamp - offset(stamp)``.  Two anchors represent a gradually
    drifting clock; one anchor is a fixed offset.  Outside the anchored stamp
    range the nearest endpoint offset is held constant.
    """

    sensor: str
    anchors: tuple[ClockAnchor, ...] = (ClockAnchor(0.0, 0.0),)

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("sensor clock model requires at least one anchor")
        stamps = [item.stamp for item in self.anchors]
        if stamps != sorted(stamps):
            raise ValueError("clock anchors must be ordered by raw stamp")
        if len(stamps) != len(set(stamps)):
            raise ValueError("clock anchors cannot share a raw stamp")

    @classmethod
    def fixed(cls, sensor: str, delay: float) -> "SensorClockModel":
        return cls(sensor, (ClockAnchor(0.0, -float(delay)),))

    @classmethod
    def linear(
        cls, sensor: str, offset: float = 0.0, drift: float = 0.0
    ) -> "SensorClockModel":
        return cls(
            sensor,
            (
                ClockAnchor(0.0, -offset),
                ClockAnchor(1000.0, 1000.0 - offset - 1000.0 * drift),
            ),
        )

    @classmethod
    def from_points(
        cls, sensor: str, points: Iterable[tuple[float, float]]
    ) -> "SensorClockModel":
        anchors = tuple(
            ClockAnchor(float(stamp), float(corrected)) for stamp, corrected in points
        )
        return cls(sensor, tuple(sorted(anchors, key=lambda item: item.stamp)))

    @property
    def delay(self) -> float:
        return self.offset_at(0.0)

    def offset_at(self, stamp: float) -> float:
        return stamp - self.corrected_time_at(stamp)

    def corrected_time_at(self, stamp: float) -> float:
        if stamp <= self.anchors[0].stamp:
            return stamp - self.anchors[0].offset
        if stamp >= self.anchors[-1].stamp:
            return stamp - self.anchors[-1].offset
        for left, right in zip(self.anchors, self.anchors[1:], strict=True):
            if left.stamp <= stamp <= right.stamp:
                ratio = 0.0 if right.stamp == left.stamp else (stamp - left.stamp) / (
                    right.stamp - left.stamp
                )
                offset = left.offset + ratio * (right.offset - left.offset)
                return stamp - offset
        raise AssertionError("unreachable clock anchor interval")

    def as_dict(self) -> dict[str, object]:
        if len(self.anchors) == 1:
            return {"type": "fixed", "delay": self.anchors[0].offset}
        return {
            "type": "piecewise_linear",
            "anchors": [
                {"stamp": item.stamp, "corrected_time": item.corrected_time}
                for item in self.anchors
            ],
        }


class DelayCalibration:
    """Maps each sensor to its current clock correction."""

    def __init__(
        self,
        models: dict[str, float | SensorClockModel | dict[str, object]] | None = None,
    ) -> None:
        self._models: dict[str, SensorClockModel] = {}
        for sensor, model in (models or {}).items():
            if isinstance(model, dict) and model.get("type") in (
                REBOOT_CLOCK_TYPE,
                "partial_adoption",
            ):
                continue
            if isinstance(model, SensorClockModel):
                clock = SensorClockModel(sensor, model.anchors)
            elif isinstance(model, SensorDelayModel):
                clock = SensorClockModel.fixed(sensor, model.delay)
            elif isinstance(model, (int, float)):
                clock = SensorClockModel.fixed(sensor, float(model))
            else:
                clock = self._model_from_dict(sensor, model)
            self._models[sensor] = clock

        self._reboot_models: dict[str, "object"] = {}
        for sensor, model in (models or {}).items():
            reboot_clock = self._reboot_from_dict_or_value(sensor, model)
            if reboot_clock is not None:
                self._reboot_models[sensor] = reboot_clock

    @staticmethod
    def _model_from_dict(sensor: str, payload: dict[str, object]) -> SensorClockModel:
        kind = payload.get("type", "fixed")
        if kind == "fixed":
            return SensorClockModel.fixed(sensor, float(payload.get("delay", 0.0)))
        if kind == "linear":
            return SensorClockModel.linear(
                sensor,
                float(payload.get("offset", 0.0)),
                float(payload.get("drift", 0.0)),
            )
        if kind != "piecewise_linear":
            raise ValueError(f"unknown clock model type: {kind!r}")
        points = [
            (float(item["stamp"]), float(item["corrected_time"]))
            for item in payload.get("anchors", [])
        ]
        return SensorClockModel.from_points(sensor, points)

    @staticmethod
    def _reboot_from_dict_or_value(sensor: str, model: object):
        from .segments import (
            PartialAdoptionClock,
            RebootBoundary,
            SegmentAnchor,
            SegmentClock,
            SegmentedSensorClock,
        )

        if isinstance(model, PartialAdoptionClock):
            return model
        if isinstance(model, dict) and model.get("type") == "partial_adoption":
            base_payload = model["base"]
            if isinstance(base_payload, dict) and base_payload.get("type") == REBOOT_CLOCK_TYPE:
                base = DelayCalibration._reboot_from_dict_or_value(sensor, base_payload)
            else:
                base = DelayCalibration._model_from_dict(sensor, base_payload)
            candidate = DelayCalibration._reboot_from_dict_or_value(
                sensor, model["candidate"]
            )
            return PartialAdoptionClock(
                sensor,
                base,
                candidate,
                frozenset(int(item) for item in model.get("adopted_sequences", ())),
            )
        if isinstance(model, SegmentedSensorClock):
            return SegmentedSensorClock(sensor, model.segments, model.boundaries)
        if isinstance(model, dict) and model.get("type") == REBOOT_CLOCK_TYPE:
            return SegmentedSensorClock(
                sensor,
                tuple(
                    SegmentClock(
                        int(item["index"]),
                        tuple(
                            SegmentAnchor(
                                float(anchor["stamp"]),
                                float(anchor["corrected_time"]),
                                float(anchor.get("uncertainty", 0.0)),
                            )
                            for anchor in item.get("anchors", [])
                        ),
                    )
                    for item in model.get("segments", [])
                ),
                tuple(
                    RebootBoundary(
                        int(item["after_sequence"]),
                        bool(item.get("confirmed", True)),
                        (
                            None
                            if item.get("reset_window") is None
                            else (
                                float(item["reset_window"][0]),
                                float(item["reset_window"][1]),
                            )
                        ),
                    )
                    for item in model.get("boundaries", [])
                ),
            )
        return None

    def clock_for(self, sensor: str) -> SensorClockModel:
        return self._models.get(sensor, SensorClockModel(sensor))

    def clock_items(self) -> tuple[tuple[str, SensorClockModel], ...]:
        return tuple(self._models.items())

    def delay_for(self, sensor: str) -> float:
        return self.clock_for(sensor).delay

    def offset_for(self, sensor: str, stamp: float) -> float:
        return self.clock_for(sensor).offset_at(stamp)

    def corrected_time(self, observation: Observation) -> float:
        reboot = self._reboot_models.get(observation.sensor)
        if reboot is not None:
            return reboot.corrected_time_at(observation.sequence, observation.time)
        return self.clock_for(observation.sensor).corrected_time_at(observation.time)

    def corrected_range(self, observation: Observation) -> tuple[float, float]:
        """Reliable corrected-time interval; wider than a point when anchors are weak."""
        reboot = self._reboot_models.get(observation.sensor)
        if reboot is not None:
            return reboot.corrected_range(observation.sequence, observation.time)
        point = self.corrected_time(observation)
        return point, point

    def reboot_clock_for(self, sensor: str):
        return self._reboot_models.get(sensor)

    def segment_index_for(self, observation: Observation) -> int | None:
        reboot = self._reboot_models.get(observation.sensor)
        if reboot is None:
            return None
        return reboot.segment_index_for(observation.sequence)

    def with_delay(self, sensor: str, delay: float) -> "DelayCalibration":
        return self.with_clock(SensorClockModel.fixed(sensor, float(delay)))

    def with_sensor_clock(
        self,
        sensor: str,
        *,
        offset: float = 0.0,
        drift: float = 0.0,
        anchors: Iterable[tuple[float, float]] | None = None,
    ) -> "DelayCalibration":
        if anchors is None:
            model = SensorClockModel.linear(sensor, offset, drift)
        else:
            model = SensorClockModel.from_points(sensor, anchors)
        return self.with_clock(model)

    def with_clock(self, model: SensorClockModel) -> "DelayCalibration":
        clone = DelayCalibration()
        clone._models = dict(self._models)
        clone._models[model.sensor] = model
        clone._reboot_models = dict(self._reboot_models)
        clone._reboot_models.pop(model.sensor, None)
        return clone

    def with_reboot_clock(self, model) -> "DelayCalibration":
        """Attach independent per-reboot-segment explanations for one sensor."""
        clone = DelayCalibration()
        clone._models = dict(self._models)
        clone._reboot_models = dict(self._reboot_models)
        clone._reboot_models[model.sensor] = model
        clone._models.pop(model.sensor, None)
        return clone

    def with_reboot_segment(
        self, model, segment_index: int, anchors, boundary=None
    ) -> "DelayCalibration":
        """Update one segment (and optionally one boundary) without touching others."""
        from .segments import SegmentClock, SegmentAnchor, _parse_boundary

        updates = {
            segment_index: (
                anchors
                if isinstance(anchors, SegmentClock)
                else SegmentClock(
                    segment_index,
                    tuple(
                        anchor
                        if isinstance(anchor, SegmentAnchor)
                        else SegmentAnchor(*(float(value) for value in anchor))
                        for anchor in anchors
                    ),
                )
            )
        }
        boundary_updates = None
        if boundary is not None:
            parsed = _parse_boundary(boundary)
            boundary_updates = {segment_index - 1: parsed}
        return self.with_reboot_clock(
            model.replacing(updates, boundary_updates)
        )

    def affected_interval(
        self, sensor: str, observations: list[Observation]
    ) -> tuple[float, float] | None:
        times = [item.time for item in observations if item.sensor == sensor]
        if not times:
            return None
        return min(times), max(times)

    def as_dict(self) -> dict[str, object]:
        result = {sensor: model.as_dict() for sensor, model in self._models.items()}
        result.update(
            {
                sensor: model.as_dict()
                for sensor, model in self._reboot_models.items()
            }
        )
        return result
