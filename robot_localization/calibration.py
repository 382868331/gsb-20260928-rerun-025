from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .types import Observation


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """A known correspondence between a raw stamp and trajectory time."""

    stamp: float
    corrected_time: float
    corrected_time_min: float | None = None
    corrected_time_max: float | None = None

    @property
    def offset(self) -> float:
        return self.stamp - self.corrected_time

    @property
    def corrected_bounds(self) -> tuple[float, float]:
        return (
            self.corrected_time if self.corrected_time_min is None else self.corrected_time_min,
            self.corrected_time if self.corrected_time_max is None else self.corrected_time_max,
        )


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
            payload: dict[str, object] = {"type": "fixed", "delay": self.anchors[0].offset}
            anchor = self.anchors[0]
            if anchor.corrected_time_min is not None or anchor.corrected_time_max is not None:
                payload["corrected_time_min"] = anchor.corrected_time_min
                payload["corrected_time_max"] = anchor.corrected_time_max
            return payload
        return {
            "type": "piecewise_linear",
            "anchors": [
                {
                    "stamp": item.stamp,
                    "corrected_time": item.corrected_time,
                    "corrected_time_min": item.corrected_time_min,
                    "corrected_time_max": item.corrected_time_max,
                }
                for item in self.anchors
            ],
        }


@dataclass(frozen=True, slots=True)
class ClockSegment:
    """Clock evidence for one uninterrupted sensor uptime segment."""

    segment_id: int | str
    anchors: tuple[ClockAnchor, ...]
    first_sequence: int | None = None
    last_sequence: int | None = None
    drift_rate_bounds: tuple[float, float] = (-0.01, 0.01)

    def __post_init__(self) -> None:
        if not self.anchors:
            raise ValueError("clock segment requires at least one anchor")
        stamps = [item.stamp for item in self.anchors]
        if stamps != sorted(stamps) or len(stamps) != len(set(stamps)):
            raise ValueError("segment anchors must be distinct and ordered")
        if self.drift_rate_bounds[0] > self.drift_rate_bounds[1]:
            raise ValueError("drift rate bounds are reversed")

    @property
    def determined(self) -> bool:
        return len(self.anchors) >= 2 and all(
            item.corrected_time_min is None and item.corrected_time_max is None
            for item in self.anchors
        )

    def local_model(self) -> SensorClockModel:
        return SensorClockModel(str(self.segment_id), self.anchors)

    def nominal_time_at(self, stamp: float) -> float:
        return self.local_model().corrected_time_at(stamp)

    def time_bounds_at(self, stamp: float) -> tuple[float, float]:
        """Return the trajectory-time interval supported by this segment.

        Two exact anchors determine the interpolation between them.  Outside
        that interval, or with only one reliable anchor, the configured clock
        rate bounds are retained instead of inventing an exact rate.
        """
        anchor_bounds = [item.corrected_bounds for item in self.anchors]
        exact_everywhere = all(lower == upper for lower, upper in anchor_bounds)
        if len(self.anchors) >= 2 and exact_everywhere:
            first = self.anchors[0]
            last = self.anchors[-1]
            if first.stamp <= stamp <= last.stamp:
                value = self.local_model().corrected_time_at(stamp)
                return value, value
            anchor = first if stamp < first.stamp else last
            delta = stamp - anchor.stamp
        else:
            nearest = min(self.anchors, key=lambda item: abs(item.stamp - stamp))
            anchor = nearest
            delta = stamp - nearest.stamp
        lower_anchor, upper_anchor = anchor.corrected_bounds
        minimum_rate = 1.0 - self.drift_rate_bounds[1]
        maximum_rate = 1.0 - self.drift_rate_bounds[0]
        if delta >= 0.0:
            lower = lower_anchor + delta * minimum_rate
            upper = upper_anchor + delta * maximum_rate
        else:
            lower = lower_anchor + delta * maximum_rate
            upper = upper_anchor + delta * minimum_rate
        return lower, upper

    def contains_sequence(self, sequence: int) -> bool:
        if self.first_sequence is not None and sequence < self.first_sequence:
            return False
        if self.last_sequence is not None and sequence > self.last_sequence:
            return False
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "drift_rate_bounds": list(self.drift_rate_bounds),
            "anchors": [
                {
                    "stamp": item.stamp,
                    "corrected_time": item.corrected_time,
                    "corrected_time_min": item.corrected_time_min,
                    "corrected_time_max": item.corrected_time_max,
                }
                for item in self.anchors
            ],
        }


@dataclass(frozen=True, slots=True)
class RebootBoundary:
    """Known or uncertain reset point between sensor sequence numbers."""

    after_sequence_min: int
    after_sequence_max: int | None = None
    confirmed: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        upper = self.after_sequence_min if self.after_sequence_max is None else self.after_sequence_max
        if upper < self.after_sequence_min:
            raise ValueError("reboot boundary range is reversed")

    @property
    def sequence_bounds(self) -> tuple[int, int]:
        return self.after_sequence_min, self.after_sequence_max or self.after_sequence_min

    def as_dict(self) -> dict[str, object]:
        return {
            "after_sequence_min": self.after_sequence_min,
            "after_sequence_max": self.after_sequence_max,
            "confirmed": self.confirmed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SegmentedSensorClock:
    """Independent clock explanations separated by device reboots."""

    sensor: str
    segments: tuple[ClockSegment, ...]
    boundaries: tuple[RebootBoundary, ...] = field(default_factory=tuple)
    fallback: SensorClockModel | None = None

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segmented clock requires at least one segment")
        ids = [str(item.segment_id) for item in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("clock segment ids must be unique")

    @classmethod
    def from_points(
        cls,
        sensor: str,
        segments: Iterable[
            tuple[int | str, Iterable[tuple[float, float]], int | None, int | None]
            | tuple[int | str, Iterable[tuple[float, float]], int | None]
        ],
        *,
        boundaries: Iterable[RebootBoundary] = (),
        fallback: SensorClockModel | None = None,
    ) -> "SegmentedSensorClock":
        clock_segments: list[ClockSegment] = []
        for index, value in enumerate(segments):
            segment_id, points = value[0], value[1]
            first = value[2] if len(value) > 2 else None
            last = value[3] if len(value) > 3 else None
            anchors = tuple(ClockAnchor(float(stamp), float(corrected)) for stamp, corrected in points)
            clock_segments.append(ClockSegment(segment_id, anchors, first, last))
        return cls(sensor, tuple(clock_segments), tuple(boundaries), fallback)

    def segment_for(
        self,
        stamp: float,
        sequence: int | None = None,
        segment_id: int | str | None = None,
    ) -> ClockSegment:
        if segment_id is not None:
            for item in self.segments:
                if str(item.segment_id) == str(segment_id):
                    return item
            raise ValueError(f"unknown clock segment {segment_id!r}")
        if sequence is not None:
            ranged = [item for item in self.segments if item.contains_sequence(sequence)]
            if ranged:
                if len(ranged) > 1:
                    raise ValueError(f"sequence {sequence} belongs to multiple clock segments")
                return ranged[0]
        # Raw stamps restart from zero; sequence ranges are the safe tiebreaker.
        reset_like = [
            item
            for item in self.segments
            if len(item.anchors) == 1 or item.anchors[0].stamp <= stamp <= item.anchors[-1].stamp
        ]
        if len(reset_like) == 1:
            return reset_like[0]
        raise ValueError("stamp is ambiguous across reboot segments; supply a sequence number")

    def corrected_time_at(
        self,
        stamp: float,
        sequence: int | None = None,
        segment_id: int | str | None = None,
    ) -> float:
        try:
            return self.segment_for(stamp, sequence, segment_id).nominal_time_at(stamp)
        except ValueError:
            if self.fallback is not None:
                return self.fallback.corrected_time_at(stamp)
            raise

    def corrected_bounds_at(
        self,
        stamp: float,
        sequence: int | None = None,
        segment_id: int | str | None = None,
    ) -> tuple[float, float]:
        try:
            return self.segment_for(stamp, sequence, segment_id).time_bounds_at(stamp)
        except ValueError:
            if self.fallback is not None:
                value = self.fallback.corrected_time_at(stamp)
                return value, value
            raise

    def offset_at(
        self,
        stamp: float,
        sequence: int | None = None,
        segment_id: int | str | None = None,
    ) -> float:
        return stamp - self.corrected_time_at(stamp, sequence, segment_id)

    @property
    def delay(self) -> float:
        return self.offset_at(0.0, self.segments[0].first_sequence)

    def with_segment(self, segment: ClockSegment) -> "SegmentedSensorClock":
        segments = [
            item if str(item.segment_id) != str(segment.segment_id) else segment
            for item in self.segments
        ]
        if all(str(item.segment_id) != str(segment.segment_id) for item in self.segments):
            segments.append(segment)
        return SegmentedSensorClock(self.sensor, tuple(segments), self.boundaries, self.fallback)

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "segmented",
            "segments": [item.as_dict() for item in self.segments],
            "boundaries": [item.as_dict() for item in self.boundaries],
            "fallback": self.fallback.as_dict() if self.fallback is not None else None,
        }


class DelayCalibration:
    """Maps each sensor to its current clock correction."""

    def __init__(
        self,
        models: dict[
            str, float | SensorClockModel | SegmentedSensorClock | dict[str, object]
        ] | None = None,
    ) -> None:
        self._models: dict[str, SensorClockModel | SegmentedSensorClock] = {}
        for sensor, model in (models or {}).items():
            if isinstance(model, SensorClockModel):
                clock = SensorClockModel(sensor, model.anchors)
            elif isinstance(model, SegmentedSensorClock):
                clock = SegmentedSensorClock(
                    sensor,
                    tuple(
                        ClockSegment(
                            item.segment_id,
                            tuple(
                                ClockAnchor(
                                    anchor.stamp,
                                    anchor.corrected_time,
                                    anchor.corrected_time_min,
                                    anchor.corrected_time_max,
                                )
                                for anchor in item.anchors
                            ),
                            item.first_sequence,
                            item.last_sequence,
                            item.drift_rate_bounds,
                        )
                        for item in model.segments
                    ),
                    model.boundaries,
                    model.fallback,
                )
            elif isinstance(model, SensorDelayModel):
                clock = SensorClockModel.fixed(sensor, model.delay)
            elif isinstance(model, (int, float)):
                clock = SensorClockModel.fixed(sensor, float(model))
            else:
                clock = self._model_from_dict(sensor, model)
            self._models[sensor] = clock

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
        if kind == "fixed" and (
            "corrected_time_min" in payload or "corrected_time_max" in payload
        ):
            anchor = ClockAnchor(
                0.0,
                -float(payload.get("delay", 0.0)),
                payload.get("corrected_time_min"),
                payload.get("corrected_time_max"),
            )
            return SensorClockModel(sensor, (anchor,))
        if kind != "piecewise_linear":
            if kind != "segmented":
                raise ValueError(f"unknown clock model type: {kind!r}")
            segments = []
            for item in payload.get("segments", []):
                bounds = item.get("drift_rate_bounds", (-0.01, 0.01))
                anchors = tuple(
                    ClockAnchor(
                        float(anchor["stamp"]),
                        float(anchor["corrected_time"]),
                        None if anchor.get("corrected_time_min") is None else float(anchor["corrected_time_min"]),
                        None if anchor.get("corrected_time_max") is None else float(anchor["corrected_time_max"]),
                    )
                    for anchor in item.get("anchors", [])
                )
                segments.append(
                    ClockSegment(
                        item["segment_id"],
                        anchors,
                        item.get("first_sequence"),
                        item.get("last_sequence"),
                        (float(bounds[0]), float(bounds[1])),
                    )
                )
            boundaries = tuple(
                RebootBoundary(
                    int(item["after_sequence_min"]),
                    None if item.get("after_sequence_max") is None else int(item["after_sequence_max"]),
                    bool(item.get("confirmed", True)),
                    str(item.get("reason", "")),
                )
                for item in payload.get("boundaries", [])
            )
            fallback_payload = payload.get("fallback")
            fallback = (
                None
                if fallback_payload is None
                else DelayCalibration._model_from_dict(sensor, fallback_payload)
            )
            if fallback is not None and not isinstance(fallback, SensorClockModel):
                raise ValueError("segmented clock fallback must be a non-segmented clock")
            return SegmentedSensorClock(sensor, tuple(segments), boundaries, fallback)
        points = tuple(
            ClockAnchor(
                float(item["stamp"]),
                float(item["corrected_time"]),
                None if item.get("corrected_time_min") is None else float(item["corrected_time_min"]),
                None if item.get("corrected_time_max") is None else float(item["corrected_time_max"]),
            )
            for item in payload.get("anchors", [])
        )
        return SensorClockModel(sensor, points)

    def clock_for(self, sensor: str) -> SensorClockModel | SegmentedSensorClock:
        return self._models.get(sensor, SensorClockModel(sensor))

    def clock_items(self) -> tuple[tuple[str, SensorClockModel], ...]:
        return tuple(self._models.items())

    def delay_for(self, sensor: str) -> float:
        clock = self.clock_for(sensor)
        return clock.delay

    def offset_for(
        self,
        sensor: str,
        stamp: float,
        sequence: int | None = None,
        segment_id: int | str | None = None,
    ) -> float:
        clock = self.clock_for(sensor)
        if isinstance(clock, SegmentedSensorClock):
            return clock.offset_at(stamp, sequence, segment_id)
        return clock.offset_at(stamp)

    def corrected_time(self, observation: Observation) -> float:
        clock = self.clock_for(observation.sensor)
        if isinstance(clock, SegmentedSensorClock):
            return clock.corrected_time_at(
                observation.time, observation.sequence, observation.clock_segment
            )
        return clock.corrected_time_at(observation.time)

    def corrected_time_bounds(self, observation: Observation) -> tuple[float, float]:
        clock = self.clock_for(observation.sensor)
        if isinstance(clock, SegmentedSensorClock):
            return clock.corrected_bounds_at(
                observation.time, observation.sequence, observation.clock_segment
            )
        value = clock.corrected_time_at(observation.time)
        return value, value

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
        return clone

    def with_segmented_clock(
        self, model: SegmentedSensorClock
    ) -> "DelayCalibration":
        clone = DelayCalibration()
        clone._models = dict(self._models)
        clone._models[model.sensor] = model
        return clone

    def affected_interval(
        self, sensor: str, observations: list[Observation]
    ) -> tuple[float, float] | None:
        times = [item.time for item in observations if item.sensor == sensor]
        if not times:
            return None
        return min(times), max(times)

    def as_dict(self) -> dict[str, object]:
        return {sensor: model.as_dict() for sensor, model in self._models.items()}
