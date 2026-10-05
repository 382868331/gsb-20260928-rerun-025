"""Piecewise clock calibration across device reboots.

A reboot resets the sensor clock to zero, so the same raw stamp can occur in
different acquisition segments and must never be treated as the same event.
Each segment carries its own anchors; calibration parameters are never carried
across a reboot boundary.  A boundary may be explicitly recorded or only
suspected within a trajectory-time window, and sparse/contradictory anchors
produce uncertainty ranges instead of invented exact times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .calibration import ClockAnchor, SensorClockModel


@dataclass(frozen=True, slots=True)
class RebootBoundary:
    """A clock reset placed between ``after_sequence`` and the next sequence.

    ``confirmed`` describes an explicit reboot record.  ``reset_window`` is the
    possible reboot position on the trajectory axis when the reset itself was
    not recorded; ``None`` means the boundary location is not bounded yet.
    """

    after_sequence: int
    confirmed: bool = True
    reset_window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.reset_window is not None:
            low, high = self.reset_window
            if high < low - 1e-12:
                raise ValueError("reset window lower bound must precede upper bound")

    def as_dict(self) -> dict[str, object]:
        return {
            "after_sequence": self.after_sequence,
            "confirmed": self.confirmed,
            "reset_window": (
                None if self.reset_window is None else list(self.reset_window)
            ),
        }


@dataclass(frozen=True, slots=True)
class SegmentAnchor:
    """One raw-stamp/trajectory-time correspondence inside a reboot segment.

    ``uncertainty`` is the symmetric half-width of the reliable range on the
    trajectory axis (0 for an exact anchor).
    """

    stamp: float
    corrected_time: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("anchor uncertainty must be non-negative")

    @property
    def offset(self) -> float:
        return self.stamp - self.corrected_time

    @property
    def corrected_range(self) -> tuple[float, float]:
        return (
            self.corrected_time - self.uncertainty,
            self.corrected_time + self.uncertainty,
        )

    def as_clock_anchor(self) -> ClockAnchor:
        return ClockAnchor(self.stamp, self.corrected_time)

    def as_dict(self) -> dict[str, object]:
        return {
            "stamp": self.stamp,
            "corrected_time": self.corrected_time,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class SegmentClock:
    """Independent clock explanation for one acquisition segment."""

    index: int
    anchors: tuple[SegmentAnchor, ...] = ()

    def __post_init__(self) -> None:
        stamps = [item.stamp for item in self.anchors]
        if stamps != sorted(stamps):
            raise ValueError("segment anchors must be ordered by raw stamp")
        if len(stamps) != len(set(stamps)):
            raise ValueError("segment anchors cannot share a raw stamp")

    @property
    def calibrated(self) -> bool:
        return bool(self.anchors)

    @property
    def single_anchor(self) -> bool:
        return len(self.anchors) == 1

    def corrected_range_at(self, stamp: float) -> tuple[float, float]:
        if not self.anchors:
            # Without a segment-specific anchor the position is unknown; the
            # raw stamp is returned only as a labelled guess, never silently
            # reusing another segment's parameters.
            return stamp, stamp
        if len(self.anchors) == 1:
            anchor = self.anchors[0]
            center = stamp - anchor.offset
            return center - anchor.uncertainty, center + anchor.uncertainty
        if stamp <= self.anchors[0].stamp:
            left = right = self.anchors[0]
            ratio = 0.0
        elif stamp >= self.anchors[-1].stamp:
            left = right = self.anchors[-1]
            ratio = 0.0
        else:
            for left, right in zip(self.anchors, self.anchors[1:], strict=True):
                if left.stamp <= stamp <= right.stamp:
                    break
            else:  # pragma: no cover - exhaustive interval search
                raise AssertionError("unreachable segment anchor interval")
            span = right.stamp - left.stamp
            ratio = 0.0 if span == 0.0 else (stamp - left.stamp) / span
        offset = left.offset + ratio * (right.offset - left.offset)
        uncertainty = left.uncertainty + ratio * (
            right.uncertainty - left.uncertainty
        )
        center = stamp - offset
        return center - uncertainty, center + uncertainty

    def corrected_time_at(self, stamp: float) -> float:
        low, high = self.corrected_range_at(stamp)
        return 0.5 * (low + high)

    def offset_at(self, stamp: float) -> float:
        return stamp - self.corrected_time_at(stamp)

    def as_dict(self) -> dict[str, object]:
        return {"anchors": [item.as_dict() for item in self.anchors]}


@dataclass(frozen=True, slots=True)
class SegmentedSensorClock:
    """Per-segment clock explanations separated by reboot boundaries.

    Segment membership is decided by observation sequence relative to the
    reboot boundaries, so repeated raw stamps after a reset never collide.
    """

    sensor: str
    segments: tuple[SegmentClock, ...] = (SegmentClock(0),)
    boundaries: tuple[RebootBoundary, ...] = ()

    def __post_init__(self) -> None:
        indices = [item.index for item in self.segments]
        if indices != list(range(len(self.segments))):
            raise ValueError("segment indices must be contiguous starting at 0")
        cuts = [item.after_sequence for item in self.boundaries]
        if cuts != sorted(cuts) or len(cuts) != len(set(cuts)):
            raise ValueError("reboot boundaries must be ordered and unique")
        if len(self.segments) != len(self.boundaries) + 1:
            raise ValueError("segments must be exactly one more than boundaries")

    @classmethod
    def from_anchors(
        cls,
        sensor: str,
        segment_anchors: Iterable[
            tuple[int, Iterable[tuple[float, float] | tuple[float, float, float]]]
        ],
        boundaries: Iterable[RebootBoundary | tuple[int, ...] | dict[str, object]],
    ) -> "SegmentedSensorClock":
        parsed_boundaries = tuple(_parse_boundary(item) for item in boundaries)
        count = len(parsed_boundaries) + 1
        by_index: dict[int, list[SegmentAnchor]] = {
            index: [] for index in range(count)
        }
        for index, anchors in segment_anchors:
            if index not in by_index:
                raise ValueError(f"anchors reference unknown segment {index}")
            by_index[index] = [_parse_anchor(item) for item in anchors]
        segments = tuple(
            SegmentClock(index, tuple(by_index[index])) for index in range(count)
        )
        return cls(sensor, segments, parsed_boundaries)

    def segment_index_for(self, sequence: int) -> int:
        index = 0
        for boundary in self.boundaries:
            if sequence <= boundary.after_sequence:
                return index
            index += 1
        return index

    def segment_for(self, sequence: int) -> SegmentClock:
        return self.segments[self.segment_index_for(sequence)]

    def boundary_after_segment(self, index: int) -> RebootBoundary | None:
        if index < 0 or index >= len(self.boundaries):
            return None
        return self.boundaries[index]

    def uncalibrated_segments(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.segments if not item.calibrated)

    def single_anchor_segments(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.segments if item.single_anchor)

    def corrected_range(self, sequence: int, stamp: float) -> tuple[float, float]:
        return self.segment_for(sequence).corrected_range_at(stamp)

    def corrected_time_at(self, sequence: int, stamp: float) -> float:
        return self.segment_for(sequence).corrected_time_at(stamp)

    def offset_at(self, sequence: int, stamp: float) -> float:
        return stamp - self.corrected_time_at(sequence, stamp)

    def segment_corrected_spans(self) -> tuple[tuple[float, float] | None, ...]:
        spans: list[tuple[float, float] | None] = []
        for segment in self.segments:
            if not segment.anchors:
                spans.append(None)
                continue
            lows: list[float] = []
            highs: list[float] = []
            for anchor in segment.anchors:
                low, high = anchor.corrected_range
                lows.append(low)
                highs.append(high)
            spans.append((min(lows), max(highs)))
        return tuple(spans)

    def contradictions(self) -> tuple[dict[str, object], ...]:
        """Adjacent segments whose reliable ranges overlap/run backwards.

        Cross-segment anchors are independent evidence.  Contradictions are
        reported and the boundary window stays open instead of forcing a smooth
        transition.
        """

        spans = self.segment_corrected_spans()
        problems: list[dict[str, object]] = []
        for index, boundary in enumerate(self.boundaries):
            before = spans[index]
            after = spans[index + 1]
            if before is None or after is None:
                continue
            if after[0] < before[1] - 1e-9:
                problems.append(
                    {
                        "between_segments": [index, index + 1],
                        "after_sequence": boundary.after_sequence,
                        "previous_range": [before[0], before[1]],
                        "next_range": [after[0], after[1]],
                        "reason": "跨重启锚点互相矛盾：后一段可靠时间范围早于前一段结束位置",
                    }
                )
        return tuple(problems)

    def replacing(
        self,
        segment_updates: dict[int, SegmentClock | Iterable[tuple]],
        boundary_updates: dict[int, RebootBoundary | tuple | dict] | None = None,
    ) -> "SegmentedSensorClock":
        """Return a copy replacing only the listed segments/boundaries.

        Confirming one segment therefore never withdraws a previously
        confirmed neighbour, and no parameters are shared across a reset.
        """

        segments = list(self.segments)
        for index, value in segment_updates.items():
            if not 0 <= index < len(segments):
                raise ValueError(f"cannot update unknown segment {index}")
            segments[index] = (
                value
                if isinstance(value, SegmentClock)
                else SegmentClock(
                    index, tuple(_parse_anchor(item) for item in value)
                )
            )
        boundaries = list(self.boundaries)
        for index, value in (boundary_updates or {}).items():
            if not 0 <= index < len(boundaries):
                raise ValueError(f"cannot update unknown boundary {index}")
            boundaries[index] = _parse_boundary(value)
        return SegmentedSensorClock(self.sensor, tuple(segments), tuple(boundaries))

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "reboot_segments",
            "segments": [
                {"index": item.index, **item.as_dict()} for item in self.segments
            ],
            "boundaries": [item.as_dict() for item in self.boundaries],
        }


@dataclass(frozen=True, slots=True)
class PartialAdoptionClock:
    """A reboot clock explanation adopted only for chosen sequences.

    A late anchor can justify a new explanation for part of the stream while
    every other observation keeps its original version.  Sequences listed in
    ``adopted_sequences`` follow ``candidate``; all others fall back to
    ``base`` (the previously published clock).  Raw stamps, sequences and
    identities are never rewritten.
    """

    sensor: str
    base: object  # SegmentedSensorClock | SensorClockModel
    candidate: "SegmentedSensorClock"
    adopted_sequences: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adopted_sequences", frozenset(self.adopted_sequences)
        )

    def adopts(self, sequence: int) -> bool:
        return sequence in self.adopted_sequences

    def corrected_time_at(self, sequence: int, stamp: float) -> float:
        if self.adopts(sequence):
            return self.candidate.corrected_time_at(sequence, stamp)
        return _base_corrected_time(self.base, sequence, stamp)

    def corrected_range(self, sequence: int, stamp: float) -> tuple[float, float]:
        if self.adopts(sequence):
            return self.candidate.corrected_range(sequence, stamp)
        return _base_corrected_range(self.base, sequence, stamp)

    def offset_at(self, sequence: int, stamp: float) -> float:
        return stamp - self.corrected_time_at(sequence, stamp)

    # Segment introspection delegates to the candidate explanation so range
    # and contradiction reporting stay available for partial candidates.
    @property
    def segments(self) -> tuple[SegmentClock, ...]:
        return self.candidate.segments

    @property
    def boundaries(self) -> tuple[RebootBoundary, ...]:
        return self.candidate.boundaries

    def segment_index_for(self, sequence: int) -> int:
        return self.candidate.segment_index_for(sequence)

    def segment_for(self, sequence: int) -> SegmentClock:
        return self.candidate.segment_for(sequence)

    def boundary_after_segment(self, index: int) -> RebootBoundary | None:
        return self.candidate.boundary_after_segment(index)

    def uncalibrated_segments(self) -> tuple[int, ...]:
        return self.candidate.uncalibrated_segments()

    def single_anchor_segments(self) -> tuple[int, ...]:
        return self.candidate.single_anchor_segments()

    def segment_corrected_spans(self) -> tuple[tuple[float, float] | None, ...]:
        return self.candidate.segment_corrected_spans()

    def contradictions(self) -> tuple[dict[str, object], ...]:
        return self.candidate.contradictions()

    def replacing(self, segment_updates, boundary_updates=None):
        return self.candidate.replacing(segment_updates, boundary_updates)

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "partial_adoption",
            "base": self.base.as_dict(),
            "candidate": self.candidate.as_dict(),
            "adopted_sequences": sorted(self.adopted_sequences),
        }


@dataclass(frozen=True, slots=True)
class CombinationRange:
    """An inclusive raw observation-sequence range taken from one candidate."""

    source_name: str
    first_sequence: int
    last_sequence: int
    clock: object
    anchor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.last_sequence < self.first_sequence:
            raise ValueError("combination range must include at least one sequence")
        object.__setattr__(self, "anchor_ids", tuple(self.anchor_ids))

    def contains(self, sequence: int) -> bool:
        return self.first_sequence <= sequence <= self.last_sequence

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "anchor_ids": list(self.anchor_ids),
        }


@dataclass(frozen=True, slots=True)
class CompositeSegmentClock:
    """Independent calibration assembled from raw sequence ranges.

    No range is merged because its displayed times happen to be close.  Each
    observation is delegated to the explicitly named candidate covering its
    sequence; uncovered observations remain uncalibrated/unknown.
    """

    sensor: str
    fallback: object
    ranges: tuple[CombinationRange, ...]
    conflicts: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranges", tuple(self.ranges))
        ordered = sorted(self.ranges, key=lambda item: item.first_sequence)
        if tuple(ordered) != self.ranges:
            raise ValueError("combination ranges must be ordered by first sequence")
        for left, right in zip(self.ranges, self.ranges[1:], strict=False):
            if right.first_sequence <= left.last_sequence:
                raise ValueError("combination ranges cannot overlap")
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    def range_for(self, sequence: int) -> CombinationRange | None:
        for item in self.ranges:
            if item.contains(sequence):
                return item
        return None

    def clock_for_sequence(self, sequence: int) -> object | None:
        covered = self.range_for(sequence)
        return None if covered is None else covered.clock

    def corrected_time_at(self, sequence: int, stamp: float) -> float:
        clock = self.clock_for_sequence(sequence)
        if clock is None:
            return _base_corrected_time(self.fallback, sequence, stamp)
        return clock.corrected_time_at(sequence, stamp)

    def corrected_range(self, sequence: int, stamp: float) -> tuple[float, float]:
        clock = self.clock_for_sequence(sequence)
        if clock is None:
            return _base_corrected_range(self.fallback, sequence, stamp)
        return clock.corrected_range(sequence, stamp)

    def offset_at(self, sequence: int, stamp: float) -> float:
        return stamp - self.corrected_time_at(sequence, stamp)

    @property
    def segments(self) -> tuple[SegmentClock, ...]:
        clocks = [item.clock for item in self.ranges]
        if not clocks:
            return ()
        return max(clocks, key=lambda item: len(item.segments)).segments

    @property
    def boundaries(self) -> tuple[RebootBoundary, ...]:
        clocks = [item.clock for item in self.ranges]
        return max(clocks, key=lambda item: len(item.boundaries)).boundaries if clocks else ()

    def segment_index_for(self, sequence: int) -> int | None:
        clock = self.clock_for_sequence(sequence)
        if clock is None and isinstance(self.fallback, SegmentedSensorClock):
            return self.fallback.segment_index_for(sequence)
        if clock is None:
            return None
        return clock.segment_index_for(sequence)

    def boundary_after_segment(self, index: int) -> RebootBoundary | None:
        if not 0 <= index < len(self.boundaries):
            return None
        return self.boundaries[index]

    def contradictions(self) -> tuple[dict[str, object], ...]:
        return self.conflicts

    def replacing_range_clock(
        self, first_sequence: int, last_sequence: int, clock: object
    ) -> "CompositeSegmentClock":
        ranges = tuple(
            item
            if not (item.first_sequence == first_sequence and item.last_sequence == last_sequence)
            else CombinationRange(
                item.source_name,
                item.first_sequence,
                item.last_sequence,
                clock,
                item.anchor_ids,
            )
            for item in self.ranges
        )
        return CompositeSegmentClock(self.sensor, self.fallback, ranges, self.conflicts)

    def with_range(
        self, item: CombinationRange, conflicts: tuple[dict[str, object], ...]
    ) -> "CompositeSegmentClock":
        kept = tuple(
            old
            for old in self.ranges
            if not (old.first_sequence <= item.last_sequence and item.first_sequence <= old.last_sequence)
        )
        return CompositeSegmentClock(
            self.sensor,
            self.fallback,
            tuple(sorted((*kept, item), key=lambda value: value.first_sequence)),
            tuple((*self.conflicts, *conflicts)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "segmented_combination",
            "fallback": self.fallback.as_dict(),
            "ranges": [
                {
                    **item.as_dict(),
                    "clock": item.clock.as_dict(),
                }
                for item in self.ranges
            ],
            "conflicts": list(self.conflicts),
        }


def _base_corrected_time(base: object, sequence: int, stamp: float) -> float:
    if isinstance(base, SegmentedSensorClock):
        return base.corrected_time_at(sequence, stamp)
    if isinstance(base, SensorClockModel):
        return base.corrected_time_at(stamp)
    raise TypeError(f"unsupported partial-adoption base clock: {type(base)!r}")


def _base_corrected_range(
    base: object, sequence: int, stamp: float
) -> tuple[float, float]:
    if isinstance(base, SegmentedSensorClock):
        return base.corrected_range(sequence, stamp)
    if isinstance(base, SensorClockModel):
        point = base.corrected_time_at(stamp)
        return point, point
    raise TypeError(f"unsupported partial-adoption base clock: {type(base)!r}")


def _parse_anchor(
    value: tuple[float, float] | tuple[float, float, float] | SegmentAnchor,
) -> SegmentAnchor:
    if isinstance(value, SegmentAnchor):
        return value
    if len(value) == 2:
        stamp, corrected = value
        return SegmentAnchor(float(stamp), float(corrected))
    if len(value) == 3:
        stamp, corrected, uncertainty = value
        return SegmentAnchor(float(stamp), float(corrected), float(uncertainty))
    raise ValueError(
        "segment anchors need (stamp, corrected) or (stamp, corrected, uncertainty)"
    )


def _parse_boundary(
    value: RebootBoundary | tuple[int, ...] | dict[str, object],
) -> RebootBoundary:
    if isinstance(value, RebootBoundary):
        return value
    if isinstance(value, dict):
        window = value.get("reset_window")
        return RebootBoundary(
            int(value["after_sequence"]),
            bool(value.get("confirmed", True)),
            None if window is None else (float(window[0]), float(window[1])),
        )
    if len(value) == 1:
        return RebootBoundary(int(value[0]))
    if len(value) == 2:
        return RebootBoundary(int(value[0]), bool(value[1]))
    if len(value) == 3:
        window = value[2]
        low, high = window[0], window[1]
        return RebootBoundary(int(value[0]), False, (float(low), float(high)))
    raise ValueError(
        "reboot boundary needs a sequence cut, optionally with confidence/window"
    )
