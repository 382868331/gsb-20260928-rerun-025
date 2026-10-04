"""Named candidate clock interpretations and late-anchor bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

from .segments import SegmentAnchor


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """A registered anchor with provenance and an optional sequence scope."""

    anchor_id: str
    sensor: str
    segment_index: int
    anchor: SegmentAnchor
    arrival_time: float | None = None
    applies_to: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "sensor": self.sensor,
            "segment_index": self.segment_index,
            "anchor": self.anchor.as_dict(),
            "arrival_time": self.arrival_time,
            "applies_to": (
                None if self.applies_to is None else list(self.applies_to)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AnchorRecord":
        anchor = payload["anchor"]
        applies_to = payload.get("applies_to")
        return cls(
            str(payload["anchor_id"]),
            str(payload["sensor"]),
            int(payload["segment_index"]),
            SegmentAnchor(
                float(anchor["stamp"]),
                float(anchor["corrected_time"]),
                float(anchor.get("uncertainty", 0.0)),
            ),
            payload.get("arrival_time"),
            None if applies_to is None else (int(applies_to[0]), int(applies_to[1])),
        )


@dataclass(frozen=True, slots=True)
class Interpretation:
    """One named clock explanation candidate.

    ``status`` is ``pending`` (comparable, adoptable), ``adopted`` (already
    published) or ``anchor_retracted`` (a dependency was withdrawn; kept for
    traceability but no longer adoptable).
    """

    name: str
    sensor: str
    clock: object  # SegmentedSensorClock | PartialAdoptionClock
    anchor_ids: tuple[str, ...] = ()
    adopted_sequences: tuple[int, ...] | None = None
    note: str = ""
    status: str = "pending"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sensor": self.sensor,
            "clock": self.clock.as_dict(),
            "anchor_ids": list(self.anchor_ids),
            "adopted_sequences": (
                None
                if self.adopted_sequences is None
                else list(self.adopted_sequences)
            ),
            "note": self.note,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Interpretation":
        from .calibration import DelayCalibration

        sensor = str(payload["sensor"])
        clock = DelayCalibration({sensor: payload["clock"]}).reboot_clock_for(
            sensor
        )
        if clock is None:
            raise ValueError(
                "interpretation clock must be a reboot-segment explanation"
            )
        adopted = payload.get("adopted_sequences")
        return cls(
            str(payload["name"]),
            sensor,
            clock,
            tuple(str(item) for item in payload.get("anchor_ids", ())),
            None if adopted is None else tuple(int(item) for item in adopted),
            str(payload.get("note", "")),
            str(payload.get("status", "pending")),
        )


@dataclass(frozen=True, slots=True)
class AnchorRetraction:
    """Result of withdrawing an anchor: every dependent interpretation."""

    anchor_id: str
    dependent_interpretations: tuple[str, ...]
