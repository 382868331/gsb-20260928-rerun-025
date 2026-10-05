"""Composite piecewise clock schemes built from registered candidates.

A single sensor can need different clock explanations before and after a
reboot, so adopting one whole candidate cannot express locally trusted
results.  A :class:`CompositeScheme` composes registered interpretations by
*original observation sequence ranges*: each piece keeps the provenance of
the candidate it was taken from, and boundaries between pieces record both
neighbouring sources explicitly.  Pieces are never merged just because their
currently displayed corrected times look close, and raw observations,
sequences and identities are never rewritten.

Before a scheme is usable its pieces are checked for intra-segment time
monotonicity, reboot breakpoints hidden inside a piece and shared-anchor
consistency.  Contradictions are reported with the exact sequence interval
and the observations/anchors that cause them; nothing is smoothed over.
"""

from __future__ import annotations

from dataclasses import dataclass

from .segments import RebootBoundary, SegmentAnchor, SegmentClock, SegmentedSensorClock


@dataclass(frozen=True, slots=True)
class CompositionPiece:
    """One original-observation range explained by one registered candidate.

    `source` is the interpretation name the explanation was taken from;
    `anchors` are copied from the source candidate segments covering
    `sequence_range` and keep their own provenance via `anchor_ids`.
    `confirmed` marks basis that was already verified and is retained
    across later local re-checks.
    """

    index: int
    source: str
    sequence_range: tuple[int, int]
    anchors: tuple[SegmentAnchor, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    confirmed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source": self.source,
            "sequence_range": list(self.sequence_range),
            "anchors": [item.as_dict() for item in self.anchors],
            "anchor_ids": list(self.anchor_ids),
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CompositionPiece":
        return cls(
            int(payload["index"]),
            str(payload["source"]),
            (int(payload["sequence_range"][0]), int(payload["sequence_range"][1])),
            tuple(
                SegmentAnchor(
                    float(item["stamp"]),
                    float(item["corrected_time"]),
                    float(item.get("uncertainty", 0.0)),
                )
                for item in payload.get("anchors", [])
            ),
            tuple(str(item) for item in payload.get("anchor_ids", ())),
            bool(payload.get("confirmed", False)),
        )


@dataclass(frozen=True, slots=True)
class CompositionBoundary:
    """Break between two pieces with explicit provenance on both sides.

    The boundary exists because the *source candidates* change (or a reboot
    was recorded), never because two observations happen to show nearby
    corrected times.
    """

    after_sequence: int
    left_source: str
    right_source: str
    confirmed: bool = False
    reset_window: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "after_sequence": self.after_sequence,
            "left_source": self.left_source,
            "right_source": self.right_source,
            "confirmed": self.confirmed,
            "reset_window": (
                None if self.reset_window is None else list(self.reset_window)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CompositionBoundary":
        window = payload.get("reset_window")
        return cls(
            int(payload["after_sequence"]),
            str(payload["left_source"]),
            str(payload["right_source"]),
            bool(payload.get("confirmed", False)),
            None if window is None else (float(window[0]), float(window[1])),
        )


@dataclass(frozen=True, slots=True)
class CompositionContradiction:
    """One concrete conflict found while composing, with its evidence.

    `kind` is one of `range_overlap`, `non_monotonic_stamp`,
    `non_monotonic_corrected`, `reboot_inside_piece` or
    `shared_anchor_conflict`.  `sequence_interval` locates the conflict
    in original observation sequence numbers and `evidence` lists the
    observations/anchors/boundaries that cause it.
    """

    kind: str
    sequence_interval: tuple[int, int]
    reason: str
    evidence: tuple[dict[str, object], ...]
    piece_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sequence_interval": list(self.sequence_interval),
            "reason": self.reason,
            "evidence": [dict(item) for item in self.evidence],
            "piece_indices": list(self.piece_indices),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CompositionContradiction":
        return cls(
            str(payload["kind"]),
            (
                int(payload["sequence_interval"][0]),
                int(payload["sequence_interval"][1]),
            ),
            str(payload["reason"]),
            tuple(dict(item) for item in payload.get("evidence", ())),
            tuple(int(item) for item in payload.get("piece_indices", ())),
        )


@dataclass(frozen=True, slots=True)
class CompositeScheme:
    """An independent piecewise calibration plan composed from candidates.

    `status` is `pending` (comparable, adoptable when free of
    contradictions) or `adopted` (published as a new version).  Multiple
    pending schemes may coexist; adoption is always explicit.
    """

    name: str
    sensor: str
    pieces: tuple[CompositionPiece, ...]
    boundaries: tuple[CompositionBoundary, ...]
    contradictions: tuple[CompositionContradiction, ...] = ()
    status: str = "pending"
    note: str = ""

    def piece_for_sequence(self, sequence: int) -> CompositionPiece | None:
        for piece in self.pieces:
            low, high = piece.sequence_range
            if low <= sequence <= high:
                return piece
        return None

    def unknown_ranges(self) -> tuple[tuple[int, int], ...]:
        """Original sequence ranges no piece covers; they stay unknown."""
        gaps: list[tuple[int, int]] = []
        ordered = sorted(self.pieces, key=lambda item: item.sequence_range)
        cursor: int | None = None
        for piece in ordered:
            low, high = piece.sequence_range
            if cursor is not None and low > cursor + 1:
                gaps.append((cursor + 1, low - 1))
            cursor = high if cursor is None else max(cursor, high)
        return tuple(gaps)

    def as_clock(self) -> SegmentedSensorClock:
        """Materialise the scheme as an independent segmented clock.

        Uncovered ranges become anchor-less segments (explicitly unknown);
        no parameters are borrowed from neighbouring pieces.
        """
        ordered = sorted(self.pieces, key=lambda item: item.sequence_range)
        boundary_by_cut = {item.after_sequence: item for item in self.boundaries}
        segments: list[SegmentClock] = []
        boundaries: list[RebootBoundary] = []
        for position, piece in enumerate(ordered):
            segments.append(SegmentClock(len(segments), piece.anchors))
            if position + 1 >= len(ordered):
                break
            low_next = ordered[position + 1].sequence_range[0]
            high_here = piece.sequence_range[1]
            record = boundary_by_cut.get(high_here)
            confirmed = record.confirmed if record is not None else False
            window = record.reset_window if record is not None else None
            boundaries.append(RebootBoundary(high_here, confirmed, window))
            if low_next > high_here + 1:
                segments.append(SegmentClock(len(segments), ()))
                boundaries.append(RebootBoundary(low_next - 1, False, None))
        return SegmentedSensorClock(self.sensor, tuple(segments), tuple(boundaries))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sensor": self.sensor,
            "pieces": [item.as_dict() for item in self.pieces],
            "boundaries": [item.as_dict() for item in self.boundaries],
            "contradictions": [item.as_dict() for item in self.contradictions],
            "status": self.status,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CompositeScheme":
        return cls(
            str(payload["name"]),
            str(payload["sensor"]),
            tuple(
                CompositionPiece.from_dict(item)
                for item in payload.get("pieces", [])
            ),
            tuple(
                CompositionBoundary.from_dict(item)
                for item in payload.get("boundaries", [])
            ),
            tuple(
                CompositionContradiction.from_dict(item)
                for item in payload.get("contradictions", [])
            ),
            str(payload.get("status", "pending")),
            str(payload.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class OrderingChange:
    """One cross-sensor observation pair whose order differs between schemes."""

    first: tuple[str, int]
    second: tuple[str, int]
    orders: tuple[tuple[str, str], ...]  # (scheme name, "before"/"after"/"same")

    def as_dict(self) -> dict[str, object]:
        return {
            "first": list(self.first),
            "second": list(self.second),
            "orders": {name: order for name, order in self.orders},
        }


@dataclass(frozen=True, slots=True)
class CompositionOrderingComparison:
    """Side-by-side effect of pending schemes on cross-sensor event order."""

    names: tuple[str, ...]
    sensor: str
    changes: tuple[OrderingChange, ...]
    event_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "sensor": self.sensor,
            "changes": [item.as_dict() for item in self.changes],
            "event_counts": {name: count for name, count in self.event_counts},
        }


@dataclass(frozen=True, slots=True)
class CompositionRecheck:
    """Result of a local re-anchor: only affected ranges were re-checked."""

    name: str
    rechecked_ranges: tuple[tuple[int, int], ...]
    retained_ranges: tuple[tuple[int, int], ...]
    contradictions: tuple[CompositionContradiction, ...]
    unknown_ranges: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rechecked_ranges": [list(item) for item in self.rechecked_ranges],
            "retained_ranges": [list(item) for item in self.retained_ranges],
            "contradictions": [item.as_dict() for item in self.contradictions],
            "unknown_ranges": [list(item) for item in self.unknown_ranges],
        }
