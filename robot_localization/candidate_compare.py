"""Side-by-side comparison of multiple registered clock explanations.

Comparing candidates is a read-only view: it never adopts a candidate,
never rewrites raw observations and never touches published versions.
Filters and selections are plain values so they can be stored in a
checkpoint and resumed after reopening.
"""

from __future__ import annotations

from dataclasses import dataclass

TIME_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class CandidateFilter:
    """Reusable filter over the interpretation registry.

    ``sequence_range`` keeps candidates that cover at least one observation
    whose sequence falls inside the inclusive range.  ``anchor_ids`` keeps
    candidates depending on any of the listed anchor sources.  ``statuses``
    keeps candidates whose interpretation status is listed.
    """

    sequence_range: tuple[int, int] | None = None
    anchor_ids: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None

    def matches(self, interpretation, covered_sequences) -> bool:
        if self.statuses is not None and interpretation.status not in self.statuses:
            return False
        if self.anchor_ids is not None and not (
            set(self.anchor_ids) & set(interpretation.anchor_ids)
        ):
            return False
        if self.sequence_range is not None:
            low, high = self.sequence_range
            if not any(low <= sequence <= high for sequence in covered_sequences):
                return False
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_range": (
                None if self.sequence_range is None else list(self.sequence_range)
            ),
            "anchor_ids": (
                None if self.anchor_ids is None else list(self.anchor_ids)
            ),
            "statuses": None if self.statuses is None else list(self.statuses),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return None
        sequence_range = payload.get("sequence_range")
        anchor_ids = payload.get("anchor_ids")
        statuses = payload.get("statuses")
        return cls(
            None
            if sequence_range is None
            else (int(sequence_range[0]), int(sequence_range[1])),
            None if anchor_ids is None else tuple(str(item) for item in anchor_ids),
            None if statuses is None else tuple(str(item) for item in statuses),
        )


@dataclass(frozen=True, slots=True)
class ComparisonSelection:
    """A saved choice of candidates to compare side by side."""

    names: tuple[str, ...]
    conflicts_only: bool = False

    def __post_init__(self) -> None:
        if len(self.names) < 2:
            raise ValueError("a comparison selection needs at least two candidates")
        if len(set(self.names)) != len(self.names):
            raise ValueError("a comparison selection cannot list a candidate twice")

    def as_dict(self) -> dict[str, object]:
        return {"names": list(self.names), "conflicts_only": self.conflicts_only}

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return None
        return cls(
            tuple(str(item) for item in payload.get("names", ())),
            bool(payload.get("conflicts_only", False)),
        )


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Drift assumptions and anchor basis of one compared candidate."""

    name: str
    sensor: str
    status: str
    note: str
    drift_assumptions: tuple[dict[str, object], ...]
    anchor_basis: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sensor": self.sensor,
            "status": self.status,
            "note": self.note,
            "drift_assumptions": [dict(item) for item in self.drift_assumptions],
            "anchor_basis": [dict(item) for item in self.anchor_basis],
        }


@dataclass(frozen=True, slots=True)
class ObservationAlignment:
    """One overlapping observation aligned across every compared candidate.

    ``category`` is ``time_conflict`` (corrected times differ),
    ``same_time_different_basis`` (same corrected time, different anchor
    basis) or ``identical``.
    """

    sensor: str
    sequence: int
    stamp: float
    corrected_times: tuple[tuple[str, float], ...]
    basis: tuple[tuple[str, tuple], ...]
    category: str

    @property
    def time_spread(self) -> float:
        times = [time for _, time in self.corrected_times]
        return max(times) - min(times)

    def as_dict(self) -> dict[str, object]:
        return {
            "sensor": self.sensor,
            "sequence": self.sequence,
            "stamp": self.stamp,
            "corrected_times": dict(self.corrected_times),
            "basis": {name: list(key) for name, key in self.basis},
            "category": self.category,
            "time_spread": self.time_spread,
        }


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """Side-by-side result for two or more registered candidates.

    Groups are split into ``time_conflicts``, ``same_time_different_basis``
    and ``identical``.  With ``conflicts_only`` the export keeps only the
    conflicting observations while the counts of every group stay visible.
    """

    sensor: str
    candidates: tuple[CandidateSummary, ...]
    time_conflicts: tuple[ObservationAlignment, ...]
    same_time_different_basis: tuple[ObservationAlignment, ...]
    identical: tuple[ObservationAlignment, ...]
    conflicts_only: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.candidates)

    def counts(self) -> dict[str, int]:
        return {
            "time_conflicts": len(self.time_conflicts),
            "same_time_different_basis": len(self.same_time_different_basis),
            "identical": len(self.identical),
        }

    def as_dict(self) -> dict[str, object]:
        groups: dict[str, object] = {
            "time_conflicts": [item.as_dict() for item in self.time_conflicts]
        }
        if not self.conflicts_only:
            groups["same_time_different_basis"] = [
                item.as_dict() for item in self.same_time_different_basis
            ]
            groups["identical"] = [item.as_dict() for item in self.identical]
        return {
            "sensor": self.sensor,
            "candidates": [item.as_dict() for item in self.candidates],
            "conflicts_only": self.conflicts_only,
            "counts": self.counts(),
            "groups": groups,
            "read_only": True,
            "message": "仅比较视图：不采用候选、不修改原始观测、不改写已发布版本",
        }
