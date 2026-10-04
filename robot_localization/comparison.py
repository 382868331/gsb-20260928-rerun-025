"""Side-by-side comparison of multiple registered clock interpretations.

Filtering and set comparison are read-only views over the interpretation
registry: they never adopt a candidate, never rewrite raw observations and
never touch published (sealed) outputs.  A saved view (filters + selection)
is plain data so it can be checkpointed and resumed after reopening.
"""

from __future__ import annotations

from dataclasses import dataclass


TIME_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Drift assumptions and anchor basis of one selected interpretation."""

    name: str
    sensor: str
    status: str
    note: str
    adopted_sequences: tuple[int, ...] | None
    drift_assumptions: tuple[dict[str, object], ...]
    anchor_basis: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sensor": self.sensor,
            "status": self.status,
            "note": self.note,
            "adopted_sequences": (
                None
                if self.adopted_sequences is None
                else list(self.adopted_sequences)
            ),
            "drift_assumptions": [dict(item) for item in self.drift_assumptions],
            "anchor_basis": [dict(item) for item in self.anchor_basis],
        }


@dataclass(frozen=True, slots=True)
class ObservationAlignment:
    """One overlapping observation aligned across the selected candidates."""

    sensor: str
    sequence: int
    stamp: float
    corrected_times: tuple[float, ...]
    max_time_spread: float
    category: str
    basis_keys: tuple[str, ...]

    def as_dict(self, names: tuple[str, ...]) -> dict[str, object]:
        return {
            "sensor": self.sensor,
            "sequence": self.sequence,
            "stamp": self.stamp,
            "corrected_times": dict(zip(names, self.corrected_times)),
            "max_time_spread": self.max_time_spread,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class InterpretationSetComparison:
    """Aligned result of comparing two or more candidate explanations.

    Observations are split into three disjoint groups: ``time_differs``
    (candidates disagree on the corrected time),
    ``same_time_different_basis`` (same corrected time, different
    drift/anchor basis) and ``identical`` (same time and same basis).
    With ``conflicts_only`` the identical group is withheld from the
    listing but its count stays visible.
    """

    names: tuple[str, ...]
    sensor: str
    candidates: tuple[CandidateSummary, ...]
    time_differs: tuple[ObservationAlignment, ...]
    same_time_different_basis: tuple[ObservationAlignment, ...]
    identical: tuple[ObservationAlignment, ...]
    identical_count: int
    conflicts_only: bool = False

    def as_dict(self) -> dict[str, object]:
        names = self.names
        return {
            "names": list(names),
            "sensor": self.sensor,
            "conflicts_only": self.conflicts_only,
            "candidates": [item.as_dict() for item in self.candidates],
            "groups": {
                "time_differs": [
                    item.as_dict(names) for item in self.time_differs
                ],
                "same_time_different_basis": [
                    item.as_dict(names)
                    for item in self.same_time_different_basis
                ],
                "identical": [item.as_dict(names) for item in self.identical],
            },
            "counts": {
                "time_differs": len(self.time_differs),
                "same_time_different_basis": len(self.same_time_different_basis),
                "identical": self.identical_count,
            },
        }
