from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from dataclasses import replace
from pathlib import Path

from .calibration import DelayCalibration, SensorClockModel
from .comparison import (
    TIME_TOLERANCE,
    CandidateSummary,
    InterpretationSetComparison,
    ObservationAlignment,
)
from .compositions import (
    CompositionBoundary,
    CompositionContradiction,
    CompositionOrderingComparison,
    CompositionPiece,
    CompositionRecheck,
    CompositeScheme,
    OrderingChange,
)
from .estimator import LocalizationEstimator, StateEstimate
from .interpretations import AnchorRecord, AnchorRetraction, Interpretation
from .segments import (
    PartialAdoptionClock,
    RebootBoundary,
    SegmentAnchor,
    SegmentClock,
    SegmentedSensorClock,
)
from .types import Observation


@dataclass(slots=True)
class Checkpoint:
    state: StateEstimate
    event_count: int
    arrival_horizon: float | None = None
    seal_time: float | None = None
    calibration: DelayCalibration | None = None
    visible_observations: list[Observation] = field(default_factory=list)
    sealed_outputs: list[StateEstimate] = field(default_factory=list)
    rejected_observations: dict[tuple[str, int], str] = field(default_factory=dict)
    staged_calibration: DelayCalibration | None = None
    staged_sensor: str | None = None
    anchor_records: list = field(default_factory=list)
    pending_interpretations: list = field(default_factory=list)
    comparison_view: dict | None = None
    compositions: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RecomputeReport:
    sensor: str
    old_delay: float
    new_delay: float
    affected_from: float
    affected_to: float
    recomputed_from: float
    sealed_blocked: tuple[float, ...]
    boundary_state_time: float | None = None
    adopted_observations: tuple[int, ...] = ()
    rejected_observations: tuple["ObservationDecision", ...] = ()
    timeline_changes: tuple["TimelinePositionChange", ...] = ()
    uncertain_intervals: tuple["UncertainInterval", ...] = ()


@dataclass(frozen=True, slots=True)
class ObservationDecision:
    sensor: str
    sequence: int
    stamp: float
    arrival_time: float | None
    baseline_corrected_time: float
    corrected_time: float
    time_shift: float
    classification: str
    reason: str
    segment_index: int | None = None
    baseline_range: tuple[float, float] | None = None
    candidate_range: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class SegmentView:
    """Per-reboot-segment summary inside a staged clock explanation."""

    index: int
    sequences: tuple[int, ...]
    stamp_span: tuple[float, float] | None
    corrected_span: tuple[float, float] | None
    anchor_count: int
    single_anchor: bool
    confirmed_boundary_before: bool
    reset_window: tuple[float, float] | None
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class DelaySettingView:
    delay: float
    corrected_interval: tuple[float, float] | None
    affected_sequences: tuple[int, ...]
    open_sequences: tuple[int, ...]
    sealed_sequences: tuple[int, ...]
    rejected_sequences: tuple[int, ...]
    clock: object | None = None


@dataclass(frozen=True, slots=True)
class DelayComparison:
    sensor: str
    baseline: DelaySettingView
    candidate: DelaySettingView
    recomputable_range: tuple[float, float] | None
    sealed_history: tuple[float, float] | None
    boundary_state_time: float | None
    decisions: tuple[ObservationDecision, ...]
    timeline: tuple["TimelinePosition", ...]
    position_changes: tuple["TimelinePositionChange", ...]
    uncertain_intervals: tuple["UncertainInterval", ...]
    segment_views: tuple[SegmentView, ...] = ()
    reboot_contradictions: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class InterpretationComparisonEntry:
    """One registered candidate explanation next to its live comparison."""

    name: str
    sensor: str
    status: str
    note: str
    anchor_ids: tuple[str, ...]
    adopted_sequences: tuple[int, ...] | None
    comparison: DelayComparison


@dataclass(frozen=True, slots=True)
class TimelinePosition:
    identity: tuple[str, int]
    sensor: str
    sequence: int
    stamp: float
    arrival_time: float | None
    baseline_time: float
    candidate_time: float
    time_shift: float
    baseline_order: int
    candidate_order: int
    order_delta: int
    classification: str
    reason: str
    segment_index: int | None = None


@dataclass(frozen=True, slots=True)
class TimelinePositionChange:
    identity: tuple[str, int]
    baseline_time: float
    candidate_time: float
    baseline_x: float | None = None
    baseline_y: float | None = None
    candidate_x: float | None = None
    candidate_y: float | None = None


@dataclass(frozen=True, slots=True)
class UncertainInterval:
    start: float
    end: float
    reason: str


class SealedHistoryError(RuntimeError):
    """Raised when a delay change would silently overwrite sealed outputs."""

    def __init__(
        self,
        sensor: str,
        affected_interval: tuple[float, float],
        sealed_times: list[float],
    ) -> None:
        self.sensor = sensor
        self.affected_interval = affected_interval
        self.sealed_times = sealed_times
        super().__init__(
            f"cannot recompute sensor {sensor!r} interval "
            f"[{affected_interval[0]:.3f}, {affected_interval[1]:.3f}]: "
            f"{len(sealed_times)} historical outputs are sealed"
        )


class LocalizationPipeline:
    """Causal event processor with delay calibration and sealed history.

    Each observation keeps its stamped ``time`` and its wall-clock
    ``arrival_time`` separately.  Fusion uses the *corrected* time
    (stamp plus the per-sensor fixed delay), while visibility is decided by
    arrival time: a burst of backfill after an outage only revises history
    and never moves the live estimate backwards to an old timestamp.
    """

    def __init__(
        self,
        estimator: LocalizationEstimator | None = None,
        calibration: DelayCalibration | None = None,
    ) -> None:
        self.estimator = estimator or LocalizationEstimator()
        self.calibration = calibration or DelayCalibration()
        self.received: list[Observation] = []
        self.checkpoints: list[Checkpoint] = []
        self.arrival_horizon: float | None = None
        self.seal_time: float | None = None
        self.sealed_outputs: list[StateEstimate] = []
        self.outputs: list[StateEstimate] = []
        self.rejected_observations: dict[tuple[str, int], str] = {}
        self.staged_calibration: DelayCalibration | None = None
        self.staged_comparison: DelayComparison | None = None
        self.staged_sensor: str | None = None
        self.anchor_records: dict[str, AnchorRecord] = {}
        self.interpretations: dict[str, Interpretation] = {}
        self.interpretation_comparisons: dict[str, DelayComparison] = {}
        self.comparison_view: dict | None = None
        self.compositions: dict[str, CompositeScheme] = {}

    # ------------------------------------------------------------------ ingest

    def add_observation(self, observation: Observation) -> None:
        self.received.append(observation)
        arrival = self.arrival_horizon
        if arrival is not None and observation.observed_at() <= arrival + 1e-9:
            if (
                self.seal_time is not None
                and self.calibration.corrected_time(observation) <= self.seal_time + 1e-9
            ):
                key = (observation.sensor, observation.sequence)
                self.rejected_observations.setdefault(
                    key,
                    "观测在封存边界之后到达，但按当前延迟校正到边界之前，不能改写封存历史",
                )
            else:
                self._rebuild_tail()

    def process_until(self, arrival_horizon: float) -> list[StateEstimate]:
        if (
            self.arrival_horizon is not None
            and arrival_horizon < self.arrival_horizon - 1e-9
        ):
            raise ValueError("offline clock cannot move backwards")
        self.arrival_horizon = arrival_horizon
        events = self._eligible(arrival_horizon)
        self._rebuild_tail(events)
        # Newly visible events (backfill) can carry old corrected times, so
        # history is rebuilt causally; the live estimate then predicts forward
        # to the arrival clock and never jumps backwards to an old timestamp.
        self.estimator.predict(arrival_horizon)
        self._trim_checkpoints()
        self.checkpoints.append(
            Checkpoint(
                self.estimator.snapshot(),
                len(events),
                arrival_horizon,
                self.seal_time,
                self.calibration,
                self._visible(arrival_horizon),
                [self._clone_state(item) for item in self.sealed_outputs],
                dict(self.rejected_observations),
                self.staged_calibration,
                self.staged_sensor,
                list(self.anchor_records.values()),
                list(self.interpretations.values()),
                self.comparison_view,
                list(self.compositions.values()),
            )
        )
        return list(self.outputs)

    # ------------------------------------------------------------------ sealing

    def seal_until(self, time: float) -> list[StateEstimate]:
        """Freeze every output at or before ``time``; those values never change."""
        if self.seal_time is not None and time < self.seal_time - 1e-9:
            raise ValueError("seal boundary cannot move backwards")
        if self.seal_time is None:
            boundary = self._boundary_state(time, LocalizationEstimator())
            self.sealed_outputs = [
                output
                for output in self.all_outputs()
                if output.time <= time + 1e-9
            ]
            if not self.sealed_outputs or boundary.time > self.sealed_outputs[-1].time + 1e-9:
                self.sealed_outputs.append(boundary)
        else:
            newly_sealed = [
                output for output in self.outputs if output.time <= time + 1e-9
            ]
            self.sealed_outputs.extend(newly_sealed)
            if not self.sealed_outputs or self.sealed_outputs[-1].time < time - 1e-9:
                boundary = self._boundary_state(
                    time,
                    self._estimator_restored(self.sealed_outputs[-len(newly_sealed) - 1]),
                )
                self.sealed_outputs.append(boundary)
        self.seal_time = self.sealed_outputs[-1].time
        self.outputs = [
            output for output in self.outputs if output.time > self.seal_time + 1e-9
        ]
        self._restore_anchor_and_rebuild()
        self._trim_checkpoints()
        self.checkpoints.append(
            Checkpoint(
                self.sealed_outputs[-1] if self.sealed_outputs else self.estimator.snapshot(),
                len(self.sealed_outputs),
                self.arrival_horizon,
                self.seal_time,
                self.calibration,
                self._visible(self.arrival_horizon),
                [self._clone_state(item) for item in self.sealed_outputs],
                dict(self.rejected_observations),
                self.staged_calibration,
                self.staged_sensor,
                list(self.anchor_records.values()),
                list(self.interpretations.values()),
                self.comparison_view,
                list(self.compositions.values()),
            )
        )
        return list(self.sealed_outputs)

    def stage_sensor_delay(self, sensor: str, delay: float) -> DelayComparison:
        """Prepare a delay revision without publishing it."""
        delay = float(delay)
        candidate = self.calibration.with_delay(sensor, delay)
        return self._stage_calibration(sensor, candidate)

    def stage_sensor_clock(
        self,
        sensor: str,
        *,
        offset: float = 0.0,
        drift: float = 0.0,
        anchors: list[tuple[float, float]] | None = None,
    ) -> DelayComparison:
        """Prepare a drifting-clock revision without publishing it."""
        candidate = self.calibration.with_sensor_clock(
            sensor, offset=offset, drift=drift, anchors=anchors
        )
        return self._stage_calibration(sensor, candidate)

    def stage_clock_model(self, model: SensorClockModel) -> DelayComparison:
        return self._stage_calibration(model.sensor, self.calibration.with_clock(model))

    def stage_reboot_segments(
        self,
        sensor: str,
        *,
        boundaries: list[RebootBoundary | tuple | dict],
        anchors: dict[int, list[tuple]] | None = None,
    ) -> DelayComparison:
        """Stage a complete multi-reboot explanation for comparison.

        Every segment starts with its own anchors; nothing is inherited from
        a pre-reboot clock.  The candidate is not published until confirmed.
        """

        model = SegmentedSensorClock.from_anchors(
            sensor, (anchors or {}).items(), boundaries
        )
        return self._stage_calibration(
            sensor, self.calibration.with_reboot_clock(model)
        )

    def stage_reboot_segment(
        self,
        sensor: str,
        segment_index: int,
        anchors: list[tuple],
        *,
        boundary: RebootBoundary | tuple | dict | None = None,
    ) -> DelayComparison:
        """Stage an explanation for one acquisition segment only.

        Other confirmed segments remain exactly as they were.  Withdrawing an
        unconfirmed segment therefore never undoes an earlier confirmation.
        """

        model = self.calibration.reboot_clock_for(sensor)
        if model is None:
            raise ValueError(
                f"sensor {sensor!r} has no reboot segments; stage boundaries first"
            )
        if not 0 <= segment_index < len(model.segments):
            raise ValueError(f"unknown reboot segment {segment_index}")
        candidate = self.calibration.with_reboot_segment(
            model, segment_index, anchors, boundary
        )
        return self._stage_calibration(sensor, candidate)

    def confirm_reboot_segment(
        self,
        sensor: str,
        segment_index: int,
        anchors: list[tuple],
        *,
        boundary: RebootBoundary | tuple | dict | None = None,
    ) -> RecomputeReport:
        """Publish one segment; only its still-open affected interval re-estimates."""
        self.stage_reboot_segment(
            sensor, segment_index, anchors, boundary=boundary
        )
        return self.publish_staged_delay()

    def _stage_calibration(
        self, sensor: str, candidate: DelayCalibration
    ) -> DelayComparison:
        comparison = self.compare_calibrations(sensor, self.calibration, candidate)
        self.staged_calibration = candidate
        self.staged_comparison = comparison
        self.staged_sensor = sensor
        return comparison

    def discard_staged_delay(self) -> None:
        """Discard a delay revision that has not been published."""
        self.staged_calibration = None
        self.staged_comparison = None
        self.staged_sensor = None

    def publish_staged_delay(self) -> RecomputeReport:
        if self.staged_calibration is None or self.staged_comparison is None:
            raise RuntimeError("no staged delay revision to publish")
        self.staged_comparison = self.compare_calibrations(
            self.staged_sensor or "", self.calibration, self.staged_calibration
        )
        comparison = self.staged_comparison
        old_delay = self.calibration.delay_for(comparison.sensor)
        new_delay = self.staged_calibration.delay_for(comparison.sensor)
        self.calibration = self.staged_calibration
        for decision in comparison.decisions:
            if decision.classification == "crossing_rejected":
                self.rejected_observations[(decision.sensor, decision.sequence)] = decision.reason
        staged_sensor = comparison.sensor
        self.discard_staged_delay()
        self._restore_anchor_and_rebuild()
        interval = comparison.baseline.corrected_interval
        return RecomputeReport(
            comparison.sensor,
            old_delay,
            new_delay,
            interval[0] if interval else 0.0,
            interval[1] if interval else 0.0,
            comparison.recomputable_range[0] if comparison.recomputable_range else 0.0,
            tuple(),
            comparison.boundary_state_time,
            comparison.candidate.open_sequences,
            tuple(
                decision
                for decision in comparison.decisions
                if decision.classification == "crossing_rejected"
            ),
            comparison.position_changes,
            comparison.uncertain_intervals,
        )

    def set_sensor_delay(self, sensor: str, delay: float) -> RecomputeReport:
        """Publish a fixed-delay update and recompute only the open interval."""
        self.stage_sensor_delay(sensor, delay)
        return self.publish_staged_delay()

    def compare_sensor_delays(self, sensor: str, candidate_delay: float) -> DelayComparison:
        candidate = self.calibration.with_delay(sensor, candidate_delay)
        return self.compare_calibrations(sensor, self.calibration, candidate)

    def compare_calibrations(
        self,
        sensor: str,
        baseline: DelayCalibration,
        candidate: DelayCalibration,
    ) -> DelayComparison:
        visible = self._visible(self.arrival_horizon)
        affected = [
            item
            for item in visible
            if item.sensor == sensor
        ]
        decisions = tuple(
            self._observation_decision(item, baseline, candidate)
            for item in affected
        )
        baseline_view = self._setting_view(sensor, affected, baseline)
        candidate_view = self._setting_view(sensor, affected, candidate, baseline)
        accepted_classes = {"open_recomputed", "open_adopted"}
        open_times = [
            item.corrected_time
            for item in decisions
            if item.classification in accepted_classes
        ]
        if open_times:
            recomputable_range = (min(open_times), max(open_times))
        elif self.seal_time is not None:
            recomputable_range = (self.seal_time, self.seal_time)
        else:
            recomputable_range = None
        sealed_history = (0.0, self.seal_time) if self.seal_time is not None else None
        timeline, position_changes = self._compare_timeline_positions(
            visible, affected, baseline, candidate, decisions
        )
        uncertain_intervals = self._open_gaps(candidate)
        segment_views = self._segment_views(sensor, affected, candidate)
        candidate_reboot = candidate.reboot_clock_for(sensor)
        contradictions = (
            candidate_reboot.contradictions() if candidate_reboot is not None else ()
        )
        return DelayComparison(
            sensor,
            baseline_view,
            candidate_view,
            recomputable_range,
            sealed_history,
            self.sealed_outputs[-1].time if self.sealed_outputs else self.seal_time,
            decisions,
            timeline,
            position_changes,
            uncertain_intervals,
            segment_views,
            contradictions,
        )

    # ------------------------------------------- interpretation registry

    def register_anchor(
        self,
        anchor_id: str,
        sensor: str,
        segment_index: int,
        anchor,
        *,
        arrival_time: float | None = None,
        applies_to: tuple[int, int] | None = None,
        source: str = "",
    ) -> AnchorRecord:
        """Register an anchor with provenance; never changes the live clock."""
        if anchor_id in self.anchor_records:
            raise ValueError(f"anchor {anchor_id!r} is already registered")
        parsed = (
            anchor
            if isinstance(anchor, SegmentAnchor)
            else SegmentAnchor(*(float(value) for value in anchor))
        )
        record = AnchorRecord(
            anchor_id,
            sensor,
            segment_index,
            parsed,
            arrival_time,
            None if applies_to is None else (int(applies_to[0]), int(applies_to[1])),
            source,
        )
        self.anchor_records[anchor_id] = record
        return record

    def register_interpretation(
        self,
        name: str,
        sensor: str,
        *,
        boundaries: list,
        anchors: dict[int, list] | None = None,
        anchor_ids: tuple[str, ...] = (),
        applies_to: tuple[int, int] | None = None,
        note: str = "",
    ) -> DelayComparison:
        """Register one boundary/drift explanation and compare it openly.

        The candidate is only stored for comparison; the live calibration
        and every previously registered candidate stay untouched, so the
        newest registration never auto-overrides anything.
        """
        model = SegmentedSensorClock.from_anchors(
            sensor, (anchors or {}).items(), boundaries
        )
        return self._register_interpretation(
            name,
            sensor,
            model,
            anchor_ids=anchor_ids,
            applies_to=applies_to,
            note=note,
        )

    def register_late_anchor(
        self,
        anchor_id: str,
        sensor: str,
        segment_index: int,
        anchor,
        *,
        applies_to: tuple[int, int],
        arrival_time: float | None = None,
        name: str | None = None,
        note: str = "",
        source: str = "",
    ) -> DelayComparison:
        """Add a late anchor as a partial-adoption interpretation.

        Only observations whose sequence falls inside ``applies_to`` follow
        the new explanation; every other observation keeps its original
        version.  Raw stamps (including duplicates across a reset) and
        identities are unchanged.
        """
        self.register_anchor(
            anchor_id,
            sensor,
            segment_index,
            anchor,
            arrival_time=arrival_time,
            applies_to=applies_to,
            source=source,
        )
        model = self.calibration.reboot_clock_for(sensor)
        if isinstance(model, PartialAdoptionClock):
            model = model.candidate
        if model is None:
            raise ValueError(
                f"sensor {sensor!r} has no reboot segments to attach a late anchor to"
            )
        if not 0 <= segment_index < len(model.segments):
            raise ValueError(f"unknown reboot segment {segment_index}")
        parsed = (
            anchor
            if isinstance(anchor, SegmentAnchor)
            else SegmentAnchor(*(float(value) for value in anchor))
        )
        segment = model.segments[segment_index]
        merged = tuple(sorted([*segment.anchors, parsed], key=lambda item: item.stamp))
        updated = model.replacing({segment_index: SegmentClock(segment_index, merged)})
        interp_name = name or f"late-anchor:{anchor_id}"
        note = note or (
            f"迟到锚点 {anchor_id} 只应用于序号范围 {applies_to}，其余观测维持原版本"
        )
        return self._register_interpretation(
            interp_name,
            sensor,
            updated,
            anchor_ids=(anchor_id,),
            applies_to=applies_to,
            note=note,
        )

    def interpretation_comparison(self, name: str) -> DelayComparison:
        return self.interpretation_comparisons[self._require_interpretation(name).name]

    def compare_interpretations(
        self, sensor: str | None = None
    ) -> tuple[InterpretationComparisonEntry, ...]:
        """List every registered candidate next to its current comparison.

        Pending comparisons are refreshed against the live calibration so
        corrected-time ranges and downstream trajectory effects stay
        traceable for each explanation.
        """
        entries: list[InterpretationComparisonEntry] = []
        for name, interp in self.interpretations.items():
            if sensor is not None and interp.sensor != sensor:
                continue
            if interp.status == "pending":
                self.interpretation_comparisons[name] = self.compare_calibrations(
                    interp.sensor,
                    self.calibration,
                    self.calibration.with_reboot_clock(interp.clock),
                )
            entries.append(
                InterpretationComparisonEntry(
                    name,
                    interp.sensor,
                    interp.status,
                    interp.note,
                    interp.anchor_ids,
                    interp.adopted_sequences,
                    self.interpretation_comparisons[name],
                )
            )
        return tuple(entries)

    # ------------------------------------- candidate filtering and set views

    def filter_interpretations(
        self,
        *,
        sensor: str | None = None,
        sequence_range: tuple[int, int] | list[int] | None = None,
        anchor_source: str | None = None,
        status: str | None = None,
    ) -> tuple[InterpretationComparisonEntry, ...]:
        """Filter candidates by observation range, anchor source and status.

        Read-only: nothing is adopted, rewritten or published here.
        """
        entries = self.compare_interpretations(sensor)
        selected: list[InterpretationComparisonEntry] = []
        for entry in entries:
            interp = self.interpretations[entry.name]
            if status is not None and interp.status != status:
                continue
            if anchor_source is not None:
                sources = {
                    self.anchor_records[anchor_id].source
                    for anchor_id in interp.anchor_ids
                    if anchor_id in self.anchor_records
                }
                if anchor_source not in sources:
                    continue
            if sequence_range is not None:
                low, high = int(sequence_range[0]), int(sequence_range[1])
                scope = interp.adopted_sequences
                if scope is None:
                    scope = entry.comparison.candidate.affected_sequences
                if not any(low <= sequence <= high for sequence in scope):
                    continue
            selected.append(entry)
        return tuple(selected)

    def compare_interpretation_set(
        self,
        names,
        *,
        conflicts_only: bool = False,
    ) -> InterpretationSetComparison:
        """Align two or more candidates over their overlapping observations.

        The comparison is a pure view: it never adopts a candidate, never
        modifies raw observations and never rewrites published outputs.
        """
        names = tuple(str(name) for name in names)
        if len(names) < 2:
            raise ValueError("select at least two interpretations to compare")
        interps = [self._require_interpretation(name) for name in names]
        sensors = {interp.sensor for interp in interps}
        if len(sensors) != 1:
            raise ValueError(
                "interpretation set must explain the same sensor: "
                + ", ".join(sorted(sensors))
            )
        sensor = interps[0].sensor
        visible = [
            item for item in self._visible(self.arrival_horizon) if item.sensor == sensor
        ]
        scopes: list[set[int]] = []
        for interp in interps:
            if interp.adopted_sequences is not None:
                scopes.append(set(interp.adopted_sequences))
            else:
                scopes.append({item.sequence for item in visible})
        overlap = set.intersection(*scopes) if scopes else set()
        calibrations = [
            self.calibration.with_reboot_clock(interp.clock) for interp in interps
        ]
        time_differs: list[ObservationAlignment] = []
        same_time_different_basis: list[ObservationAlignment] = []
        identical: list[ObservationAlignment] = []
        for observation in sorted(visible, key=lambda item: item.sequence):
            if observation.sequence not in overlap:
                continue
            times = tuple(
                calibration.corrected_time(observation)
                for calibration in calibrations
            )
            basis = tuple(
                self._basis_key(interp.clock, observation) for interp in interps
            )
            spread = max(times) - min(times)
            if spread > TIME_TOLERANCE:
                category = "time_differs"
            elif len(set(basis)) > 1:
                category = "same_time_different_basis"
            else:
                category = "identical"
            alignment = ObservationAlignment(
                observation.sensor,
                observation.sequence,
                observation.time,
                times,
                spread,
                category,
                basis,
            )
            if category == "time_differs":
                time_differs.append(alignment)
            elif category == "same_time_different_basis":
                same_time_different_basis.append(alignment)
            else:
                identical.append(alignment)
        identical_count = len(identical)
        return InterpretationSetComparison(
            names,
            sensor,
            tuple(self._candidate_summary(interp) for interp in interps),
            tuple(time_differs),
            tuple(same_time_different_basis),
            () if conflicts_only else tuple(identical),
            identical_count,
            conflicts_only,
        )

    def export_interpretation_set(
        self,
        names,
        path,
        *,
        conflicts_only: bool = False,
    ) -> InterpretationSetComparison:
        """Write the current set comparison to the given path as JSON."""
        comparison = self.compare_interpretation_set(
            names, conflicts_only=conflicts_only
        )
        Path(path).write_text(
            json.dumps(comparison.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return comparison

    def save_comparison_view(
        self,
        *,
        filters: dict | None = None,
        selection=None,
        conflicts_only: bool = False,
    ) -> dict:
        """Persist the current filter/selection so a reopen resumes the view."""
        self.comparison_view = {
            "filters": dict(filters or {}),
            "selection": [str(name) for name in (selection or [])],
            "conflicts_only": bool(conflicts_only),
        }
        return self.comparison_view

    def resume_comparison_view(self) -> dict:
        """Recompute the saved filter matches and selection comparison."""
        view = self.comparison_view or {}
        filters = dict(view.get("filters") or {})
        selection = [str(name) for name in (view.get("selection") or [])]
        conflicts_only = bool(view.get("conflicts_only", False))
        matches = [entry.name for entry in self.filter_interpretations(**filters)]
        comparison = None
        if len(selection) >= 2 and all(
            name in self.interpretations for name in selection
        ):
            comparison = self.compare_interpretation_set(
                selection, conflicts_only=conflicts_only
            )
        return {
            "filters": filters,
            "selection": selection,
            "conflicts_only": conflicts_only,
            "matches": matches,
            "comparison": comparison,
        }

    def _candidate_summary(self, interp: Interpretation) -> CandidateSummary:
        clock = interp.clock
        inner = (
            clock.candidate if isinstance(clock, PartialAdoptionClock) else clock
        )
        drift_assumptions: list[dict[str, object]] = []
        if isinstance(inner, SegmentedSensorClock):
            for segment in inner.segments:
                offsets = [anchor.offset for anchor in segment.anchors]
                if len(segment.anchors) >= 2:
                    first, last = segment.anchors[0], segment.anchors[-1]
                    span = last.stamp - first.stamp
                    implied_drift = (
                        0.0
                        if abs(span) <= 1e-12
                        else (last.offset - first.offset) / span
                    )
                else:
                    implied_drift = 0.0
                drift_assumptions.append(
                    {
                        "segment_index": segment.index,
                        "calibrated": segment.calibrated,
                        "single_anchor": segment.single_anchor,
                        "anchor_count": len(segment.anchors),
                        "offsets": offsets,
                        "implied_drift": implied_drift,
                    }
                )
        else:
            model = inner
            drift_assumptions.append(
                {
                    "segment_index": None,
                    "calibrated": True,
                    "single_anchor": False,
                    "anchor_count": len(getattr(model, "anchors", ()) or ()),
                    "offsets": [],
                    "implied_drift": float(getattr(model, "drift", 0.0) or 0.0),
                }
            )
        anchor_basis = [
            self.anchor_records[anchor_id].as_dict()
            for anchor_id in interp.anchor_ids
            if anchor_id in self.anchor_records
        ]
        return CandidateSummary(
            interp.name,
            interp.sensor,
            interp.status,
            interp.note,
            interp.adopted_sequences,
            tuple(drift_assumptions),
            tuple(anchor_basis),
        )

    @staticmethod
    def _basis_key(clock, observation: Observation) -> str:
        """Which segment/anchors back this observation's corrected time."""
        sequence = observation.sequence
        if isinstance(clock, PartialAdoptionClock):
            adopted = clock.adopts(sequence)
            inner = clock.candidate if adopted else clock.base
            prefix = "adopted" if adopted else "base"
        else:
            inner = clock
            prefix = "full"
        if isinstance(inner, SegmentedSensorClock):
            segment = inner.segment_for(sequence)
            anchors = ",".join(
                f"{anchor.stamp}:{anchor.corrected_time}:{anchor.uncertainty}"
                for anchor in segment.anchors
            )
            return f"{prefix}|segment:{segment.index}|{anchors}"
        return f"{prefix}|model|{inner.as_dict()!r}"

    def adopt_interpretation(self, name: str) -> RecomputeReport:
        """Explicitly publish one pending candidate; nothing adopts itself."""
        interp = self._require_interpretation(name)
        if interp.status != "pending":
            raise RuntimeError(
                f"interpretation {name!r} is {interp.status}; "
                "only pending candidates can be adopted"
            )
        self.staged_calibration = self.calibration.with_reboot_clock(interp.clock)
        self.staged_sensor = interp.sensor
        self.staged_comparison = self.compare_calibrations(
            interp.sensor, self.calibration, self.staged_calibration
        )
        report = self.publish_staged_delay()
        self.interpretations[name] = replace(interp, status="adopted")
        return report

    def discard_interpretation(self, name: str) -> None:
        """Withdraw a pending candidate explicitly; others are unaffected."""
        interp = self._require_interpretation(name)
        if interp.status != "pending":
            raise RuntimeError(
                f"interpretation {name!r} is {interp.status}; cannot discard"
            )
        del self.interpretations[name]
        self.interpretation_comparisons.pop(name, None)

    def retract_anchor(self, anchor_id: str) -> AnchorRetraction:
        """Withdraw an anchor and list every interpretation depending on it.

        Dependent interpretations are kept with status
        ``anchor_retracted`` for traceability instead of being silently
        rewritten or deleted; unrelated candidates stay pending.
        """
        if anchor_id not in self.anchor_records:
            raise KeyError(f"unknown anchor {anchor_id!r}")
        self.anchor_records.pop(anchor_id)
        dependents = tuple(
            name
            for name, interp in self.interpretations.items()
            if interp.status == "pending" and anchor_id in interp.anchor_ids
        )
        for name in dependents:
            self.interpretations[name] = replace(
                self.interpretations[name], status="anchor_retracted"
            )
        return AnchorRetraction(anchor_id, dependents)

    def _register_interpretation(
        self,
        name: str,
        sensor: str,
        model: SegmentedSensorClock,
        *,
        anchor_ids: tuple[str, ...],
        applies_to: tuple[int, int] | None,
        note: str,
    ) -> DelayComparison:
        if name in self.interpretations:
            raise ValueError(
                f"interpretation {name!r} already registered; "
                "choose a new name instead of overriding it"
            )
        clock: object = model
        adopted: tuple[int, ...] | None = None
        if applies_to is not None:
            low, high = applies_to
            adopted = tuple(
                item.sequence
                for item in self.received
                if item.sensor == sensor and low <= item.sequence <= high
            )
            adopted = tuple(sorted(adopted))
            base = self.calibration.reboot_clock_for(sensor)
            if base is None:
                base = self.calibration.clock_for(sensor)
            clock = PartialAdoptionClock(sensor, base, model, frozenset(adopted))
        interpretation = Interpretation(
            name, sensor, clock, tuple(anchor_ids), adopted, note
        )
        self.interpretations[name] = interpretation
        comparison = self.compare_calibrations(
            sensor, self.calibration, self.calibration.with_reboot_clock(clock)
        )
        self.interpretation_comparisons[name] = comparison
        return comparison

    def _require_interpretation(self, name: str) -> Interpretation:
        if name not in self.interpretations:
            raise KeyError(f"unknown interpretation {name!r}")
        return self.interpretations[name]

    # ------------------------------------------------- composite schemes

    def compose_scheme(
        self,
        name: str,
        sensor: str,
        pieces,
        *,
        note: str = "",
    ) -> CompositeScheme:
        """Compose registered candidates into one piecewise scheme by range.

        Each piece is `(interpretation_name, (low_sequence, high_sequence))`
        (or a dict with `source`/`sequence_range`) and keeps the
        provenance of the candidate it came from.  Boundaries between pieces
        record both neighbouring sources; observations are never merged just
        because their currently displayed corrected times look close.  The
        scheme is only registered for comparison; nothing is published.
        """
        if name in self.compositions:
            raise ValueError(
                f"composition {name!r} already registered; "
                "choose a new name instead of overriding it"
            )
        source_clocks: dict[str, SegmentedSensorClock] = {}
        parsed: list[CompositionPiece] = []
        for item in pieces:
            if isinstance(item, dict):
                source_name = str(item["source"])
                seq_range = item["sequence_range"]
            else:
                source_name, seq_range = item[0], item[1]
            source_name = str(source_name)
            interp = self._require_interpretation(source_name)
            if interp.sensor != sensor:
                raise ValueError(
                    f"interpretation {source_name!r} explains {interp.sensor!r}, "
                    f"not {sensor!r}"
                )
            clock = interp.clock
            if isinstance(clock, PartialAdoptionClock):
                clock = clock.candidate
            if not isinstance(clock, SegmentedSensorClock):
                raise ValueError(
                    "composition pieces require reboot-segment interpretations"
                )
            source_clocks[source_name] = clock
            low, high = int(seq_range[0]), int(seq_range[1])
            if high < low:
                raise ValueError("piece sequence range must be ordered")
            first_segment = clock.segment_index_for(low)
            last_segment = clock.segment_index_for(high)
            anchors: list[SegmentAnchor] = []
            seen_stamps: set[float] = set()
            for segment in clock.segments[first_segment : last_segment + 1]:
                for anchor in segment.anchors:
                    if anchor.stamp in seen_stamps:
                        continue
                    seen_stamps.add(anchor.stamp)
                    anchors.append(anchor)
            anchors.sort(key=lambda anchor: anchor.stamp)
            anchor_ids: list[str] = []
            for anchor_id in interp.anchor_ids:
                record = self.anchor_records.get(anchor_id)
                if record is None:
                    continue
                scope = record.applies_to
                if scope is None or not (scope[1] < low or scope[0] > high):
                    anchor_ids.append(anchor_id)
            parsed.append(
                CompositionPiece(
                    len(parsed),
                    source_name,
                    (low, high),
                    tuple(anchors),
                    tuple(anchor_ids),
                )
            )
        parsed.sort(key=lambda piece: piece.sequence_range)
        parsed = [
            replace(piece, index=index) for index, piece in enumerate(parsed)
        ]
        boundaries: list[CompositionBoundary] = []
        for left, right in zip(parsed, parsed[1:]):
            cut = left.sequence_range[1]
            confirmed = False
            window: tuple[float, float] | None = None
            for piece in (left, right):
                for boundary in source_clocks[piece.source].boundaries:
                    if boundary.after_sequence != cut:
                        continue
                    confirmed = confirmed or boundary.confirmed
                    if window is None:
                        window = boundary.reset_window
            boundaries.append(
                CompositionBoundary(
                    cut, left.source, right.source, confirmed, window
                )
            )
        scheme = CompositeScheme(
            name, sensor, tuple(parsed), tuple(boundaries), (), "pending", note
        )
        scheme = replace(
            scheme, contradictions=self._composition_contradictions(scheme)
        )
        self.compositions[name] = scheme
        return scheme

    def _composition_contradictions(
        self,
        scheme: CompositeScheme,
        piece_indices: set[int] | None = None,
    ) -> tuple[CompositionContradiction, ...]:
        """Check monotonicity, reboot breaks and shared anchors; never smooth."""
        def wanted(index: int) -> bool:
            return piece_indices is None or index in piece_indices

        problems: list[CompositionContradiction] = []
        pieces = scheme.pieces
        for position, left in enumerate(pieces):
            for right in pieces[position + 1 :]:
                if not (wanted(left.index) or wanted(right.index)):
                    continue
                low = max(left.sequence_range[0], right.sequence_range[0])
                high = min(left.sequence_range[1], right.sequence_range[1])
                if low <= high:
                    problems.append(
                        CompositionContradiction(
                            "range_overlap",
                            (low, high),
                            "两段组合范围在同一原始观测上重叠，同一观测不能同时属于两套解释",
                            (
                                {
                                    "source": left.source,
                                    "sequence_range": list(left.sequence_range),
                                },
                                {
                                    "source": right.source,
                                    "sequence_range": list(right.sequence_range),
                                },
                            ),
                            (left.index, right.index),
                        )
                    )
        visible = [
            item
            for item in self._visible(self.arrival_horizon)
            if item.sensor == scheme.sensor
        ]
        for piece in pieces:
            if not wanted(piece.index):
                continue
            low, high = piece.sequence_range
            own = sorted(
                (item for item in visible if low <= item.sequence <= high),
                key=lambda item: item.sequence,
            )
            previous: Observation | None = None
            for observation in own:
                if previous is not None and observation.time < previous.time - 1e-9:
                    problems.append(
                        CompositionContradiction(
                            "non_monotonic_stamp",
                            (previous.sequence, observation.sequence),
                            "同一时钟段内原始戳记随序号倒退，段内可能存在未标出的重启断点",
                            (
                                {
                                    "sequence": previous.sequence,
                                    "stamp": previous.time,
                                },
                                {
                                    "sequence": observation.sequence,
                                    "stamp": observation.time,
                                },
                            ),
                            (piece.index,),
                        )
                    )
                previous = observation
            if piece.anchors:
                clock = SegmentClock(piece.index, piece.anchors)
                previous_entry: tuple[Observation, float] | None = None
                for observation in own:
                    corrected = clock.corrected_time_at(observation.time)
                    if (
                        previous_entry is not None
                        and corrected < previous_entry[1] - 1e-9
                    ):
                        problems.append(
                            CompositionContradiction(
                                "non_monotonic_corrected",
                                (previous_entry[0].sequence, observation.sequence),
                                "按该段锚点校正后的时间随序号倒退，锚点与观测顺序矛盾",
                                (
                                    {
                                        "sequence": previous_entry[0].sequence,
                                        "stamp": previous_entry[0].time,
                                        "corrected_time": previous_entry[1],
                                    },
                                    {
                                        "sequence": observation.sequence,
                                        "stamp": observation.time,
                                        "corrected_time": corrected,
                                    },
                                ),
                                (piece.index,),
                            )
                        )
                    previous_entry = (observation, corrected)
            source_interp = self.interpretations.get(piece.source)
            source_clock = (
                None if source_interp is None else source_interp.clock
            )
            if isinstance(source_clock, PartialAdoptionClock):
                source_clock = source_clock.candidate
            if isinstance(source_clock, SegmentedSensorClock):
                for boundary in source_clock.boundaries:
                    if low <= boundary.after_sequence < high:
                        problems.append(
                            CompositionContradiction(
                                "reboot_inside_piece",
                                (boundary.after_sequence, boundary.after_sequence + 1),
                                "来源候选在该范围内部记录了重启断点，组合段必须沿断点拆分而不能整段沿用",
                                (
                                    {
                                        "source": piece.source,
                                        "after_sequence": boundary.after_sequence,
                                        "confirmed": boundary.confirmed,
                                    },
                                ),
                                (piece.index,),
                            )
                        )
        for boundary in scheme.boundaries:
            left = scheme.piece_for_sequence(boundary.after_sequence)
            right = scheme.piece_for_sequence(boundary.after_sequence + 1)
            if left is None or right is None:
                continue
            if not (wanted(left.index) or wanted(right.index)):
                continue
            if boundary.confirmed:
                continue
            for left_anchor in left.anchors:
                for right_anchor in right.anchors:
                    if (
                        abs(left_anchor.stamp - right_anchor.stamp) <= 1e-9
                        and abs(left_anchor.corrected_time - right_anchor.corrected_time)
                        > 1e-9
                    ):
                        problems.append(
                            CompositionContradiction(
                                "shared_anchor_conflict",
                                (boundary.after_sequence, boundary.after_sequence + 1),
                                "边界两侧来源对同一原始戳记给出不同校正时间，且该边界没有确认的重启记录",
                                (
                                    {
                                        "source": left.source,
                                        "stamp": left_anchor.stamp,
                                        "corrected_time": left_anchor.corrected_time,
                                    },
                                    {
                                        "source": right.source,
                                        "stamp": right_anchor.stamp,
                                        "corrected_time": right_anchor.corrected_time,
                                    },
                                ),
                                (left.index, right.index),
                            )
                        )
        return tuple(problems)

    def compare_composition_ordering(
        self, names
    ) -> CompositionOrderingComparison:
        """Compare how pending schemes reorder cross-sensor event pairs.

        Read-only: every scheme stays unconfirmed, raw observations and
        published outputs are untouched.
        """
        names = tuple(str(name) for name in names)
        if len(names) < 2:
            raise ValueError("select at least two compositions to compare")
        schemes = [self._require_composition(name) for name in names]
        sensors = {scheme.sensor for scheme in schemes}
        if len(sensors) != 1:
            raise ValueError(
                "compositions must explain the same sensor: "
                + ", ".join(sorted(sensors))
            )
        sensor = schemes[0].sensor
        visible = self._visible(self.arrival_horizon)
        positions: dict[str, dict[tuple[str, int], int]] = {}
        counts: dict[str, int] = {}
        for scheme in schemes:
            calibration = self.calibration.with_reboot_clock(scheme.as_clock())
            ordered = sorted(
                visible,
                key=lambda item: (
                    calibration.corrected_time(item),
                    item.sensor,
                    item.sequence,
                ),
            )
            positions[scheme.name] = {
                (item.sensor, item.sequence): index
                for index, item in enumerate(ordered)
            }
            counts[scheme.name] = len(ordered)
        own = [item for item in visible if item.sensor == sensor]
        others = [item for item in visible if item.sensor != sensor]
        changes: list[OrderingChange] = []
        for first in own:
            first_key = (first.sensor, first.sequence)
            for second in others:
                second_key = (second.sensor, second.sequence)
                orders: list[tuple[str, str]] = []
                distinct: set[str] = set()
                for scheme in schemes:
                    placement = positions[scheme.name]
                    left = placement.get(first_key)
                    right = placement.get(second_key)
                    if left is None or right is None or left == right:
                        sign = "same"
                    elif left < right:
                        sign = "before"
                    else:
                        sign = "after"
                    orders.append((scheme.name, sign))
                    distinct.add(sign)
                if len(distinct) > 1:
                    changes.append(OrderingChange(first_key, second_key, tuple(orders)))
        return CompositionOrderingComparison(
            names,
            sensor,
            tuple(changes),
            tuple((name, counts[name]) for name in names),
        )

    def adopt_composition(self, name: str) -> RecomputeReport:
        """Explicitly publish one pending scheme as a new version.

        Adoption never rewrites the original receive order or any published
        (sealed) result; other pending schemes stay available.  A scheme
        with unresolved contradictions cannot be adopted: the conflicting
        intervals and their evidence are reported instead of being smoothed.
        """
        scheme = self._require_composition(name)
        if scheme.status != "pending":
            raise RuntimeError(
                f"composition {name!r} is {scheme.status}; "
                "only pending schemes can be adopted"
            )
        if scheme.contradictions:
            details = "; ".join(
                f"[{item.sequence_interval[0]}, {item.sequence_interval[1]}] "
                f"{item.kind}: {item.reason}"
                for item in scheme.contradictions
            )
            raise RuntimeError(
                f"composition {name!r} has unresolved contradictions: {details}"
            )
        self.staged_calibration = self.calibration.with_reboot_clock(
            scheme.as_clock()
        )
        self.staged_sensor = scheme.sensor
        self.staged_comparison = self.compare_calibrations(
            scheme.sensor, self.calibration, self.staged_calibration
        )
        report = self.publish_staged_delay()
        self.compositions[name] = replace(scheme, status="adopted")
        return report

    def discard_composition(self, name: str) -> None:
        """Withdraw a pending scheme explicitly; others are unaffected."""
        scheme = self._require_composition(name)
        if scheme.status != "pending":
            raise RuntimeError(
                f"composition {name!r} is {scheme.status}; cannot discard"
            )
        del self.compositions[name]

    def register_composition_anchor(
        self,
        name: str,
        anchor_id: str,
        piece_index: int,
        anchor,
        *,
        source: str = "",
        arrival_time: float | None = None,
    ) -> CompositionRecheck:
        """Add a local anchor to one piece and re-check only that range.

        Pieces outside the affected range keep their previously confirmed
        basis untouched; ranges still without anchors remain unknown.
        """
        scheme = self._require_composition(name)
        if not 0 <= piece_index < len(scheme.pieces):
            raise ValueError(f"unknown composition piece {piece_index}")
        piece = scheme.pieces[piece_index]
        parsed = (
            anchor
            if isinstance(anchor, SegmentAnchor)
            else SegmentAnchor(*(float(value) for value in anchor))
        )
        self.register_anchor(
            anchor_id,
            scheme.sensor,
            piece_index,
            parsed,
            arrival_time=arrival_time,
            applies_to=piece.sequence_range,
            source=source,
        )
        merged = tuple(sorted([*piece.anchors, parsed], key=lambda item: item.stamp))
        anchor_ids = tuple(sorted({*piece.anchor_ids, anchor_id}))
        pieces = list(scheme.pieces)
        pieces[piece_index] = replace(piece, anchors=merged, anchor_ids=anchor_ids)
        updated = replace(scheme, pieces=tuple(pieces))
        affected = {piece_index}
        fresh = self._composition_contradictions(updated, affected)
        kept = tuple(
            item
            for item in scheme.contradictions
            if not (set(item.piece_indices) & affected)
        )
        updated = replace(updated, contradictions=kept + fresh)
        self.compositions[name] = updated
        return CompositionRecheck(
            name,
            (piece.sequence_range,),
            tuple(
                item.sequence_range
                for item in updated.pieces
                if item.index != piece_index
            ),
            updated.contradictions,
            self._composition_unknown_ranges(updated),
        )

    def _composition_unknown_ranges(
        self, scheme: CompositeScheme
    ) -> tuple[tuple[int, int], ...]:
        """Uncovered original ranges, including beyond the outermost pieces."""
        gaps = set(scheme.unknown_ranges())
        sequences = [
            item.sequence
            for item in self.received
            if item.sensor == scheme.sensor
        ]
        if sequences:
            ordered = sorted(scheme.pieces, key=lambda item: item.sequence_range)
            low, high = min(sequences), max(sequences)
            if not ordered:
                gaps.add((low, high))
            else:
                if low < ordered[0].sequence_range[0]:
                    gaps.add((low, ordered[0].sequence_range[0] - 1))
                if high > ordered[-1].sequence_range[1]:
                    gaps.add((ordered[-1].sequence_range[1] + 1, high))
        return tuple(sorted(gaps))

    def _require_composition(self, name: str) -> CompositeScheme:
        if name not in self.compositions:
            raise KeyError(f"unknown composition {name!r}")
        return self.compositions[name]

    # ------------------------------------------------------------- inspection

    def state(self) -> StateEstimate:
        return self.estimator.snapshot()

    def all_outputs(self) -> list[StateEstimate]:
        return [*self.sealed_outputs, *self.outputs]

    def visible_data_boundary(self) -> dict[str, float | int | None]:
        visible = self._visible(self.arrival_horizon)
        stamps = [item.time for item in visible]
        arrivals = [item.observed_at() for item in visible]
        return {
            "arrival_horizon": self.arrival_horizon,
            "visible_count": len(visible),
            "min_stamp": min(stamps) if stamps else None,
            "max_stamp": max(stamps) if stamps else None,
            "max_arrival": max(arrivals) if arrivals else None,
            "seal_time": self.seal_time,
        }

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self) -> Checkpoint:
        checkpoint = Checkpoint(
            self.estimator.snapshot(),
            len(self._eligible(self.arrival_horizon)),
            self.arrival_horizon,
            self.seal_time,
            self.calibration,
            self._visible(self.arrival_horizon),
            [self._clone_state(item) for item in self.sealed_outputs],
            dict(self.rejected_observations),
            self.staged_calibration,
            self.staged_sensor,
            list(self.anchor_records.values()),
            list(self.interpretations.values()),
            self.comparison_view,
            list(self.compositions.values()),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def checkpoint_observations(self) -> list[Observation]:
        if self.arrival_horizon is None:
            return []
        return self._visible(self.arrival_horizon)

    def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        prior_observations: list[Observation] | None = None,
    ) -> None:
        self.calibration = checkpoint.calibration or DelayCalibration()
        self.arrival_horizon = checkpoint.arrival_horizon
        self.seal_time = checkpoint.seal_time
        visible = (
            prior_observations
            if prior_observations is not None
            else list(checkpoint.visible_observations)
        )
        self.received = list(visible)
        self.rejected_observations = dict(checkpoint.rejected_observations)
        events = self._eligible(self.arrival_horizon) if self.arrival_horizon is not None else []
        if self.seal_time is not None:
            self.sealed_outputs = [
                self._clone_state(item) for item in checkpoint.sealed_outputs
            ]
            self._restore_anchor_and_rebuild(events)
        else:
            self.sealed_outputs = []
            self.estimator = self._estimator_restored(checkpoint.state)
            self.outputs = [self._clone_state(item) for item in []]
            self._rebuild_tail(events)
            if self.arrival_horizon is not None:
                self.estimator.predict(self.arrival_horizon)
        self.staged_calibration = checkpoint.staged_calibration
        self.staged_sensor = checkpoint.staged_sensor
        self.comparison_view = (
            None
            if checkpoint.comparison_view is None
            else dict(checkpoint.comparison_view)
        )
        if self.staged_calibration is not None and self.staged_sensor:
            self.staged_comparison = self.compare_calibrations(
                self.staged_sensor, self.calibration, self.staged_calibration
            )
        else:
            self.staged_comparison = None
        self.checkpoints = [checkpoint]
        self.anchor_records = {}
        for item in checkpoint.anchor_records:
            record = (
                item
                if isinstance(item, AnchorRecord)
                else AnchorRecord.from_dict(item)
            )
            self.anchor_records[record.anchor_id] = record
        self.interpretations = {}
        self.interpretation_comparisons = {}
        for item in checkpoint.pending_interpretations:
            interp = (
                item
                if isinstance(item, Interpretation)
                else Interpretation.from_dict(item)
            )
            self.interpretations[interp.name] = interp
            self.interpretation_comparisons[interp.name] = self.compare_calibrations(
                interp.sensor,
                self.calibration,
                self.calibration.with_reboot_clock(interp.clock),
            )
        self.compositions = {}
        for item in checkpoint.compositions:
            scheme = (
                item
                if isinstance(item, CompositeScheme)
                else CompositeScheme.from_dict(item)
            )
            self.compositions[scheme.name] = scheme

    def replay(self) -> list[StateEstimate]:
        if self.arrival_horizon is None:
            events = sorted(
                self.received,
                key=lambda item: self._sort_key(item),
            )
        else:
            events = self._eligible(self.arrival_horizon)
        return self._replay_outputs(events)

    # ------------------------------------------------------------- internals

    def _retime(self, observation: Observation) -> Observation:
        return self._retime_with(observation, self.calibration)

    def _retime_with(
        self, observation: Observation, calibration: DelayCalibration
    ) -> Observation:
        corrected = calibration.corrected_time(observation)
        if abs(corrected - observation.time) <= 1e-12:
            return observation
        return type(observation)(
            time=corrected,
            sensor=observation.sensor,
            sequence=observation.sequence,
            arrival_time=observation.arrival_time,
            **{
                key: value
                for key, value in ((f.name, getattr(observation, f.name)) for f in fields(observation))
                if key not in {"time", "sensor", "sequence", "arrival_time"}
            },
        )

    @staticmethod
    def _estimator_restored(state: StateEstimate) -> LocalizationEstimator:
        estimator = LocalizationEstimator()
        estimator.restore(state)
        return estimator

    def _sort_key(self, observation: Observation, calibration: DelayCalibration | None = None):
        calibration = calibration or self.calibration
        return (
            calibration.corrected_time(observation),
            observation.sensor,
            observation.sequence,
        )

    def _visible(self, arrival_horizon: float | None) -> list[Observation]:
        if arrival_horizon is None:
            return []
        return [
            item
            for item in self.received
            if item.observed_at() <= arrival_horizon + 1e-9
        ]

    def _eligible(self, arrival_horizon: float | None) -> list[Observation]:
        visible = self._visible(arrival_horizon)
        if self.seal_time is not None:
            visible = [
                item
                for item in visible
                if self.calibration.corrected_time(item) > self.seal_time + 1e-9
            ]
        return sorted(visible, key=lambda item: self._sort_key(item))

    def _eligible_with(
        self,
        arrival_horizon: float | None,
        calibration: DelayCalibration,
    ) -> list[Observation]:
        visible = self._visible(arrival_horizon)
        if self.seal_time is not None:
            visible = [
                item
                for item in visible
                if calibration.corrected_time(item) > self.seal_time + 1e-9
            ]
        return sorted(visible, key=lambda item: self._sort_key(item, calibration))

    def _tail_events(
        self,
        events: list[Observation] | None = None,
        calibration: DelayCalibration | None = None,
    ) -> list[Observation]:
        calibration = calibration or self.calibration
        events = events if events is not None else self._eligible(self.arrival_horizon)
        if self.seal_time is None:
            return events
        return [
            item
            for item in events
            if calibration.corrected_time(item) > self.seal_time + 1e-9
        ]

    def _boundary_state(
        self,
        time: float,
        estimator: LocalizationEstimator,
    ) -> StateEstimate:
        for observation in self._eligible(self.arrival_horizon):
            corrected_time = self.calibration.corrected_time(observation)
            if corrected_time <= estimator.time + 1e-9:
                continue
            if corrected_time <= time + 1e-9:
                estimator.step(self._retime(observation))
        estimator.predict(time)
        return estimator.snapshot()

    @staticmethod
    def _clone_state(state: StateEstimate) -> StateEstimate:
        return StateEstimate(
            time=state.time,
            x=state.x,
            y=state.y,
            heading=state.heading,
            linear_velocity=state.linear_velocity,
            yaw_rate=state.yaw_rate,
            forward_acceleration=state.forward_acceleration,
            covariance=state.covariance.copy(),
            last_wheel_time=state.last_wheel_time,
            last_imu_time=state.last_imu_time,
            degraded_reasons=set(state.degraded_reasons),
        )

    def _observation_decision(
        self,
        observation: Observation,
        baseline: DelayCalibration,
        candidate: DelayCalibration,
    ) -> ObservationDecision:
        old_time = baseline.corrected_time(observation)
        new_time = candidate.corrected_time(observation)
        boundary = self.seal_time
        if boundary is None:
            classification = "open_recomputed"
            reason = "没有封存边界，观测可按候选延迟参与全开放区间重算"
        elif new_time > boundary + 1e-9 and old_time > boundary + 1e-9:
            classification = "open_recomputed"
            reason = "旧、新校正时间均晚于封存边界，仅影响开放区间"
        elif new_time <= boundary + 1e-9 and old_time <= boundary + 1e-9:
            classification = "sealed_unchanged"
            reason = "旧、新校正时间均不晚于封存边界，历史保持原值"
        elif new_time <= boundary + 1e-9 and old_time > boundary + 1e-9:
            classification = "crossing_rejected"
            reason = "候选延迟会把开放观测移入封存区间，不能回写边界前状态"
        else:
            classification = "open_adopted"
            reason = "候选延迟把观测移到封存边界之后，从封存边界状态开始参与开放重算"
        time_shift = old_time - new_time
        segment_index = candidate.segment_index_for(observation)
        baseline_range = baseline.corrected_range(observation)
        candidate_range = candidate.corrected_range(observation)
        return ObservationDecision(
            observation.sensor,
            observation.sequence,
            observation.time,
            observation.arrival_time,
            old_time,
            new_time,
            time_shift,
            classification,
            reason,
            segment_index,
            baseline_range,
            candidate_range,
        )

    def _setting_view(
        self,
        sensor: str,
        affected: list[Observation],
        view_calibration: DelayCalibration,
        baseline: DelayCalibration | None = None,
    ) -> DelaySettingView:
        baseline = baseline or view_calibration
        delay = view_calibration.delay_for(sensor)
        times = [view_calibration.corrected_time(item) for item in affected]
        open_sequences = []
        sealed_sequences = []
        rejected_sequences = []
        for item in affected:
            decision = self._observation_decision(item, baseline, view_calibration)
            corrected = view_calibration.corrected_time(item)
            if self.seal_time is not None and corrected <= self.seal_time + 1e-9:
                sealed_sequences.append(item.sequence)
            if decision.classification == "crossing_rejected":
                rejected_sequences.append(item.sequence)
            if corrected > (self.seal_time if self.seal_time is not None else -1e300) + 1e-9:
                open_sequences.append(item.sequence)
        return DelaySettingView(
            delay,
            (min(times), max(times)) if times else None,
            tuple(item.sequence for item in affected),
            tuple(open_sequences),
            tuple(sealed_sequences),
            tuple(rejected_sequences),
            view_calibration.clock_for(sensor),
        )

    def _compare_timeline_positions(
        self,
        visible: list[Observation],
        affected: list[Observation],
        baseline: DelayCalibration,
        candidate: DelayCalibration,
        decisions: tuple[ObservationDecision, ...],
    ) -> tuple[tuple[TimelinePosition, ...], tuple[TimelinePositionChange, ...]]:
        affected_keys = {(item.sensor, item.sequence) for item in affected}
        decision_by_key = {(item.sensor, item.sequence): item for item in decisions}
        baseline_events = self._eligible_with(self.arrival_horizon, baseline)
        candidate_events = self._eligible_with(self.arrival_horizon, candidate)
        baseline_states = self._simulate_event_states(baseline_events, baseline)
        candidate_states = self._simulate_event_states(candidate_events, candidate)
        baseline_order = {
            (item.sensor, item.sequence): index
            for index, item in enumerate(baseline_events)
        }
        candidate_order = {
            (item.sensor, item.sequence): index
            for index, item in enumerate(candidate_events)
        }
        timeline: list[TimelinePosition] = []
        position_changes: list[TimelinePositionChange] = []
        for observation in sorted(visible, key=lambda item: (item.sensor, item.sequence)):
            key = (observation.sensor, observation.sequence)
            old_time = baseline.corrected_time(observation)
            new_time = candidate.corrected_time(observation)
            decision = decision_by_key.get(key)
            if decision is not None:
                classification = decision.classification
                reason = decision.reason
            elif key in self.rejected_observations:
                classification = "previously_rejected"
                reason = self.rejected_observations[key]
            else:
                classification = "identity_unchanged"
                reason = "来源与序号不变，仅在新的全局顺序中定位"
            old_order = baseline_order.get(key)
            new_order = candidate_order.get(key)
            timeline.append(
                TimelinePosition(
                    key,
                    observation.sensor,
                    observation.sequence,
                    observation.time,
                    observation.arrival_time,
                    old_time,
                    new_time,
                    old_time - new_time,
                    -1 if old_order is None else old_order,
                    -1 if new_order is None else new_order,
                    0 if old_order is None or new_order is None else new_order - old_order,
                    classification,
                    reason,
                    candidate.segment_index_for(observation),
                )
            )
            if key in affected_keys:
                old_state = baseline_states.get(key)
                new_state = candidate_states.get(key)
                position_changes.append(
                    TimelinePositionChange(
                        key,
                        old_time,
                        new_time,
                        None if old_state is None else old_state.x,
                        None if old_state is None else old_state.y,
                        None if new_state is None else new_state.x,
                        None if new_state is None else new_state.y,
                    )
                )
        return tuple(timeline), tuple(position_changes)

    def _simulate_event_states(
        self, events: list[Observation], calibration: DelayCalibration
    ) -> dict[tuple[str, int], StateEstimate]:
        estimator = LocalizationEstimator()
        if self.sealed_outputs:
            estimator.restore(self.sealed_outputs[-1])
        states: dict[tuple[str, int], StateEstimate] = {}
        for observation in events:
            estimator.step(self._retime_with(observation, calibration))
            states[(observation.sensor, observation.sequence)] = estimator.snapshot()
        return states

    def _open_gaps(self, calibration: DelayCalibration) -> tuple[UncertainInterval, ...]:
        events = self._eligible_with(self.arrival_horizon, calibration)
        estimator = LocalizationEstimator()
        thresholds = {
            sensor: estimator.wheel_timeout if "wheel" in sensor else estimator.imu_timeout
            for sensor in {item.sensor for item in events}
        }
        gaps: list[UncertainInterval] = []
        for sensor, timeout in thresholds.items():
            times = [
                calibration.corrected_time(item)
                for item in events
                if item.sensor == sensor
            ]
            if not times:
                continue
            start_boundary = self.seal_time if self.seal_time is not None else times[0]
            end_boundary = self.arrival_horizon if self.arrival_horizon is not None else times[-1]
            boundaries = [start_boundary, *times, end_boundary]
            for start, end in zip(boundaries, boundaries[1:]):
                if end - start > timeout + 1e-9:
                    gaps.append(
                        UncertainInterval(
                            start,
                            end,
                            f"{sensor} 在开放区间缺少 {end - start:.3f}s 观测，位置仍需预测",
                        )
                    )
        return tuple(gaps)

    def _segment_views(
        self,
        sensor: str,
        affected: list[Observation],
        calibration: DelayCalibration,
    ) -> tuple[SegmentView, ...]:
        model = calibration.reboot_clock_for(sensor)
        if model is None:
            return ()
        contradiction_pairs = {
            tuple(item["between_segments"]) for item in model.contradictions()
        }
        views: list[SegmentView] = []
        for segment in model.segments:
            items = [
                item
                for item in affected
                if model.segment_index_for(item.sequence) == segment.index
            ]
            sequences = tuple(item.sequence for item in items)
            stamp_span = (
                (min(item.time for item in items), max(item.time for item in items))
                if items
                else None
            )
            lows = [calibration.corrected_range(item)[0] for item in items]
            highs = [calibration.corrected_range(item)[1] for item in items]
            corrected_span = (min(lows), max(highs)) if items else None
            boundary = model.boundary_after_segment(segment.index - 1)
            if not segment.calibrated:
                status = "uncalibrated"
                reason = "该重启段尚无独立锚点，位置只能作为未知范围保留，不沿用前段参数"
            elif segment.single_anchor:
                status = "single_anchor"
                reason = "仅有一个可靠锚点：只能确定固定偏移，漂移与重启位置仍保留范围"
            elif (segment.index, segment.index + 1) in contradiction_pairs or (
                segment.index - 1,
                segment.index,
            ) in contradiction_pairs:
                status = "anchor_conflict"
                reason = "与相邻重启段的锚点范围互相矛盾，保留范围而不强行拼接"
            elif boundary is not None and boundary.reset_window is not None:
                status = "boundary_uncertain"
                reason = "重启位置只在推测窗口内确认，段内校时随边界窗口保留不确定性"
            else:
                status = "calibrated"
                reason = "该段具有独立校时依据"
            views.append(
                SegmentView(
                    segment.index,
                    sequences,
                    stamp_span,
                    corrected_span,
                    len(segment.anchors),
                    segment.single_anchor,
                    boundary.confirmed if boundary is not None else True,
                    None if boundary is None else boundary.reset_window,
                    status,
                    reason,
                )
            )
        return tuple(views)

    def _trim_checkpoints(self) -> None:
        if self.sealed_outputs:
            base_time = self.sealed_outputs[-1].time
            self.checkpoints = [
                checkpoint
                for checkpoint in self.checkpoints
                if checkpoint.state.time <= base_time + 1e-9
                and checkpoint.seal_time is not None
            ][:1]
        else:
            self.checkpoints = []

    def _rebuild_tail(
        self,
        events: list[Observation] | None = None,
        *,
        record: bool = True,
    ) -> None:
        if self.arrival_horizon is None:
            return
        if events is None:
            events = self._eligible(self.arrival_horizon)
        if self.sealed_outputs:
            self.estimator.restore(self.sealed_outputs[-1])
            tail = self._tail_events(events)
        else:
            self.estimator = LocalizationEstimator()
            tail = events
        rebuilt: list[StateEstimate] = []
        for observation in tail:
            self.estimator.step(self._retime(observation))
            rebuilt.append(self.estimator.snapshot())
        self.outputs = rebuilt
        if record:
            self._trim_checkpoints()
            self.checkpoints.append(
                Checkpoint(
                    self.estimator.snapshot(),
                    len(events),
                    self.arrival_horizon,
                    self.seal_time,
                    self.calibration,
                    self._visible(self.arrival_horizon),
                    [self._clone_state(item) for item in self.sealed_outputs],
                    dict(self.rejected_observations),
                    self.staged_calibration,
                    self.staged_sensor,
                    list(self.anchor_records.values()),
                    list(self.interpretations.values()),
                    self.comparison_view,
                    list(self.compositions.values()),
                )
            )

    def _restore_anchor_and_rebuild(
        self, events: list[Observation] | None = None
    ) -> None:
        if self.arrival_horizon is not None:
            self._rebuild_tail(events, record=False)
            self.estimator.predict(self.arrival_horizon)

    def _replay_outputs(
        self,
        events: list[Observation],
        estimator: LocalizationEstimator | None = None,
    ) -> list[StateEstimate]:
        estimator = estimator or LocalizationEstimator()
        outputs: list[StateEstimate] = []
        for observation in events:
            estimator.step(self._retime(observation))
            outputs.append(estimator.snapshot())
        if estimator is not self.estimator:
            self.estimator = estimator
        return outputs
