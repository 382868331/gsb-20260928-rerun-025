from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from .calibration import DelayCalibration
from .estimator import LocalizationEstimator, StateEstimate
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
    staged_plan: tuple[str, float, float, float] | None = None


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


@dataclass(frozen=True, slots=True)
class ObservationDecision:
    sensor: str
    sequence: int
    stamp: float
    arrival_time: float | None
    corrected_time: float
    classification: str
    reason: str


@dataclass(frozen=True, slots=True)
class DelaySettingView:
    delay: float
    corrected_interval: tuple[float, float] | None
    affected_sequences: tuple[int, ...]
    open_sequences: tuple[int, ...]
    sealed_sequences: tuple[int, ...]
    rejected_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DelayComparison:
    sensor: str
    baseline: DelaySettingView
    candidate: DelaySettingView
    recomputable_range: tuple[float, float] | None
    sealed_history: tuple[float, float] | None
    boundary_state_time: float | None
    decisions: tuple[ObservationDecision, ...]


@dataclass(frozen=True, slots=True)
class ObservationPositionChange:
    sensor: str
    sequence: int
    baseline_time: float
    candidate_time: float
    time_shift: float
    position_delta: float
    simultaneous_with: tuple[tuple[str, int], ...]
    ordering_swapped_with: tuple[tuple[str, int], ...]
    cause: str


@dataclass(frozen=True, slots=True)
class UncertainInterval:
    from_time: float
    to_time: float
    gap: float
    reason: str


@dataclass(frozen=True, slots=True)
class ClockSettingView:
    delay: float
    drift: float
    epoch: float
    corrected_interval: tuple[float, float] | None
    affected_sequences: tuple[int, ...]
    open_sequences: tuple[int, ...]
    sealed_sequences: tuple[int, ...]
    rejected_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClockComparison:
    sensor: str
    baseline: ClockSettingView
    candidate: ClockSettingView
    recomputable_range: tuple[float, float] | None
    sealed_history: tuple[float, float] | None
    boundary_state_time: float | None
    decisions: tuple[ObservationDecision, ...]
    position_changes: tuple[ObservationPositionChange, ...]
    uncertain_intervals: tuple[UncertainInterval, ...]
    simultaneous_groups: tuple[tuple[tuple[str, int], ...], ...]
    applied_identities: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ClockRevisionReport:
    sensor: str
    old_delay: float
    new_delay: float
    old_drift: float
    new_drift: float
    epoch: float
    boundary_state_time: float | None
    adopted_observations: tuple[int, ...]
    rejected_observations: tuple[ObservationDecision, ...]
    position_changes: tuple[ObservationPositionChange, ...]
    uncertain_intervals: tuple[UncertainInterval, ...]
    simultaneous_groups: tuple[tuple[tuple[str, int], ...], ...]
    applied_identities: tuple[tuple[str, int], ...]


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
        self.staged_clock_plan: tuple[str, float, float, float] | None = None
        self.staged_clock: DelayCalibration | None = None
        self.staged_clock_comparison: ClockComparison | None = None

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
        # Re-sealing freezes more history; an unpublished draft stays the same
        # plan, but its boundary explanation is refreshed against the new seal.
        if self.staged_clock_plan is not None:
            sensor, delay, drift, epoch = self.staged_clock_plan
            self.staged_clock = self.calibration.with_clock(sensor, delay, drift, epoch)
            self.staged_clock_comparison = self._build_clock_comparison(
                sensor, self.staged_clock
            )
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
                self.staged_clock_plan,
            )
        )
        return list(self.sealed_outputs)

    def stage_sensor_delay(self, sensor: str, delay: float) -> DelayComparison:
        """Prepare a delay revision without publishing it."""
        delay = float(delay)
        candidate = self.calibration.with_delay(sensor, delay)
        comparison = self.compare_calibrations(sensor, self.calibration, candidate)
        self.staged_calibration = candidate
        self.staged_comparison = comparison
        return comparison

    def discard_staged_delay(self) -> None:
        """Discard a delay revision that has not been published."""
        self.staged_calibration = None
        self.staged_comparison = None
        self.staged_clock = None
        self.staged_clock_plan = None
        self.staged_clock_comparison = None

    def publish_staged_delay(self) -> RecomputeReport:
        if self.staged_calibration is None or self.staged_comparison is None:
            raise RuntimeError("no staged delay revision to publish")
        comparison = self.staged_comparison
        old_delay = comparison.baseline.delay
        new_delay = comparison.candidate.delay
        self.calibration = self.staged_calibration
        for decision in comparison.decisions:
            if decision.classification == "crossing_rejected":
                self.rejected_observations[(decision.sensor, decision.sequence)] = decision.reason
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
        )

    def set_sensor_delay(self, sensor: str, delay: float) -> RecomputeReport:
        """Publish a fixed-delay update and recompute only the open interval."""
        self.stage_sensor_delay(sensor, delay)
        return self.publish_staged_delay()

    def compare_sensor_delays(self, sensor: str, candidate_delay: float) -> DelayComparison:
        candidate = self.calibration.with_delay(sensor, candidate_delay)
        return self.compare_calibrations(sensor, self.calibration, candidate)

    # ----------------------------------------------------- drifting clock plans

    def stage_clock_revision(
        self,
        sensor: str,
        delay: float,
        drift: float = 0.0,
        epoch: float = 0.0,
    ) -> ClockComparison:
        """Prepare an offset-plus-drift clock revision without publishing it.

        The same draft is retained across saves and re-opens (checkpoints)
        until it is published with :meth:`publish_staged_clock_revision` or
        withdrawn with :meth:`discard_staged_delay`.
        """
        candidate = self.calibration.with_clock(sensor, delay, drift, epoch)
        comparison = self._build_clock_comparison(sensor, candidate)
        self.staged_clock = candidate
        self.staged_clock_plan = (sensor, float(delay), float(drift), float(epoch))
        self.staged_clock_comparison = comparison
        return comparison

    def has_pending_clock_revision(self) -> bool:
        return self.staged_clock_plan is not None

    def pending_clock_revision(self) -> tuple[str, float, float, float] | None:
        return self.staged_clock_plan

    def publish_staged_clock_revision(self) -> ClockRevisionReport:
        if (
            self.staged_clock is None
            or self.staged_clock_comparison is None
            or self.staged_clock_plan is None
        ):
            raise RuntimeError("no staged clock revision to publish")
        comparison = self.staged_clock_comparison
        sensor, delay, drift, epoch = self.staged_clock_plan
        old_model = self.calibration.model_for(sensor)
        for decision in comparison.decisions:
            if decision.classification == "crossing_rejected":
                self.rejected_observations[(decision.sensor, decision.sequence)] = decision.reason
        self.calibration = self.staged_clock
        self.staged_clock = None
        self.staged_clock_plan = None
        self.staged_clock_comparison = None
        self._restore_anchor_and_rebuild()
        return ClockRevisionReport(
            sensor=sensor,
            old_delay=old_model.delay,
            new_delay=delay,
            old_drift=old_model.drift,
            new_drift=drift,
            epoch=epoch,
            boundary_state_time=comparison.boundary_state_time,
            adopted_observations=comparison.candidate.open_sequences,
            rejected_observations=tuple(
                item
                for item in comparison.decisions
                if item.classification == "crossing_rejected"
            ),
            position_changes=comparison.position_changes,
            uncertain_intervals=comparison.uncertain_intervals,
            simultaneous_groups=comparison.simultaneous_groups,
            applied_identities=comparison.applied_identities,
        )

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
            and (item.sensor, item.sequence) not in self.rejected_observations
        ]
        decisions = tuple(
            self._observation_decision(item, baseline, candidate)
            for item in affected
        )
        baseline_view = self._setting_view(sensor, baseline.delay_for(sensor), affected, baseline, candidate)
        candidate_view = self._setting_view(sensor, candidate.delay_for(sensor), affected, baseline, candidate)
        open_times = [
            candidate.corrected_time(item)
            for item in affected
            if self._observation_decision(item, baseline, candidate).classification
            in {"open_recomputed", "open_adopted"}
        ]
        if open_times:
            recomputable_range = (min(open_times), max(open_times))
        elif self.seal_time is not None:
            recomputable_range = (self.seal_time, self.seal_time)
        else:
            recomputable_range = None
        sealed_history = (0.0, self.seal_time) if self.seal_time is not None else None
        return DelayComparison(
            sensor,
            baseline_view,
            candidate_view,
            recomputable_range,
            sealed_history,
            self.sealed_outputs[-1].time if self.sealed_outputs else self.seal_time,
            decisions,
        )

    # ------------------------------------------------------------- inspection

    # ------------------------------------------------------- clock comparison

    GAP_TIMEOUT = 0.3

    def _build_clock_comparison(
        self,
        sensor: str,
        candidate: DelayCalibration,
    ) -> ClockComparison:
        """Build the side-by-side original-vs-retimed explanation."""
        visible = self._visible(self.arrival_horizon)
        affected = [
            item
            for item in visible
            if item.sensor == sensor
            and (item.sensor, item.sequence) not in self.rejected_observations
        ]
        decisions = tuple(
            self._observation_decision(item, self.calibration, candidate)
            for item in affected
        )
        baseline_view = self._clock_view(sensor, affected, self.calibration, candidate)
        candidate_view = self._clock_view(sensor, affected, candidate, candidate)

        candidate_order, applied_identities = self._ordered_for(
            candidate, include_crossing_rejected=False
        )
        baseline_order, _ = self._ordered_for(
            self.calibration, include_crossing_rejected=False
        )
        rejected_keys = {
            (item.sensor, item.sequence)
            for item in decisions
            if item.classification == "crossing_rejected"
        }
        candidate_times = {
            (item.sensor, item.sequence): candidate.corrected_time(item)
            for item in candidate_order
        }
        simultaneous_groups = self._simultaneous_groups(candidate_order, candidate_times)
        swaps = self._ordering_swaps(baseline_order, candidate_order)
        baseline_states, candidate_states = self._dry_run_pair(candidate)
        position_changes = tuple(
            change
            for observation in affected
            if (observation.sensor, observation.sequence) not in rejected_keys
            for change in [
                self._position_change(
                    observation,
                    candidate,
                    candidate_order,
                    simultaneous_groups,
                    swaps,
                    baseline_states,
                    candidate_states,
                )
            ]
        )
        uncertain_intervals = self._uncertain_intervals(
            sensor, candidate_order, candidate
        )
        open_times = [state.time for state in candidate_states]
        if open_times:
            recomputable_range = (min(open_times), max(open_times))
        elif self.seal_time is not None:
            recomputable_range = (self.seal_time, self.seal_time)
        else:
            recomputable_range = None
        return ClockComparison(
            sensor=sensor,
            baseline=baseline_view,
            candidate=candidate_view,
            recomputable_range=recomputable_range,
            sealed_history=(0.0, self.seal_time) if self.seal_time is not None else None,
            boundary_state_time=(
                self.sealed_outputs[-1].time if self.sealed_outputs else self.seal_time
            ),
            decisions=decisions,
            position_changes=position_changes,
            uncertain_intervals=uncertain_intervals,
            simultaneous_groups=simultaneous_groups,
            applied_identities=tuple(sorted(applied_identities)),
        )

    def _ordered_for(
        self,
        calibration: DelayCalibration,
        *,
        include_crossing_rejected: bool,
    ) -> tuple[list[Observation], set[tuple[str, int]]]:
        """Return events sorted on *corrected* time with deterministic ties.

        Ties are broken by sensor name then sequence, so observations from
        different sources that become simultaneous still keep distinct
        identities: each is applied exactly once and none is merged away.
        """
        events: list[Observation] = []
        identities: set[tuple[str, int]] = set()
        for item in self._visible(self.arrival_horizon):
            key = (item.sensor, item.sequence)
            if key in self.rejected_observations or key in identities:
                continue
            if (
                not include_crossing_rejected
                and self.seal_time is not None
                and calibration.corrected_time(item) <= self.seal_time + 1e-9
            ):
                continue
            events.append(item)
            identities.add(key)
        events.sort(
            key=lambda item: (
                calibration.corrected_time(item),
                item.sensor,
                item.sequence,
            )
        )
        return events, identities

    @staticmethod
    def _simultaneous_groups(
        ordered: list[Observation],
        times: dict[tuple[str, int], float],
    ) -> tuple[tuple[tuple[str, int], ...], ...]:
        """Group adjacent events sharing one corrected time.

        ``ordered`` is already sorted by corrected time, so equal-time events
        are adjacent; every group has at least two distinct identities.
        """
        groups: list[tuple[tuple[str, int], ...]] = []
        run: list[Observation] = []
        previous_time: float | None = None
        for item in ordered:
            current = times[(item.sensor, item.sequence)]
            if previous_time is not None and abs(current - previous_time) <= 1e-9:
                run.append(item)
            else:
                run = [item]
            if len(run) >= 2:
                groups.append(
                    tuple((member.sensor, member.sequence) for member in run)
                )
            previous_time = current
        return tuple(groups)

    def _ordering_swaps(
        self,
        baseline_order: list[Observation],
        candidate_order: list[Observation],
    ) -> dict[tuple[str, int], tuple[tuple[str, int], ...]]:
        baseline_rank = {
            (item.sensor, item.sequence): rank
            for rank, item in enumerate(baseline_order)
        }
        candidate_rank = {
            (item.sensor, item.sequence): rank
            for rank, item in enumerate(candidate_order)
        }
        swaps: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {}
        common = baseline_rank.keys() & candidate_rank.keys()
        for key in common:
            swapped = tuple(
                sorted(
                    other
                    for other in common
                    if other != key
                    and (
                        baseline_rank[key] - baseline_rank[other]
                    )
                    * (
                        candidate_rank[key] - candidate_rank[other]
                    )
                    < 0
                )
            )
            if swapped:
                swaps[key] = swapped
        return swaps

    def _dry_run_pair(
        self,
        candidate: DelayCalibration,
    ) -> tuple[list[StateEstimate], list[StateEstimate]]:
        baseline_events, _ = self._ordered_for(
            self.calibration, include_crossing_rejected=False
        )
        candidate_events, _ = self._ordered_for(
            candidate, include_crossing_rejected=False
        )
        return (
            self._dry_run(baseline_events, self.calibration),
            self._dry_run(candidate_events, candidate),
        )

    def _dry_run(
        self,
        events: list[Observation],
        calibration: DelayCalibration,
    ) -> list[StateEstimate]:
        if self.sealed_outputs:
            estimator = self._estimator_restored(self.sealed_outputs[-1])
        else:
            estimator = LocalizationEstimator()
        states: list[StateEstimate] = []
        for observation in events:
            corrected = calibration.corrected_time(observation)
            if self.seal_time is not None and corrected <= self.seal_time + 1e-9:
                continue
            estimator.step(self._retime_with(observation, calibration))
            states.append(estimator.snapshot())
        return states

    @staticmethod
    def _state_at_time(
        states: list[StateEstimate],
        time: float,
    ) -> StateEstimate | None:
        eligible = [state for state in states if state.time <= time + 1e-9]
        return eligible[-1] if eligible else None

    def _position_change(
        self,
        observation: Observation,
        candidate: DelayCalibration,
        candidate_order: list[Observation],
        simultaneous_groups: tuple[tuple[tuple[str, int], ...], ...],
        swaps: dict[tuple[str, int], tuple[tuple[str, int], ...]],
        baseline_states: list[StateEstimate],
        candidate_states: list[StateEstimate],
    ) -> ObservationPositionChange:
        key = (observation.sensor, observation.sequence)
        old_time = self.calibration.corrected_time(observation)
        new_time = candidate.corrected_time(observation)
        candidate_state = self._state_after(candidate_states, observation, candidate)
        baseline_reference = self._state_at_time(baseline_states, new_time)
        delta = 0.0
        if candidate_state is not None and baseline_reference is not None:
            delta = math.hypot(
                candidate_state.x - baseline_reference.x,
                candidate_state.y - baseline_reference.y,
            )
        simultaneous = tuple(
            member
            for group in simultaneous_groups
            if key in group
            for member in group
            if member != key
        )
        return ObservationPositionChange(
            sensor=observation.sensor,
            sequence=observation.sequence,
            baseline_time=old_time,
            candidate_time=new_time,
            time_shift=new_time - old_time,
            position_delta=delta,
            simultaneous_with=simultaneous,
            ordering_swapped_with=swaps.get(key, ()),
            cause=(
                "校时后观测落在轨迹时间的新位置，位置变化来自时钟漂移校时"
                if delta > 1e-9
                else "校时改变观测时间，但该观测更新时刻的位置估计未发生数值变化"
            ),
        )

    @staticmethod
    def _state_after(
        states: list[StateEstimate],
        observation: Observation,
        calibration: DelayCalibration,
    ) -> StateEstimate | None:
        target = calibration.corrected_time(observation)
        matches = [
            state for state in states if abs(state.time - target) <= 1e-9
        ]
        return matches[-1] if matches else None

    def _uncertain_intervals(
        self,
        sensor: str,
        ordered: list[Observation],
        calibration: DelayCalibration,
    ) -> tuple[UncertainInterval, ...]:
        intervals: list[UncertainInterval] = []
        boundary = self.seal_time if self.seal_time is not None else 0.0
        # Intervals with no observation from *any* source: the open estimate
        # can only predict there, so uncertainty is intrinsic, not retiming.
        previous_time = boundary
        for observation in ordered:
            time = calibration.corrected_time(observation)
            gap = time - previous_time
            if gap > self.GAP_TIMEOUT:
                intervals.append(
                    UncertainInterval(
                        previous_time,
                        time,
                        gap,
                        "该区间没有任何来源的观测，重估只能依赖预测，位置仍因缺少观测而不确定，不能归因于校时",
                    )
                )
            previous_time = time
        horizon = self.arrival_horizon
        if horizon is not None and horizon - previous_time > self.GAP_TIMEOUT:
            intervals.append(
                UncertainInterval(
                    previous_time,
                    horizon,
                    horizon - previous_time,
                    "开放尾段尚无观测，位置仍因缺少观测而不确定，与校时无关",
                )
            )
        # Affected-sensor outages: another source may still constrain pose,
        # but the sensor-specific degrees of freedom remain unobserved.
        sensor_times = [
            calibration.corrected_time(item)
            for item in ordered
            if item.sensor == sensor
        ]
        if not sensor_times:
            return tuple(intervals)
        previous = max(boundary, sensor_times[0])
        if sensor_times[0] - boundary > self.GAP_TIMEOUT:
            intervals.append(
                UncertainInterval(
                    boundary,
                    sensor_times[0],
                    sensor_times[0] - boundary,
                    "校时开始处该传感器尚无观测，相关自由度仍因缺少观测而不确定",
                )
            )
        for time in sensor_times:
            if time - previous > self.GAP_TIMEOUT:
                intervals.append(
                    UncertainInterval(
                        previous,
                        time,
                        time - previous,
                        "该传感器在此区间断流，其他来源可部分约束，但相关位置仍因缺少观测而不确定",
                    )
                )
            previous = time
        if horizon is not None and horizon - previous > self.GAP_TIMEOUT:
            intervals.append(
                UncertainInterval(
                    previous,
                    horizon,
                    horizon - previous,
                    "该传感器在开放尾段没有新观测，位置仍因缺少观测而不确定，与校时无关",
                )
            )
        ordered = sorted(intervals, key=lambda item: (item.from_time, item.to_time))
        merged: list[UncertainInterval] = []
        for interval in ordered:
            if merged and interval.from_time <= merged[-1].to_time + 1e-6:
                previous = merged[-1]
                merged[-1] = UncertainInterval(
                    from_time=min(previous.from_time, interval.from_time),
                    to_time=max(previous.to_time, interval.to_time),
                    gap=max(previous.gap, interval.gap),
                    reason=interval.reason
                    if len(interval.reason) >= len(previous.reason)
                    else previous.reason,
                )
            else:
                merged.append(interval)
        return tuple(merged)

    def _clock_view(
        self,
        sensor: str,
        affected: list[Observation],
        calibration: DelayCalibration,
        candidate: DelayCalibration,
    ) -> ClockSettingView:
        model = calibration.model_for(sensor)
        times = [calibration.corrected_time(item) for item in affected]
        open_sequences: list[int] = []
        sealed_sequences: list[int] = []
        rejected_sequences: list[int] = []
        for item in affected:
            decision = self._observation_decision(item, self.calibration, candidate)
            corrected = calibration.corrected_time(item)
            if self.seal_time is not None and corrected <= self.seal_time + 1e-9:
                sealed_sequences.append(item.sequence)
            if (
                calibration is candidate
                and decision.classification == "crossing_rejected"
            ):
                rejected_sequences.append(item.sequence)
            if corrected > (self.seal_time if self.seal_time is not None else -1e300) + 1e-9:
                open_sequences.append(item.sequence)
        return ClockSettingView(
            delay=model.delay,
            drift=model.drift,
            epoch=model.epoch,
            corrected_interval=(min(times), max(times)) if times else None,
            affected_sequences=tuple(item.sequence for item in affected),
            open_sequences=tuple(open_sequences),
            sealed_sequences=tuple(sealed_sequences),
            rejected_sequences=tuple(rejected_sequences),
        )

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
            self.staged_clock_plan,
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
        self.staged_clock = None
        self.staged_clock_comparison = None
        self.staged_clock_plan = checkpoint.staged_plan
        if checkpoint.staged_plan is not None:
            sensor, delay, drift, epoch = checkpoint.staged_plan
            self.staged_clock = self.calibration.with_clock(sensor, delay, drift, epoch)
        self.checkpoints = [checkpoint]
        if self.staged_clock is not None:
            self.staged_clock_comparison = self._build_clock_comparison(
                self.staged_clock_plan[0], self.staged_clock
            )

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

    @staticmethod
    def _retime_with(
        observation: Observation,
        calibration: DelayCalibration,
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

    def _sort_key(self, observation: Observation):
        return (
            self.calibration.corrected_time(observation),
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
        visible = [
            item
            for item in self._visible(arrival_horizon)
            if (item.sensor, item.sequence) not in self.rejected_observations
        ]
        return sorted(visible, key=lambda item: self._sort_key(item))

    def _tail_events(self, events: list[Observation] | None = None) -> list[Observation]:
        events = events if events is not None else self._eligible(self.arrival_horizon)
        if self.seal_time is None:
            return events
        return [
            item
            for item in events
            if self.calibration.corrected_time(item) > self.seal_time + 1e-9
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
        return ObservationDecision(
            observation.sensor,
            observation.sequence,
            observation.time,
            observation.arrival_time,
            new_time,
            classification,
            reason,
        )

    def _setting_view(
        self,
        sensor: str,
        delay: float,
        affected: list[Observation],
        baseline: DelayCalibration,
        candidate: DelayCalibration,
    ) -> DelaySettingView:
        times = [item.time - delay for item in affected]
        open_sequences = []
        sealed_sequences = []
        rejected_sequences = []
        for item in affected:
            decision = self._observation_decision(item, baseline, candidate)
            corrected = item.time - delay
            if self.seal_time is not None and corrected <= self.seal_time + 1e-9:
                sealed_sequences.append(item.sequence)
            if decision.classification == "crossing_rejected" and delay == candidate.delay_for(sensor):
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
        )

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
