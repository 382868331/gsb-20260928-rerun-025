from __future__ import annotations

from dataclasses import dataclass, field, fields

from .calibration import DelayCalibration, SensorClockModel
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
    staged_calibration: DelayCalibration | None = None
    staged_sensor: str | None = None


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
        )

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
        if self.staged_calibration is not None and self.staged_sensor:
            self.staged_comparison = self.compare_calibrations(
                self.staged_sensor, self.calibration, self.staged_calibration
            )
        else:
            self.staged_comparison = None
        self.checkpoints = [checkpoint]

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
