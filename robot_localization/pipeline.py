from __future__ import annotations

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
        corrected = self.calibration.corrected_time(observation)
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
