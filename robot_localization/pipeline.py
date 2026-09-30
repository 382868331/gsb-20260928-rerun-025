from __future__ import annotations

from dataclasses import dataclass, field, fields

from .calibration import DelayCalibration
from .estimator import LocalizationEstimator, StateEstimate
from .types import Observation


@dataclass(frozen=True, slots=True)
class ObservationImpact:
    sensor: str
    sequence: int
    stamp_time: float
    arrival_time: float | None
    old_corrected_time: float
    new_corrected_time: float
    action: str
    reason: str


@dataclass(slots=True)
class Checkpoint:
    state: StateEstimate
    event_count: int
    arrival_horizon: float | None = None
    seal_time: float | None = None
    calibration: DelayCalibration | None = None
    visible_observations: list[Observation] = field(default_factory=list)
    sealed_at_arrival: float | None = None
    pending_corrections: list[tuple[str, float, float]] = field(default_factory=list)
    rejected_observations: list[ObservationImpact] = field(default_factory=list)
    held_observation_keys: list[tuple[str, int, float | None]] = field(default_factory=list)
    sealed_outputs: list[StateEstimate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RecomputeReport:
    sensor: str
    old_delay: float
    new_delay: float
    affected_from: float
    affected_to: float
    recomputed_from: float
    sealed_blocked: tuple[float, ...]
    recomputed_to: float | None = None
    immutable_history: tuple[float, ...] = ()
    affected_observations: tuple[ObservationImpact, ...] = ()
    rejected_observations: tuple[ObservationImpact, ...] = ()
    accepted_observations: tuple[ObservationImpact, ...] = ()


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
        self.sealed_at_arrival: float | None = None
        self.sealed_outputs: list[StateEstimate] = []
        self.outputs: list[StateEstimate] = []
        self.held_observation_keys: set[tuple[str, int, float | None]] = set()
        self.late_rejected_observations: dict[
            tuple[str, int, float | None], ObservationImpact
        ] = {}
        self.pending_corrections: list[tuple[str, float, float]] = []

    # ------------------------------------------------------------------ ingest

    def add_observation(self, observation: Observation) -> None:
        self.received.append(observation)
        arrival = self.arrival_horizon
        if arrival is not None and observation.observed_at() <= arrival + 1e-9:
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
                self.sealed_at_arrival,
                list(self.pending_corrections),
                list(self.late_rejected_observations.values()),
                list(self.held_observation_keys),
                list(self.sealed_outputs),
            )
        )
        return list(self.outputs)

    # ------------------------------------------------------------------ sealing

    def seal_until(self, time: float) -> list[StateEstimate]:
        """Freeze every output at or before ``time``; those values never change."""
        if self.seal_time is not None and time < self.seal_time - 1e-9:
            raise ValueError("seal boundary cannot move backwards")
        combined = self.all_outputs()
        keep = [output for output in combined if output.time <= time + 1e-9]
        self.sealed_outputs = keep
        self.seal_time = keep[-1].time if keep else time
        self.sealed_at_arrival = self.arrival_horizon
        self.outputs = [
            output for output in combined if output.time > self.seal_time + 1e-9
        ]
        self.pending_corrections.clear()
        self._rebuild_tail()
        self._trim_checkpoints()
        self.checkpoints.append(
            Checkpoint(
                self.sealed_outputs[-1] if self.sealed_outputs else self.estimator.snapshot(),
                len(self.sealed_outputs),
                self.arrival_horizon,
                self.seal_time,
                self.calibration,
                self._visible(self.arrival_horizon),
                self.sealed_at_arrival,
                list(self.pending_corrections),
                list(self.late_rejected_observations.values()),
                list(self.held_observation_keys),
                list(self.sealed_outputs),
            )
        )
        return list(self.sealed_outputs)

    def set_sensor_delay(self, sensor: str, delay: float) -> RecomputeReport:
        """Update a fixed delay estimate and recompute only the open interval."""
        delay = float(delay)
        old_delay = self.calibration.delay_for(sensor)
        report = self.preview_delay_change(sensor, delay)
        if abs(delay - old_delay) <= 1e-12:
            return report
        self.calibration = self.calibration.with_delay(sensor, delay)
        for impact in report.rejected_observations:
            key = (impact.sensor, impact.sequence, impact.arrival_time)
            if impact.new_corrected_time <= (self.seal_time or float("inf")) + 1e-9:
                self.late_rejected_observations[key] = impact
            if (
                self.seal_time is not None
                and impact.old_corrected_time <= self.seal_time + 1e-9
                and impact.new_corrected_time > self.seal_time + 1e-9
            ):
                self.held_observation_keys.add(key)
        for impact in report.accepted_observations:
            key = (impact.sensor, impact.sequence, impact.arrival_time)
            self.held_observation_keys.discard(key)
            self.late_rejected_observations.pop(key, None)
        self.pending_corrections.append((sensor, old_delay, delay))
        self._rebuild_tail()
        return report

    def preview_delay_change(
        self, sensor: str, delay: float, baseline_delay: float | None = None
    ) -> RecomputeReport:
        """Compare a candidate delay with the current setting without mutating state."""
        delay = float(delay)
        old_delay = (
            self.calibration.delay_for(sensor)
            if baseline_delay is None
            else float(baseline_delay)
        )
        visible = self._visible(self.arrival_horizon)
        affected = [item for item in visible if item.sensor == sensor]
        if not affected:
            return RecomputeReport(
                sensor, old_delay, delay, 0.0, 0.0,
                self.seal_time if self.seal_time is not None else 0.0,
                tuple(),
            )
        old_times = [item.time - old_delay for item in affected]
        new_times = [item.time - delay for item in affected]
        impacts: list[ObservationImpact] = []
        accepted: list[ObservationImpact] = []
        rejected: list[ObservationImpact] = []
        seal = self.seal_time
        for observation, old_time, new_time in zip(affected, old_times, new_times, strict=True):
            action, reason = self._classify_delay_impact(observation, old_time, new_time)
            impact = ObservationImpact(
                observation.sensor,
                observation.sequence,
                observation.time,
                observation.arrival_time,
                old_time,
                new_time,
                action,
                reason,
            )
            impacts.append(impact)
            (accepted if action == "recompute" else rejected).append(impact)
        immutable = tuple(output.time for output in self.sealed_outputs)
        blocked = tuple(
            output.time
            for output in self.sealed_outputs
            if min(new_times) - 1e-9 <= output.time <= max(new_times) + 1e-9
        )
        open_new_times = [item.new_corrected_time for item in accepted]
        return RecomputeReport(
            sensor,
            old_delay,
            delay,
            min(old_times),
            max(old_times),
            seal if seal is not None else 0.0,
            blocked,
            recomputed_to=max(open_new_times) if open_new_times else None,
            immutable_history=immutable,
            affected_observations=tuple(impacts),
            rejected_observations=tuple(rejected),
            accepted_observations=tuple(accepted),
        )

    def compare_delay_settings(
        self, sensor: str, left_delay: float, right_delay: float
    ) -> tuple[RecomputeReport, RecomputeReport]:
        """Return side-by-side plans for two delay choices against current data."""
        return (
            self.preview_delay_change(sensor, left_delay),
            self.preview_delay_change(sensor, right_delay),
        )

    def undo_last_correction(self) -> RecomputeReport | None:
        """Undo the newest correction that has not been sealed or published."""
        if not self.pending_corrections:
            return None
        sensor, previous_delay, current_delay = self.pending_corrections.pop()
        reverse_report = self.preview_delay_change(
            sensor, previous_delay, baseline_delay=current_delay
        )
        self.calibration = self.calibration.with_delay(sensor, previous_delay)
        for impact in reverse_report.rejected_observations:
            key = (impact.sensor, impact.sequence, impact.arrival_time)
            if impact.new_corrected_time <= (self.seal_time or float("inf")) + 1e-9:
                self.late_rejected_observations[key] = impact
            if (
                self.seal_time is not None
                and impact.old_corrected_time <= self.seal_time + 1e-9
                and impact.new_corrected_time > self.seal_time + 1e-9
            ):
                self.held_observation_keys.add(key)
        for impact in reverse_report.accepted_observations:
            key = (impact.sensor, impact.sequence, impact.arrival_time)
            self.held_observation_keys.discard(key)
            self.late_rejected_observations.pop(key, None)
        self._rebuild_tail()
        return reverse_report

    def _classify_delay_impact(
        self, observation: Observation, old_time: float, new_time: float
    ) -> tuple[str, str]:
        seal = self.seal_time
        if seal is None:
            return "recompute", "no sealed history; corrected observation is in the open interval"
        known_at_seal = (
            self.sealed_at_arrival is not None
            and observation.observed_at() <= self.sealed_at_arrival + 1e-9
        )
        if new_time <= seal + 1e-9:
            if known_at_seal:
                return (
                    "reject",
                    "new corrected time is in sealed history; "
                    "the existing boundary state remains authoritative",
                )
            return (
                "reject",
                "late observation corrects to a sealed time and cannot be "
                "inserted without rewriting history",
            )
        if old_time <= seal + 1e-9 and known_at_seal:
            return (
                "reject",
                "observation cannot migrate out of the sealed anchor; reusing "
                "it in the tail would double-count it",
            )
        if not known_at_seal:
            return (
                "recompute",
                "late observation was absent when sealing occurred and its "
                "corrected time lies in the open interval",
            )
        if old_time <= seal + 1e-9:
            return (
                "recompute",
                "both corrected times remain after the sealed boundary anchor",
            )
        return "recompute", "both corrected times remain after the sealed boundary anchor"

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
            len(self._eligible(self.arrival_horizon)) if self.arrival_horizon is not None else 0,
            self.arrival_horizon,
            self.seal_time,
            self.calibration,
            self._visible(self.arrival_horizon),
            self.sealed_at_arrival,
            list(self.pending_corrections),
            list(self.late_rejected_observations.values()),
            list(self.held_observation_keys),
            list(self.sealed_outputs),
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
        self.sealed_at_arrival = checkpoint.sealed_at_arrival
        self.pending_corrections = list(checkpoint.pending_corrections)
        self.held_observation_keys = set(checkpoint.held_observation_keys)
        self.late_rejected_observations = {
            (item.sensor, item.sequence, item.arrival_time): item
            for item in checkpoint.rejected_observations
        }
        visible = (
            prior_observations
            if prior_observations is not None
            else list(checkpoint.visible_observations)
        )
        self.received = list(visible)
        events = self._eligible(self.arrival_horizon) if self.arrival_horizon is not None else []
        if self.seal_time is not None:
            self.sealed_outputs = list(checkpoint.sealed_outputs)
            base = self.sealed_outputs[-1] if self.sealed_outputs else checkpoint.state
            tail_events = [
                event for event in events if self._is_open_event(event, base)
            ]
            self.outputs = self._replay_outputs(
                tail_events, self._estimator_restored(base)
            )
        else:
            self.sealed_outputs = []
            self.estimator = self._estimator_restored(checkpoint.state)
            self.outputs = self._replay_outputs(
                events[checkpoint.event_count :],
                self.estimator,
            )
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
        return sorted(self._visible(arrival_horizon), key=lambda item: self._sort_key(item))

    def _observation_key(self, observation: Observation) -> tuple[str, int, float | None]:
        return (observation.sensor, observation.sequence, observation.arrival_time)

    def _is_open_event(
        self, observation: Observation, anchor: StateEstimate | None = None
    ) -> bool:
        corrected_time = self.calibration.corrected_time(observation)
        if anchor is not None and corrected_time <= anchor.time + 1e-9:
            return False
        key = self._observation_key(observation)
        return key not in self.held_observation_keys and key not in self.late_rejected_observations

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

    def _rebuild_tail(self, events: list[Observation] | None = None) -> None:
        if self.arrival_horizon is None:
            return
        if events is None:
            events = self._eligible(self.arrival_horizon)
        if self.sealed_outputs:
            self.estimator.restore(self.sealed_outputs[-1])
            tail = [
                event for event in events
                if self._is_open_event(event, self.sealed_outputs[-1])
            ]
        else:
            self.estimator = LocalizationEstimator()
            tail = [event for event in events if self._is_open_event(event)]
        rebuilt: list[StateEstimate] = []
        for observation in tail:
            self.estimator.step(self._retime(observation))
            rebuilt.append(self.estimator.snapshot())
        self.outputs = rebuilt
        self._trim_checkpoints()
        self.checkpoints.append(
            Checkpoint(
                self.estimator.snapshot(),
                len(events),
                self.arrival_horizon,
                self.seal_time,
                self.calibration,
                self._visible(self.arrival_horizon),
                self.sealed_at_arrival,
                list(self.pending_corrections),
                list(self.late_rejected_observations.values()),
                list(self.held_observation_keys),
                list(self.sealed_outputs),
            )
        )

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
