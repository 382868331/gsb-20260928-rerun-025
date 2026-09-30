from __future__ import annotations

from dataclasses import dataclass, field

from .estimator import LocalizationEstimator, StateEstimate
from .types import Observation


@dataclass(slots=True)
class Checkpoint:
    state: StateEstimate
    event_count: int
    horizon: float | None = None
    visible_arrival: float | None = None
    sealed_horizon: float = 0.0
    sensor_delays: dict[str, float] = field(default_factory=dict)
    sealed_states: list[StateEstimate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    reason: str
    requested_start: float
    requested_end: float
    recomputed_start: float | None
    recomputed_end: float | None
    sealed_start: float | None
    sealed_end: float | None
    sensor: str | None = None
    message: str = ""


class LocalizationPipeline:
    """Causal processor using corrected observation time and arrival time."""

    def __init__(
        self,
        estimator: LocalizationEstimator | None = None,
        *,
        sensor_delays: dict[str, float] | None = None,
    ) -> None:
        self.estimator = estimator or LocalizationEstimator()
        self.received: list[Observation] = []
        self.checkpoints: list[Checkpoint] = []
        self.horizon: float | None = None
        self.visible_arrival: float | None = None
        self.outputs: list[StateEstimate] = []
        self.sealed_outputs: list[StateEstimate] = []
        self.sealed_horizon: float = 0.0
        self.sensor_delays: dict[str, float] = dict(sensor_delays or {})
        self.revision_records: list[RevisionRecord] = []

    def add_observation(self, observation: Observation) -> None:
        self.received.append(observation)
        if self.horizon is None:
            return
        arrival = self._arrival_time(observation)
        corrected = self.corrected_time(observation)
        if self._is_sealed(observation, corrected):
            self._record_sealed_arrival(observation, corrected)
        elif (
            arrival < self.horizon - 1e-9
            or corrected < self.estimator.time - 1e-9
        ) and corrected <= self.horizon + 1e-9:
            self._rebuild_active("late_observation", corrected, corrected)
        elif arrival <= self.horizon + 1e-9 and corrected <= self.horizon + 1e-9:
            self._rebuild_active(
                "simultaneous_observation",
                corrected,
                corrected,
                record_revision=False,
            )

    def process_until(self, horizon: float) -> list[StateEstimate]:
        if self.horizon is not None and horizon < self.horizon - 1e-9:
            raise ValueError("offline clock cannot move backwards")
        self.horizon = horizon
        self.visible_arrival = horizon
        return self._rebuild_active("horizon_advance", self.sealed_horizon, horizon)

    def corrected_time(self, observation: Observation) -> float:
        return observation.time - self.sensor_delays.get(observation.sensor, 0.0)

    def set_sensor_delay(self, sensor: str, delay: float) -> RevisionRecord:
        if delay < 0.0:
            raise ValueError("sensor delay must be non-negative")
        old_delay = self.sensor_delays.get(sensor, 0.0)
        self.sensor_delays[sensor] = delay
        affected = [
            item
            for item in self.received
            if item.sensor == sensor
            and (
                self.horizon is None
                or self._arrival_time(item) <= self.horizon
            )
        ]
        if not affected:
            record = RevisionRecord(
                reason="sensor_delay_change",
                requested_start=0.0,
                requested_end=0.0,
                recomputed_start=None,
                recomputed_end=None,
                sealed_start=None,
                sealed_end=None,
                sensor=sensor,
                message="no visible observations were affected",
            )
            self.revision_records.append(record)
            return record
        old_times = [item.time - old_delay for item in affected]
        new_times = [item.time - delay for item in affected]
        requested_start = min(min(old_times), min(new_times))
        requested_end = max(max(old_times), max(new_times))
        if self.horizon is not None:
            self._rebuild_active(
                "sensor_delay_change",
                requested_start,
                requested_end,
                sensor=sensor,
                message=f"delay changed from {old_delay} to {delay}",
            )
        else:
            self._revision_record(
                "sensor_delay_change",
                requested_start,
                requested_end,
                sensor,
                f"delay changed from {old_delay} to {delay}",
            )
        return self.revision_records[-1]

    def seal_outputs_through(self, horizon: float) -> list[StateEstimate]:
        if self.horizon is None:
            raise ValueError("cannot seal before processing outputs")
        horizon = min(horizon, self.horizon)
        newly_sealed = [
            state
            for state in self.outputs[len(self.sealed_outputs):]
            if state.time <= horizon + 1e-9 and state.time > self.sealed_horizon + 1e-9
        ]
        self.sealed_outputs.extend(newly_sealed)
        if newly_sealed:
            self.sealed_horizon = newly_sealed[-1].time
        self.outputs = self.sealed_outputs + self.outputs[len(self.sealed_outputs):]
        return newly_sealed

    def state(self) -> StateEstimate:
        return self.estimator.snapshot()

    def save_checkpoint(self) -> Checkpoint:
        checkpoint = Checkpoint(
            self.estimator.snapshot(),
            len(self._eligible(self.horizon or 0.0)),
            self.horizon,
            self.visible_arrival,
            self.sealed_horizon,
            dict(self.sensor_delays),
            list(self.sealed_outputs),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def checkpoint_observations(self) -> list[Observation]:
        return self._eligible(self.horizon or 0.0)

    def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        prior_observations: list[Observation] | None = None,
    ) -> None:
        self.estimator.restore(checkpoint.state)
        self.horizon = checkpoint.horizon if checkpoint.horizon is not None else checkpoint.state.time
        self.visible_arrival = checkpoint.visible_arrival
        self.sealed_horizon = checkpoint.sealed_horizon
        self.sensor_delays = dict(checkpoint.sensor_delays)
        self.sealed_outputs = list(checkpoint.sealed_states)
        if not self.sealed_outputs:
            self.sealed_outputs = [
                state for state in self.outputs if state.time <= self.sealed_horizon + 1e-9
            ]
        self.outputs = self.outputs[: checkpoint.event_count]
        self.checkpoints = [checkpoint]
        if prior_observations is not None:
            self.received = list(prior_observations)
            if len(self._eligible(self.horizon)) != checkpoint.event_count:
                raise ValueError("visible observation count does not match checkpoint")

    def replay(self) -> list[StateEstimate]:
        estimator = LocalizationEstimator()
        sealed_outputs: list[StateEstimate] = []
        if self.horizon is None:
            events = sorted(self.received, key=self._event_order)
        else:
            sealed_outputs = list(self.sealed_outputs)
            if sealed_outputs:
                estimator.restore(sealed_outputs[-1])
            events = self._eligible(self.horizon)
        outputs = [estimator.step(event, self.corrected_time(event)) for event in events]
        if self.horizon is not None and (not outputs or outputs[-1].time < self.horizon - 1e-9):
            outputs.append(estimator.predict(self.horizon))
        return sealed_outputs + outputs

    def _eligible(self, horizon: float) -> list[Observation]:
        return sorted(
            (
                item
                for item in self.received
                if self.corrected_time(item) <= horizon + 1e-9
                and self._arrival_time(item) <= horizon + 1e-9
                and not self._is_sealed(item, self.corrected_time(item))
            ),
            key=self._event_order,
        )

    def _event_order(self, observation: Observation) -> tuple[float, str, int]:
        return (self.corrected_time(observation), observation.sensor, observation.sequence)

    def _arrival_time(self, observation: Observation) -> float:
        if observation.arrival_time is not None:
            return observation.arrival_time
        return self.horizon if self.horizon is not None else 0.0

    def _is_sealed(self, observation: Observation, corrected_time: float) -> bool:
        return (
            corrected_time <= self.sealed_horizon + 1e-9
            or observation.time <= self.sealed_horizon + 1e-9
        )

    def _rebuild_active(
        self,
        reason: str,
        requested_start: float,
        requested_end: float,
        *,
        sensor: str | None = None,
        message: str = "",
        record_revision: bool = True,
    ) -> list[StateEstimate]:
        if self.horizon is None:
            return []
        horizon = self.horizon
        estimator = LocalizationEstimator()
        if self.sealed_outputs:
            estimator.restore(self.sealed_outputs[-1])
        events = self._eligible(horizon)
        active_events = [
            event
            for event in events
            if self.corrected_time(event) > self.sealed_horizon + 1e-9
        ]
        active_outputs = [
            estimator.step(event, self.corrected_time(event))
            for event in active_events
        ]
        if not active_outputs or active_outputs[-1].time < horizon - 1e-9:
            active_outputs.append(estimator.predict(horizon))
        self.estimator = estimator
        self.outputs = self.sealed_outputs + active_outputs
        self.checkpoints.append(
            Checkpoint(
                estimator.snapshot(),
                len(events),
                horizon,
                self.visible_arrival,
                self.sealed_horizon,
                dict(self.sensor_delays),
            )
        )
        if record_revision and reason not in {"horizon_advance", "simultaneous_observation"}:
            self._revision_record(
                reason,
                requested_start,
                requested_end,
                sensor,
                message,
            )
        return active_outputs

    def _revision_record(
        self,
        reason: str,
        requested_start: float,
        requested_end: float,
        sensor: str | None = None,
        message: str = "",
    ) -> RevisionRecord:
        sealed_interval = [
            state.time
            for state in self.sealed_outputs
            if requested_start - 1e-9 <= state.time <= requested_end + 1e-9
        ]
        sealed_start = min(sealed_interval) if sealed_interval else None
        sealed_end = max(sealed_interval) if sealed_interval else None
        recomputed_start = (
            max(requested_start, self.sealed_horizon)
            if requested_end > self.sealed_horizon + 1e-9
            else None
        )
        recomputed_end = requested_end if recomputed_start is not None else None
        if recomputed_end is not None and self.horizon is not None:
            recomputed_end = min(recomputed_end, self.horizon)
        if not message and sealed_start is not None:
            message = "sealed outputs are retained and were not silently replaced"
        elif not message and recomputed_start is not None:
            message = "visible active interval was recomputed"
        record = RevisionRecord(
            reason,
            requested_start,
            requested_end,
            recomputed_start,
            recomputed_end,
            sealed_start,
            sealed_end,
            sensor,
            message,
        )
        self.revision_records.append(record)
        return record

    def _record_sealed_arrival(self, observation: Observation, corrected: float) -> None:
        self._revision_record(
            "sealed_late_arrival",
            corrected,
            corrected,
            observation.sensor,
            "observation arrived after its interval was sealed; retained but not fused",
        )
