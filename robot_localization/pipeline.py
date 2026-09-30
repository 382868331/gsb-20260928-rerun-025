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


@dataclass(frozen=True, slots=True)
class RecomputeReport:
    sensor: str
    old_delay: float
    new_delay: float
    affected_from: float
    affected_to: float
    recomputed_from: float
    sealed_blocked: tuple[float, ...]


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
        self.outputs = [
            output for output in combined if output.time > self.seal_time + 1e-9
        ]
        self._trim_checkpoints()
        self.checkpoints.append(
            Checkpoint(
                self.sealed_outputs[-1] if self.sealed_outputs else self.estimator.snapshot(),
                len(self.sealed_outputs),
                self.arrival_horizon,
                self.seal_time,
                self.calibration,
                self._visible(self.arrival_horizon),
            )
        )
        return list(self.sealed_outputs)

    def set_sensor_delay(self, sensor: str, delay: float) -> RecomputeReport:
        """Update a fixed delay estimate and recompute only the open interval."""
        delay = float(delay)
        old_delay = self.calibration.delay_for(sensor)
        if abs(delay - old_delay) <= 1e-12:
            interval = self.calibration.affected_interval(sensor, self.received)
            return RecomputeReport(
                sensor,
                old_delay,
                delay,
                interval[0] if interval else 0.0,
                interval[1] if interval else 0.0,
                self.seal_time if self.seal_time is not None else 0.0,
                tuple(),
            )
        interval = self.calibration.affected_interval(sensor, self.received)
        if interval is None:
            self.calibration = self.calibration.with_delay(sensor, delay)
            return RecomputeReport(sensor, old_delay, delay, 0.0, 0.0, 0.0, tuple())
        affected_from, affected_to = interval
        blocked = [
            output.time
            for output in self.sealed_outputs
            if affected_from - 1e-9 <= output.time <= affected_to + 1e-9
        ]
        if blocked:
            raise SealedHistoryError(sensor, interval, blocked)
        self.calibration = self.calibration.with_delay(sensor, delay)
        self._rebuild_tail()
        return RecomputeReport(
            sensor,
            old_delay,
            delay,
            affected_from,
            affected_to,
            self.seal_time if self.seal_time is not None else 0.0,
            tuple(blocked),
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
            len(self._eligible(self.arrival_horizon)) if self.arrival_horizon is not None else 0,
            self.arrival_horizon,
            self.seal_time,
            self.calibration,
            self._visible(self.arrival_horizon),
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
        events = self._eligible(self.arrival_horizon) if self.arrival_horizon is not None else []
        if self.seal_time is not None:
            self.sealed_outputs = self._replay_outputs(events[: checkpoint.event_count])
            base = self.sealed_outputs[-1] if self.sealed_outputs else None
            tail_events = events[checkpoint.event_count :]
            if base is not None:
                self.outputs = self._replay_outputs(
                    tail_events, self._estimator_restored(base)
                )
            else:
                self.outputs = self._replay_outputs(tail_events)
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
        sealed_count = len(self.sealed_outputs)
        if self.sealed_outputs:
            self.estimator.restore(self.sealed_outputs[-1])
            tail = events[sealed_count:]
        else:
            self.estimator = LocalizationEstimator()
            tail = events
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
