from __future__ import annotations

from dataclasses import dataclass

from .estimator import LocalizationEstimator, StateEstimate
from .types import Observation


@dataclass(slots=True)
class Checkpoint:
    state: StateEstimate
    event_count: int


class LocalizationPipeline:
    """Offline causal event processor with rollback for late messages."""

    def __init__(self, estimator: LocalizationEstimator | None = None) -> None:
        self.estimator = estimator or LocalizationEstimator()
        self.received: list[Observation] = []
        self.checkpoints: list[Checkpoint] = []
        self.horizon: float | None = None
        self.outputs: list[StateEstimate] = []

    def add_observation(self, observation: Observation) -> None:
        self.received.append(observation)
        if self.horizon is not None and observation.time <= self.horizon + 1e-9:
            self._rebuild_from(observation.time)

    def process_until(self, horizon: float) -> list[StateEstimate]:
        if self.horizon is not None and horizon < self.horizon - 1e-9:
            raise ValueError("offline clock cannot move backwards")
        self.horizon = horizon
        events = self._eligible(horizon)
        new_outputs: list[StateEstimate] = []
        for observation in events[self._last_event_count():]:
            self.estimator.step(observation)
            new_outputs.append(self.estimator.snapshot())
        if not new_outputs:
            self.estimator.predict(horizon)
        snapshot = self.estimator.snapshot()
        self.outputs = self.outputs[: self._last_event_count()] + new_outputs
        self.checkpoints.append(Checkpoint(snapshot, len(events)))
        return new_outputs

    def state(self) -> StateEstimate:
        return self.estimator.snapshot()

    def save_checkpoint(self) -> Checkpoint:
        checkpoint = Checkpoint(self.estimator.snapshot(), self._last_event_count())
        self.checkpoints.append(checkpoint)
        return checkpoint

    def checkpoint_observations(self) -> list[Observation]:
        horizon = self.horizon or 0.0
        return self._eligible(horizon)[: self._last_event_count()]

    def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        prior_observations: list[Observation] | None = None,
    ) -> None:
        self.estimator.restore(checkpoint.state)
        self.horizon = checkpoint.state.time
        self.outputs = self.outputs[: checkpoint.event_count]
        self.checkpoints = [checkpoint]
        if prior_observations is not None:
            self.received = sorted(
                prior_observations,
                key=lambda item: (item.time, item.sensor, item.sequence),
            )
            if len(self.received) != checkpoint.event_count:
                raise ValueError("prior observation count does not match checkpoint")

    def replay(self) -> list[StateEstimate]:
        estimator = LocalizationEstimator()
        horizon = self.horizon
        if horizon is None:
            events = sorted(
                self.received,
                key=lambda item: (item.time, item.sensor, item.sequence),
            )
        else:
            events = self._eligible(horizon)
        return [estimator.step(event) for event in events]

    def _eligible(self, horizon: float) -> list[Observation]:
        return sorted(
            (event for event in self.received if event.time <= horizon + 1e-9),
            key=lambda item: (item.time, item.sensor, item.sequence),
        )

    def _rebuild_from(self, late_time: float) -> None:
        base = next(
            (
                checkpoint
                for checkpoint in reversed(self.checkpoints)
                if checkpoint.state.time < late_time - 1e-9
            ),
            None,
        )
        if base is None:
            self.estimator = LocalizationEstimator()
            event_start = 0
            self.outputs = []
            self.checkpoints = []
        else:
            self.estimator.restore(base.state)
            event_start = base.event_count
            self.outputs = self.outputs[:event_start]
            self.checkpoints = [base]
        events = self._eligible(self.horizon or 0.0)
        for observation in events[event_start:]:
            self.estimator.step(observation)
            self.outputs.append(self.estimator.snapshot())
        self.checkpoints.append(Checkpoint(self.estimator.snapshot(), len(events)))

    def _last_event_count(self) -> int:
        return self.checkpoints[-1].event_count if self.checkpoints else 0
