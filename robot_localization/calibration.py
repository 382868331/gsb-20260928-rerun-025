from __future__ import annotations

from dataclasses import dataclass

from .types import Observation


@dataclass(frozen=True, slots=True)
class SensorDelayModel:
    """Per-sensor clock model: a fixed transport offset plus linear drift.

    A sensor clock can run slightly slow or fast rather than differing from
    the trajectory clock by one constant offset.  The corrected observation
    time is::

        corrected = epoch + (stamp - epoch) / (1.0 + drift) - delay

    ``delay`` is the transport age at ``epoch`` and ``drift`` is the rate at
    which the stamped clock pulls ahead of the trajectory clock per stamped
    second (``0.0`` reproduces the fixed-delay behaviour).  Stamp and arrival
    time are still retained separately.
    """

    sensor: str
    delay: float = 0.0
    drift: float = 0.0
    epoch: float = 0.0

    def __post_init__(self) -> None:
        if self.delay < 0.0:
            raise ValueError("fixed delay estimate must be non-negative")
        if not (-1.0 < self.drift < 1.0):
            raise ValueError("clock drift rate must stay within (-1, 1)")

    def corrected(self, stamp: float) -> float:
        return self.epoch + (stamp - self.epoch) / (1.0 + self.drift) - self.delay


class DelayCalibration:
    """Maps each sensor to its current clock model (offset plus drift)."""

    def __init__(
        self,
        models: dict[str, float | SensorDelayModel | tuple[float, float, float]] | None = None,
    ) -> None:
        self._models: dict[str, SensorDelayModel] = {}
        for sensor, value in (models or {}).items():
            model = self._coerce(sensor, value)
            self._models[sensor] = model

    @staticmethod
    def _coerce(sensor: str, value) -> SensorDelayModel:
        if isinstance(value, SensorDelayModel):
            if value.sensor != sensor:
                value = SensorDelayModel(sensor, value.delay, value.drift, value.epoch)
            return value
        if isinstance(value, (tuple, list)):
            delay, drift, epoch = (list(value) + [0.0, 0.0])[:3]
            return SensorDelayModel(sensor, float(delay), float(drift), float(epoch))
        return SensorDelayModel(sensor, float(value))

    def delay_for(self, sensor: str) -> float:
        model = self._models.get(sensor)
        return 0.0 if model is None else model.delay

    def drift_for(self, sensor: str) -> float:
        model = self._models.get(sensor)
        return 0.0 if model is None else model.drift

    def model_for(self, sensor: str) -> SensorDelayModel:
        return self._models.get(sensor, SensorDelayModel(sensor))

    def corrected_time(self, observation: Observation) -> float:
        model = self._models.get(observation.sensor)
        if model is None:
            return observation.time
        return model.corrected(observation.time)

    def with_delay(self, sensor: str, delay: float) -> "DelayCalibration":
        """Return a copy with a fixed delay, preserving any stored drift."""
        clone = DelayCalibration()
        clone._models = dict(self._models)
        previous = self._models.get(sensor)
        clone._models[sensor] = SensorDelayModel(
            sensor,
            float(delay),
            previous.drift if previous is not None else 0.0,
            previous.epoch if previous is not None else 0.0,
        )
        return clone

    def with_clock(
        self,
        sensor: str,
        delay: float = 0.0,
        drift: float = 0.0,
        epoch: float = 0.0,
    ) -> "DelayCalibration":
        """Return a copy with a full offset-plus-drift clock model."""
        clone = DelayCalibration()
        clone._models = dict(self._models)
        clone._models[sensor] = SensorDelayModel(
            sensor, float(delay), float(drift), float(epoch)
        )
        return clone

    def affected_interval(
        self, sensor: str, observations: list[Observation]
    ) -> tuple[float, float] | None:
        times = [item.time for item in observations if item.sensor == sensor]
        if not times:
            return None
        return min(times), max(times)

    def as_dict(self) -> dict[str, float]:
        """Delay-only view, used by legacy metrics and serialisation paths."""
        return {sensor: model.delay for sensor, model in self._models.items()}

    def clock_models(self) -> dict[str, SensorDelayModel]:
        return dict(self._models)
