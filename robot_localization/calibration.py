from __future__ import annotations

from dataclasses import dataclass

from .types import Observation


@dataclass(frozen=True, slots=True)
class SensorDelayModel:
    """Fixed per-sensor transport delay estimate.

    ``delay`` is the transport age: a measurement stamped on arrival was
    produced ``delay`` seconds earlier, so the corrected observation time is
    ``stamp - delay``.  Stamp and arrival time are still retained separately.
    """

    sensor: str
    delay: float = 0.0

    def __post_init__(self) -> None:
        if self.delay < 0.0:
            raise ValueError("fixed delay estimate must be non-negative")


class DelayCalibration:
    """Maps each sensor to its current fixed delay estimate."""

    def __init__(self, models: dict[str, float] | None = None) -> None:
        self._models: dict[str, SensorDelayModel] = {}
        for sensor, delay in (models or {}).items():
            model = SensorDelayModel(sensor, float(delay))
            self._models[sensor] = model

    def delay_for(self, sensor: str) -> float:
        model = self._models.get(sensor)
        return 0.0 if model is None else model.delay

    def corrected_time(self, observation: Observation) -> float:
        return observation.time - self.delay_for(observation.sensor)

    def with_delay(self, sensor: str, delay: float) -> "DelayCalibration":
        clone = DelayCalibration()
        clone._models = dict(self._models)
        clone._models[sensor] = SensorDelayModel(sensor, float(delay))
        return clone

    def affected_interval(
        self, sensor: str, observations: list[Observation]
    ) -> tuple[float, float] | None:
        times = [item.time for item in observations if item.sensor == sensor]
        if not times:
            return None
        return min(times), max(times)

    def as_dict(self) -> dict[str, float]:
        return {sensor: model.delay for sensor, model in self._models.items()}
