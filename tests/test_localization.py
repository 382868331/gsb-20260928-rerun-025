import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_localization.estimator import LocalizationEstimator
from robot_localization.evaluation import compare_with_truth
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import (
    CheckpointStore,
    JsonlMessageSource,
    JsonlRecorder,
    serialize_observation,
)
from robot_localization.sources import SyntheticScenarioSource, TimedDelayedScenarioSource
from robot_localization.types import ImuObservation, WheelObservation


class EstimatorTests(unittest.TestCase):
    def test_straight_motion_and_uncertainty(self) -> None:
        estimator = LocalizationEstimator()
        for step in range(1, 21):
            time = step * 0.05
            estimator.step(
                ImuObservation(time, "imu", step, yaw_rate=0.0, forward_acceleration=0.0)
            )
            estimator.step(
                WheelObservation(
                    time,
                    "wheels",
                    step,
                    linear_velocity=1.0,
                    yaw_rate=0.0,
                )
            )
        state = estimator.snapshot()
        self.assertAlmostEqual(state.x, 1.0, places=2)
        self.assertAlmostEqual(state.y, 0.0, places=3)
        self.assertAlmostEqual(state.heading, 0.0, places=3)
        self.assertGreater(state.position_95_radius, 0.0)

    def test_wheel_slip_is_robustly_downweighted(self) -> None:
        estimator = LocalizationEstimator()
        for step in range(1, 41):
            time = step * 0.05
            estimator.step(
                ImuObservation(time, "imu", step, yaw_rate=0.0, forward_acceleration=0.0)
            )
            estimator.step(
                WheelObservation(time, "wheels", step, linear_velocity=1.0, yaw_rate=0.0)
            )
        trusted = estimator.snapshot()
        for step in range(41, 61):
            time = step * 0.05
            estimator.step(
                ImuObservation(time, "imu", step, yaw_rate=0.0, forward_acceleration=0.0)
            )
            estimator.step(
                WheelObservation(time, "wheels", step, linear_velocity=3.0, yaw_rate=0.0)
            )
        slipped = estimator.snapshot()
        self.assertIn("wheel_slip_suspected", slipped.degraded_reasons)
        self.assertLess(abs(slipped.linear_velocity - trusted.linear_velocity), 0.25)

    def test_stream_gap_degrades_quality(self) -> None:
        estimator = LocalizationEstimator()
        estimator.step(ImuObservation(0.05, "imu", 1, 0.0, 0.0))
        estimator.step(WheelObservation(0.05, "wheels", 1, 1.0, 0.0))
        estimator.step(ImuObservation(0.1, "imu", 2, 0.0, 0.0))
        estimator.predict(1.0)
        state = estimator.snapshot()
        uncertainty_before_gap = np.diag(estimator.covariance[:2, :2]).sum()
        estimator.predict(2.0)
        uncertainty_after_gap = np.diag(estimator.covariance[:2, :2]).sum()
        self.assertIn("wheel_stream_gap", state.degraded_reasons)
        self.assertIn("imu_stream_gap", state.degraded_reasons)
        self.assertGreater(uncertainty_after_gap, uncertainty_before_gap)


class PipelineTests(unittest.TestCase):
    def test_future_observation_is_not_used_early(self) -> None:
        pipeline = LocalizationPipeline()
        pipeline.add_observation(ImuObservation(2.0, "imu", 2, 0.0, 0.0))
        pipeline.add_observation(WheelObservation(0.1, "wheels", 1, 1.0, 0.0))
        pipeline.process_until(0.1)
        self.assertAlmostEqual(pipeline.state().time, 0.1)
        self.assertAlmostEqual(pipeline.state().linear_velocity, 1.0, places=1)

    def test_late_observation_matches_full_ordered_replay(self) -> None:
        source = SyntheticScenarioSource()
        messages = source.observations()
        online = LocalizationPipeline()
        horizon = 0.0
        late_index = next(
            index
            for index, item in enumerate(messages)
            if isinstance(item, WheelObservation) and 4.8 <= item.time <= 5.2
        )
        for index, message in enumerate(messages):
            online.add_observation(message)
            if message.time < 10.0:
                horizon = max(horizon, message.time)
                online.process_until(horizon)
            if index == late_index:
                before_late = online.state().x
        online.process_until(10.0)
        replay = LocalizationPipeline()
        for message in messages:
            replay.add_observation(message)
        replayed = replay.replay()
        self.assertEqual(len(online.outputs), len(replayed))
        for actual, expected in zip(online.outputs, replayed, strict=True):
            self.assertTrue(
                np.allclose(
                    [actual.x, actual.y, actual.heading],
                    [expected.x, expected.y, expected.heading],
                    atol=1e-10,
                )
            )
        self.assertNotAlmostEqual(before_late, online.state().x)

    def test_checkpoint_round_trip_and_replay_metrics(self) -> None:
        messages = SyntheticScenarioSource().observations()
        first = LocalizationPipeline()
        horizon = 0.0
        checkpoint = None
        checkpoint_observations = []
        for index, message in enumerate(messages):
            first.add_observation(message)
            horizon = max(horizon, message.time)
            first.process_until(horizon)
            if index == 60:
                checkpoint = first.save_checkpoint()
                checkpoint_observations = first.checkpoint_observations()
        self.assertIsNotNone(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            CheckpointStore(path).save(
                checkpoint.state, checkpoint_observations
            )
            loaded, prior_observations = CheckpointStore(path).load()
        resumed_pipeline = LocalizationPipeline()
        resumed_pipeline.restore_checkpoint(
            type(checkpoint)(state=loaded, event_count=checkpoint.event_count),
            prior_observations,
        )
        resumed_horizon = horizon
        for message in messages[61:]:
            resumed_pipeline.add_observation(message)
            resumed_horizon = max(resumed_horizon, message.time)
            resumed_pipeline.process_until(resumed_horizon)
        full_replay = LocalizationPipeline()
        for message in messages:
            full_replay.add_observation(message)
        expected = full_replay.replay()
        final_expected = expected[-1]
        self.assertAlmostEqual(first.state().x, final_expected.x, places=10)
        self.assertAlmostEqual(first.state().y, final_expected.y, places=10)
        self.assertAlmostEqual(first.state().heading, final_expected.heading, places=10)
        self.assertAlmostEqual(resumed_pipeline.state().x, final_expected.x, places=10)
        self.assertAlmostEqual(resumed_pipeline.state().y, final_expected.y, places=10)
        self.assertAlmostEqual(
            resumed_pipeline.state().heading, final_expected.heading, places=10
        )
        self.assertAlmostEqual(loaded.x, checkpoint.state.x, places=12)

    def test_jsonl_round_trip(self) -> None:
        observations = [
            WheelObservation(0.1, "wheels", 1, 1.0, 0.1),
            ImuObservation(0.15, "imu", 1, 0.0, 0.2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.jsonl"
            JsonlRecorder(path).write_many(observations)
            loaded = list(JsonlMessageSource(path))
        self.assertEqual([serialize_observation(item) for item in loaded], [serialize_observation(item) for item in observations])


class TimedCalibrationTests(unittest.TestCase):
    def _run_timed_scenario(self, delay: float):
        source = TimedDelayedScenarioSource()
        pipeline = LocalizationPipeline(sensor_delays={"rear_wheels": delay})
        for observation in source.observations():
            pipeline.add_observation(observation)
            arrival = observation.arrival_time if observation.arrival_time is not None else observation.time
            pipeline.process_until(arrival)
        return source, pipeline

    def test_delay_correction_keeps_separate_times_and_improves_uncertainty(self) -> None:
        uncorrected_source, uncorrected = self._run_timed_scenario(0.0)
        corrected_source, corrected = self._run_timed_scenario(0.5)
        first_wheel = next(
            item for item in corrected_source.observations() if item.sensor == "rear_wheels"
        )
        self.assertAlmostEqual(first_wheel.time, 0.6)
        self.assertAlmostEqual(first_wheel.arrival_time, 0.6)
        self.assertAlmostEqual(corrected.corrected_time(first_wheel), 0.1)
        uncorrected_metrics = compare_with_truth(
            uncorrected.outputs, uncorrected_source.ground_truth()
        )
        corrected_metrics = compare_with_truth(
            corrected.outputs, corrected_source.ground_truth()
        )
        self.assertLess(
            corrected_metrics.rmse_position,
            uncorrected_metrics.rmse_position * 0.25,
        )
        self.assertGreater(corrected_metrics.uncertainty_coverage, 0.95)
        self.assertLess(uncorrected_metrics.uncertainty_coverage, 0.2)

    def test_outage_degrades_and_burst_does_not_move_state_backwards(self) -> None:
        source = TimedDelayedScenarioSource()
        pipeline = LocalizationPipeline(sensor_delays={"rear_wheels": 0.5})
        before_burst = None
        at_burst = None
        uncertainty_before = None
        for observation in source.observations():
            arrival = observation.arrival_time if observation.arrival_time is not None else observation.time
            pipeline.add_observation(observation)
            if before_burst is None and arrival >= 4.2:
                pipeline.process_until(4.1)
                before_burst = pipeline.state()
                uncertainty_before = before_burst.position_95_radius
            if arrival <= 10.0:
                pipeline.process_until(max(arrival, 4.2 if before_burst is not None else 0.0))
            if before_burst is not None and at_burst is None and arrival >= 4.2:
                at_burst = pipeline.state()
        self.assertIn("wheel_stream_gap", before_burst.degraded_reasons)
        self.assertGreater(uncertainty_before, 0.0)
        self.assertAlmostEqual(at_burst.time, 4.2)
        self.assertAlmostEqual(pipeline.state().time, 10.0)

    def test_delay_change_recomputes_active_interval_and_preserves_sealed_outputs(self) -> None:
        source = TimedDelayedScenarioSource()
        pipeline = LocalizationPipeline(sensor_delays={"rear_wheels": 0.5})
        for observation in source.observations():
            arrival = observation.arrival_time if observation.arrival_time is not None else observation.time
            pipeline.add_observation(observation)
            pipeline.process_until(arrival)
            if arrival >= 2.0:
                break
        sealed = pipeline.seal_outputs_through(1.0)
        sealed_before = [(state.time, state.x, state.y) for state in sealed]
        for observation in source.observations():
            arrival = observation.arrival_time if observation.arrival_time is not None else observation.time
            if 2.0 < arrival <= 3.0:
                pipeline.add_observation(observation)
                pipeline.process_until(arrival)
        record = pipeline.set_sensor_delay("rear_wheels", 0.6)
        self.assertAlmostEqual(record.sealed_start, 0.1)
        self.assertAlmostEqual(record.sealed_end, 1.0)
        self.assertAlmostEqual(record.recomputed_start, 1.0)
        sealed_after = [
            (state.time, state.x, state.y)
            for state in pipeline.sealed_outputs
        ]
        self.assertEqual(sealed_after[: len(sealed_before)], sealed_before)

    def test_sealed_late_arrival_is_retained_but_not_silently_replaced(self) -> None:
        pipeline = LocalizationPipeline()
        for step in range(1, 5):
            time = step * 0.25
            pipeline.add_observation(
                ImuObservation(time, "imu", step, 0.0, 0.0, arrival_time=time)
            )
            pipeline.process_until(time)
        sealed = pipeline.seal_outputs_through(0.5)
        sealed_half = next(state for state in sealed if abs(state.time - 0.5) < 1e-9)
        pipeline.process_until(2.0)
        late_wheel = WheelObservation(
            0.5,
            "rear_wheels",
            99,
            2.0,
            0.0,
            arrival_time=2.1,
        )
        pipeline.add_observation(late_wheel)
        self.assertIn(late_wheel, pipeline.received)
        self.assertEqual(pipeline.revision_records[-1].reason, "sealed_late_arrival")
        self.assertAlmostEqual(pipeline.revision_records[-1].sealed_end, 0.5)
        self.assertIs(pipeline.sealed_outputs[1], sealed_half)

    def test_checkpoint_preserves_visible_arrival_boundary(self) -> None:
        source = TimedDelayedScenarioSource()
        first = LocalizationPipeline(sensor_delays={"rear_wheels": 0.5})
        for observation in source.observations():
            arrival = observation.arrival_time if observation.arrival_time is not None else observation.time
            first.add_observation(observation)
            first.process_until(arrival)
            if arrival >= 4.1:
                break
        checkpoint = first.save_checkpoint()
        visible = first.checkpoint_observations()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            metadata = {
                "horizon": 4.1,
                "visible_arrival": 4.1,
                "sealed_horizon": 0.0,
                "sensor_delays": {"rear_wheels": 0.5},
            }
            CheckpointStore(path).save(checkpoint.state, visible, metadata)
            loaded_state, loaded_observations, loaded_metadata = CheckpointStore(
                path
            ).load_with_metadata()
        resumed = LocalizationPipeline(sensor_delays={"rear_wheels": 0.5})
        resumed.restore_checkpoint(
            type(checkpoint)(
                state=loaded_state,
                event_count=checkpoint.event_count,
                horizon=4.1,
                visible_arrival=4.1,
                sealed_horizon=0.0,
                sensor_delays={"rear_wheels": 0.5},
            ),
            loaded_observations,
        )
        resumed.add_observation(
            WheelObservation(
                9.0,
                "rear_wheels",
                999,
                1.0,
                0.0,
                arrival_time=5.0,
            )
        )
        self.assertEqual(loaded_metadata["visible_arrival"], 4.1)
        self.assertEqual(len(resumed._eligible(4.1)), len(visible))
        self.assertEqual(len(resumed.checkpoint_observations()), len(visible))


class ScenarioTests(unittest.TestCase):
    def test_known_turn_slip_path_accuracy(self) -> None:
        source = SyntheticScenarioSource()
        pipeline = LocalizationPipeline()
        for observation in source.observations():
            pipeline.add_observation(observation)
        estimates = pipeline.replay()
        final = estimates[-1]
        slip_events = [
            estimate
            for estimate in estimates
            if 6.0 <= estimate.time <= 8.0
            and "wheel_slip_suspected" in estimate.degraded_reasons
        ]
        truth = source.ground_truth()[-1]
        position_error = math.hypot(final.x - truth.x, final.y - truth.y)
        heading_error = abs((final.heading - truth.heading + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(position_error, 0.25)
        self.assertLess(heading_error, 0.08)
        self.assertTrue(slip_events)


if __name__ == "__main__":
    unittest.main()
