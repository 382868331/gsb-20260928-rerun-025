import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_localization.estimator import LocalizationEstimator
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import (
    CheckpointStore,
    JsonlMessageSource,
    JsonlRecorder,
    serialize_observation,
)
from robot_localization.sources import SyntheticScenarioSource
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
