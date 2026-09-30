import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_localization.estimator import LocalizationEstimator
from robot_localization.calibration import DelayCalibration
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import (
    CheckpointStore,
    JsonlMessageSource,
    JsonlRecorder,
    serialize_observation,
)
from robot_localization.sources import SyntheticScenarioSource
from robot_localization.types import ImuObservation, WheelObservation


TRUE_DELAYS = {"rear_wheels": 0.30, "imu": 0.12}


def delayed_source() -> SyntheticScenarioSource:
    return SyntheticScenarioSource(fixed_delays=TRUE_DELAYS)


def run_online(source: SyntheticScenarioSource, calibration: DelayCalibration | None = None):
    pipeline = LocalizationPipeline(calibration=calibration)
    live_times = []
    for message in source.observations():
        pipeline.add_observation(message)
        pipeline.process_until(message.observed_at())
        live_times.append(pipeline.state().time)
    return pipeline, live_times


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


class DelayCalibrationTests(unittest.TestCase):
    def test_stamp_and_arrival_are_retained_separately(self) -> None:
        source = delayed_source()
        delayed_messages = [m for m in source.observations() if m.arrival_time is not None]
        self.assertTrue(delayed_messages)
        for message in delayed_messages:
            self.assertGreaterEqual(message.arrival_time, message.time - 1e-9)

    def test_delay_correction_reduces_known_path_error(self) -> None:
        source = delayed_source()
        truth = {round(point.time, 3): point for point in source.ground_truth()}

        def rmse(calibration: DelayCalibration | None) -> float:
            pipeline, _ = run_online(source, calibration)
            errors = []
            for output in pipeline.all_outputs():
                point = min(truth.values(), key=lambda item: abs(item.time - output.time))
                if abs(point.time - output.time) < 0.03:
                    errors.append(math.hypot(output.x - point.x, output.y - point.y) ** 2)
            return math.sqrt(sum(errors) / len(errors))

        corrected = rmse(DelayCalibration(TRUE_DELAYS))
        uncorrected = rmse(DelayCalibration())
        self.assertLess(corrected, 0.5 * uncorrected)

    def test_changing_delay_recomputes_only_open_interval(self) -> None:
        from robot_localization.pipeline import SealedHistoryError

        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.10, "imu": 0.12})
        )
        messages = source.observations()
        report = None
        for message in messages:
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if report is None and message.observed_at() >= 2.6:
                report = pipeline.set_sensor_delay("rear_wheels", 0.30)
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
        self.assertIsNotNone(report)
        self.assertEqual(report.sensor, "rear_wheels")
        self.assertAlmostEqual(report.new_delay, 0.30)
        self.assertEqual(report.recomputed_from, 0.0)
        self.assertFalse(report.sealed_blocked)
        sealed_values = [state.x for state in pipeline.sealed_outputs]
        with self.assertRaises(SealedHistoryError):
            pipeline.set_sensor_delay("rear_wheels", 0.45)
        # Sealed outputs remain byte-for-byte stable; only the open tail
        # could be recomputed.
        self.assertEqual([state.x for state in pipeline.sealed_outputs], sealed_values)

    def test_delay_change_touching_sealed_output_is_rejected(self) -> None:
        from robot_localization.pipeline import SealedHistoryError

        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.10, "imu": 0.12})
        )
        sealed_value = None
        for message in source.observations():
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
                sealed_value = pipeline.sealed_outputs[-1].x
        with self.assertRaises(SealedHistoryError) as context:
            pipeline.set_sensor_delay("rear_wheels", 0.45)
        self.assertTrue(context.exception.sealed_times)
        # Sealed outputs are untouched after the rejected update.
        self.assertEqual(pipeline.sealed_outputs[-1].x, sealed_value)

    def test_backfill_burst_never_moves_live_state_backwards(self) -> None:
        source = delayed_source()
        pipeline, live_times = run_online(
            source, DelayCalibration(TRUE_DELAYS)
        )
        self.assertEqual(live_times, sorted(live_times))
        # The wheel outage degrades the live estimate while the gap is open.
        gap_states = [
            state
            for state in pipeline.all_outputs()
            if 7.1 <= state.time <= 7.6
        ]
        self.assertTrue(
            any("wheel_stream_gap" in state.degraded_reasons for state in gap_states)
        )
        # Uncertainty expanded during the outage instead of staying constant.
        radii = [state.position_95_radius for state in gap_states]
        self.assertGreater(max(radii), radii[0])
        # After the burst arrives, history was revised but the live clock did
        # not jump to an old timestamp.
        self.assertGreaterEqual(pipeline.state().time, max(live_times) - 1e-9)

    def test_restore_keeps_the_same_visible_data_boundary(self) -> None:
        source = delayed_source()
        messages = source.observations()
        pipeline = LocalizationPipeline(calibration=DelayCalibration(TRUE_DELAYS))
        cutoff = len(messages) - 25
        boundary_at_checkpoint = None
        for index, message in enumerate(messages):
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if index == cutoff:
                checkpoint = pipeline.save_checkpoint()
                prior = pipeline.checkpoint_observations()
                boundary_at_checkpoint = pipeline.visible_data_boundary()
        restored = LocalizationPipeline()
        restored.restore_checkpoint(checkpoint, prior)
        after = restored.visible_data_boundary()
        self.assertEqual(boundary_at_checkpoint, after)
        # Messages that arrived after the checkpoint stay hidden.
        self.assertEqual(len(restored.received), len(prior))


if __name__ == "__main__":
    unittest.main()
