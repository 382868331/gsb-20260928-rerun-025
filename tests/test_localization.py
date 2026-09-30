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
        self.assertAlmostEqual(report.recomputed_from, 0.1)
        self.assertFalse(report.sealed_blocked)
        self.assertIsNone(report.boundary_state_time)

    def test_delay_change_keeps_sealed_history_and_recomputes_open_tail(self) -> None:
        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.10, "imu": 0.12})
        )
        sealed_before = {}
        open_before = {}
        for message in source.observations():
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
                sealed_before = {
                    round(state.time, 6): (state.x, state.y)
                    for state in pipeline.sealed_outputs
                }
                open_before = {
                    round(state.time, 6): (state.x, state.y)
                    for state in pipeline.outputs
                }
        comparison = pipeline.stage_sensor_delay("rear_wheels", 0.45)
        self.assertEqual(comparison.sealed_history, (0.0, 5.0))
        self.assertEqual(comparison.boundary_state_time, 5.0)
        self.assertTrue(comparison.candidate.open_sequences)
        self.assertTrue(comparison.candidate.sealed_sequences)
        self.assertTrue(
            any(
                item.classification == "crossing_rejected"
                for item in comparison.decisions
            )
        )
        self.assertTrue(
            any(item.classification == "open_recomputed" for item in comparison.decisions)
        )
        report = pipeline.publish_staged_delay()
        self.assertEqual(report.boundary_state_time, 5.0)
        self.assertTrue(report.rejected_observations)
        self.assertEqual(
            {
                round(state.time, 6): (state.x, state.y)
                for state in pipeline.sealed_outputs
            },
            sealed_before,
        )
        open_after = {
            round(state.time, 6): (state.x, state.y)
            for state in pipeline.outputs
        }
        self.assertNotEqual(open_after, open_before)

    def test_unpublished_correction_can_be_discarded(self) -> None:
        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.30, "imu": 0.12})
        )
        for message in source.observations():
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
        before = [(state.time, state.x, state.y) for state in pipeline.all_outputs()]
        pipeline.stage_sensor_delay("rear_wheels", 0.45)
        pipeline.discard_staged_delay()
        self.assertEqual(pipeline.calibration.delay_for("rear_wheels"), 0.30)
        self.assertEqual(
            [(state.time, state.x, state.y) for state in pipeline.all_outputs()],
            before,
        )
        self.assertFalse(pipeline.rejected_observations)

    def test_reseal_save_restore_preserve_timeline(self) -> None:
        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.30, "imu": 0.12})
        )
        for message in source.observations():
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
        pipeline.seal_until(6.0)
        checkpoint = pipeline.save_checkpoint()
        expected = [
            (state.time, state.x, state.y, state.heading)
            for state in pipeline.all_outputs()
        ]
        restored = LocalizationPipeline()
        restored.restore_checkpoint(checkpoint)
        actual = [
            (state.time, state.x, state.y, state.heading)
            for state in restored.all_outputs()
        ]
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            atol=1e-12,
        )
        self.assertEqual(restored.seal_time, 6.0)
        self.assertEqual(
            restored.visible_data_boundary(),
            pipeline.visible_data_boundary(),
        )

    def test_file_checkpoint_restore_preserves_sealed_timeline(self) -> None:
        source = delayed_source()
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration({"rear_wheels": 0.30, "imu": 0.12})
        )
        cutoff = None
        for index, message in enumerate(source.observations()):
            pipeline.add_observation(message)
            pipeline.process_until(message.observed_at())
            if pipeline.seal_time is None and message.observed_at() >= 5.3:
                pipeline.seal_until(5.0)
            if index == len(source.observations()) - 25:
                cutoff = index
                checkpoint = pipeline.save_checkpoint()
        self.assertIsNotNone(cutoff)
        expected = [
            (state.time, state.x, state.y, state.heading)
            for state in pipeline.all_outputs()
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            CheckpointStore(path).save(
                checkpoint.state,
                checkpoint.visible_observations,
                calibration=checkpoint.calibration.as_dict(),
                arrival_horizon=checkpoint.arrival_horizon,
                seal_time=checkpoint.seal_time,
                sealed_outputs=checkpoint.sealed_outputs,
                rejected_observations=checkpoint.rejected_observations,
            )
            state, observations, metadata = CheckpointStore(path).load_with_metadata()
            loaded_checkpoint = type(checkpoint)(
                state=state,
                event_count=len(metadata["sealed_outputs"]),
                arrival_horizon=metadata["arrival_horizon"],
                seal_time=metadata["seal_time"],
                calibration=DelayCalibration(metadata["fixed_delays"]),
                visible_observations=observations,
                sealed_outputs=metadata["sealed_outputs"],
                rejected_observations=metadata["rejected_observations"],
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(loaded_checkpoint)
        for message in source.observations()[cutoff + 1 :]:
            resumed.add_observation(message)
            resumed.process_until(message.observed_at())
        actual = [
            (state.time, state.x, state.y, state.heading)
            for state in resumed.all_outputs()
        ]
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            atol=1e-10,
        )

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
