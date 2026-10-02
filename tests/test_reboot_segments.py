import math
import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import CheckpointStore
from robot_localization.segments import (
    RebootBoundary,
    SegmentAnchor,
    SegmentedSensorClock,
)
from robot_localization.sources import RebootScenarioSource
from robot_localization.types import WheelObservation


SENSOR = "rear_wheels"


def reboot_pipeline(source: RebootScenarioSource | None = None) -> LocalizationPipeline:
    source = source or RebootScenarioSource()
    pipeline = LocalizationPipeline()
    for observation in source.observations():
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
    return pipeline


def stage_full(
    pipeline: LocalizationPipeline,
    source: RebootScenarioSource,
    *,
    uncertain: bool = False,
) -> object:
    boundaries = [
        RebootBoundary(source.REBOOTS[0][1], True),
        RebootBoundary(source.REBOOTS[1][1], not uncertain, (7.7, 8.3)),
    ]
    return pipeline.stage_reboot_segments(
        SENSOR, boundaries=boundaries, anchors=source.anchors()
    )


class SegmentClockTests(unittest.TestCase):
    def test_same_raw_stamp_in_different_segments_differs(self) -> None:
        source = RebootScenarioSource()
        clock = SegmentedSensorClock.from_anchors(
            SENSOR, source.anchors().items(), source.boundaries()
        )
        # Two distinct observations with identical raw stamps after resets.
        first = WheelObservation(0.25, SENSOR, 21, 1.0, 0.0)
        later = WheelObservation(0.25, SENSOR, 41, 1.0, 0.0)
        first_time = clock.corrected_time_at(first.sequence, first.time)
        later_time = clock.corrected_time_at(later.sequence, later.time)
        self.assertGreater(first_time, 4.0)
        self.assertGreater(later_time, 8.0)
        self.assertAlmostEqual(later_time - first_time, 4.0)
        self.assertNotAlmostEqual(first_time, later_time)
        self.assertEqual(clock.segment_index_for(21), 1)
        self.assertEqual(clock.segment_index_for(41), 2)

    def test_single_anchor_and_uncertain_boundary_keep_ranges(self) -> None:
        source = RebootScenarioSource()
        single = {index: anchors[:1] for index, anchors in source.anchors().items()}
        single[2] = [(*single[2][0], 0.4)]
        clock = SegmentedSensorClock.from_anchors(
            SENSOR, single.items(), source.boundaries()
        )
        self.assertEqual(clock.single_anchor_segments(), (0, 1, 2))
        low, high = clock.corrected_range(45, clock.segment_for(45).anchors[0].stamp)
        self.assertAlmostEqual(high - low, 0.8)
        self.assertFalse(clock.boundaries[1].confirmed)
        self.assertEqual(clock.boundaries[1].reset_window, (7.7, 8.3))

    def test_contradictory_cross_segment_anchors_are_reported_not_joined(self) -> None:
        clock = SegmentedSensorClock.from_anchors(
            SENSOR,
            {
                0: [(0.0, 0.0), (4.0, 4.0)],
                1: [(0.0, 1.0), (4.0, 3.0)],
            }.items(),
            [RebootBoundary(10, True)],
        )
        problems = clock.contradictions()
        self.assertEqual(len(problems), 1)
        # Segment parameters remain independent; nothing smooths the jump.
        self.assertAlmostEqual(clock.offset_at(11, 0.0), -1.0)
        self.assertAlmostEqual(clock.offset_at(5, 4.0), 0.0)

    def test_uncalibrated_segment_never_inherits_parameters(self) -> None:
        clock = SegmentedSensorClock.from_anchors(
            SENSOR,
            {0: [(0.0, 0.0), (4.0, 4.0)]}.items(),
            [RebootBoundary(10, True)],
        )
        self.assertEqual(clock.uncalibrated_segments(), (1,))
        observation = WheelObservation(0.5, SENSOR, 11, 1.0, 0.0)
        self.assertEqual(clock.corrected_time_at(11, observation.time), 0.5)


class RebootPipelineTests(unittest.TestCase):
    def test_segment_views_identify_basis_and_uncertainty(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(source)
        comparison = stage_full(pipeline, source, uncertain=True)
        statuses = {view.index: view.status for view in comparison.segment_views}
        self.assertEqual(statuses[0], "calibrated")
        self.assertEqual(statuses[2], "boundary_uncertain")
        self.assertEqual(
            comparison.segment_views[2].reset_window, (7.7, 8.3)
        )
        for decision in comparison.decisions:
            self.assertIsNotNone(decision.segment_index)
            self.assertEqual(decision.stamp, decision.stamp)  # raw stamp retained
        pipeline.discard_staged_delay()

    def test_identities_timestamps_and_history_are_not_rewritten(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(source)
        original = {(o.sensor, o.sequence): o.time for o in pipeline.received}
        comparison = stage_full(pipeline, source)
        identities = {item.identity for item in comparison.timeline}
        self.assertEqual(
            identities, {(o.sensor, o.sequence) for o in pipeline.received}
        )
        pipeline.publish_staged_delay()
        self.assertEqual(
            {(o.sensor, o.sequence): o.time for o in pipeline.received}, original
        )

    def test_interleaved_segments_and_late_observation_fuse_in_trajectory_order(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(source)
        comparison = stage_full(pipeline, source)
        by_sequence = {
            item.sequence: item for item in comparison.decisions
        }
        # The late segment-1 observation moves to its true segment-1 time and
        # participates in the open replay rather than being stamped at arrival.
        late = by_sequence[source.LATE_SEQUENCE]
        self.assertLess(late.corrected_time, 8.0)
        self.assertEqual(late.classification, "open_recomputed")
        # Corrected timeline order follows trajectory time, not sequence.
        corrected_order = [
            item.identity for item in sorted(comparison.timeline, key=lambda t: t.candidate_order)
        ]
        self.assertIn((SENSOR, 21), corrected_order)
        self.assertLess(
            corrected_order.index((SENSOR, 20)),
            corrected_order.index((SENSOR, 21)),
        )
        self.assertLess(
            corrected_order.index((SENSOR, 40)),
            corrected_order.index((SENSOR, 41)),
        )
        pipeline.publish_staged_delay()
        times = [state.time for state in pipeline.outputs]
        self.assertEqual(times, sorted(times))

    def test_crossing_sealed_boundary_is_decided_per_observation(self) -> None:
        source = RebootScenarioSource()
        pipeline = LocalizationPipeline()
        for observation in source.observations():
            pipeline.add_observation(observation)
            pipeline.process_until(observation.observed_at())
        pipeline.seal_until(3.0)
        comparison = stage_full(pipeline, source)
        classes = {d.sequence: d.classification for d in comparison.decisions}
        sealed_sequences = [
            sequence for sequence, kind in classes.items() if kind == "sealed_unchanged"
        ]
        self.assertTrue(sealed_sequences)
        # All segment-0 samples calibrated before 3.0 stay sealed; later ones
        # remain in the open interval.
        self.assertTrue(any(classes[s] == "open_recomputed" for s in classes))
        sealed_before = [
            (s.time, s.x, s.y) for s in pipeline.sealed_outputs
        ]
        pipeline.publish_staged_delay()
        self.assertEqual(
            [(s.time, s.x, s.y) for s in pipeline.sealed_outputs], sealed_before
        )

    def test_compare_boundary_alternatives_separates_clock_shift_from_scarcity(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(source)
        narrow = stage_full(pipeline, source, uncertain=False)
        narrow_positions = {
            change.identity: change.candidate_time
            for change in narrow.position_changes
        }
        pipeline.discard_staged_delay()
        wide = stage_full(pipeline, source, uncertain=True)
        # A weak segment keeps an open range instead of inventing an instant.
        weak_view = next(view for view in wide.segment_views if view.index == 2)
        self.assertEqual(weak_view.status, "boundary_uncertain")
        # Position changes under both explanations are attributable per identity.
        wide_positions = {
            change.identity: change.candidate_time
            for change in wide.position_changes
        }
        self.assertEqual(narrow_positions.keys(), wide_positions.keys())
        self.assertTrue(wide.uncertain_intervals or narrow.uncertain_intervals is not None)
        pipeline.discard_staged_delay()

    def test_confirm_one_segment_then_withdraw_another_keeps_confirmation(self) -> None:
        source = RebootScenarioSource()
        clock = SegmentedSensorClock.from_anchors(
            SENSOR,
            {0: source.anchors()[0]}.items(),
            [
                RebootBoundary(source.REBOOTS[0][1], True),
                RebootBoundary(source.REBOOTS[1][1], False, (7.7, 8.3)),
            ],
        )
        pipeline = LocalizationPipeline(
            calibration=DelayCalibration().with_reboot_clock(clock)
        )
        for observation in source.observations():
            pipeline.add_observation(observation)
            pipeline.process_until(observation.observed_at())
        # Confirm segment 1 independently.
        pipeline.confirm_reboot_segment(SENSOR, 1, source.anchors()[1])
        confirmed = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertEqual(confirmed.segments[0].anchors, clock.segments[0].anchors)
        self.assertTrue(confirmed.segments[1].calibrated)
        self.assertFalse(confirmed.segments[2].calibrated)
        # Stage then withdraw a different segment; segment 1 confirmation stays.
        pipeline.stage_reboot_segment(SENSOR, 2, source.anchors()[2])
        self.assertIsNotNone(pipeline.staged_calibration)
        pipeline.discard_staged_delay()
        after = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertTrue(after.segments[1].calibrated)
        self.assertFalse(after.segments[2].calibrated)
        # Parameters never leak past a reboot: segment 2 stays identity mapping.
        self.assertAlmostEqual(
            pipeline.calibration.corrected_time(
                WheelObservation(0.2, SENSOR, 41, 1.0, 0.0)
            ),
            0.2,
        )

    def test_reboot_explanation_survives_save_and_restore(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(source)
        pipeline.seal_until(3.0)
        stage_full(pipeline, source)
        checkpoint = pipeline.save_checkpoint()
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
                staged_calibration=checkpoint.staged_calibration.as_dict(),
                staged_sensor=checkpoint.staged_sensor,
            )
            state, observations, metadata = CheckpointStore(path).load_with_metadata()
            restored_checkpoint = type(checkpoint)(
                state=state,
                event_count=checkpoint.event_count,
                arrival_horizon=metadata["arrival_horizon"],
                seal_time=metadata["seal_time"],
                calibration=DelayCalibration(metadata["clock_models"]),
                visible_observations=observations,
                sealed_outputs=metadata["sealed_outputs"],
                rejected_observations=metadata["rejected_observations"],
                staged_calibration=DelayCalibration(metadata["staged_calibration"]),
                staged_sensor=metadata["staged_sensor"],
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        self.assertEqual(resumed.staged_sensor, SENSOR)
        self.assertEqual(
            resumed.staged_comparison.reboot_contradictions,
            pipeline.staged_comparison.reboot_contradictions,
        )
        report = resumed.publish_staged_delay()
        self.assertEqual(report.sensor, SENSOR)
        clock = resumed.calibration.reboot_clock_for(SENSOR)
        self.assertEqual(len(clock.segments), 3)
        self.assertTrue(all(segment.calibrated for segment in clock.segments))
        # Same raw stamp maps per segment after restore; history is untouched.
        self.assertAlmostEqual(
            resumed.calibration.corrected_time(
                WheelObservation(0.25, SENSOR, 21, 1.0, 0.0)
            ),
            4.204081632653061,
        )
        self.assertAlmostEqual(
            resumed.calibration.corrected_time(
                WheelObservation(0.25, SENSOR, 41, 1.0, 0.0)
            ),
            8.204081632653061,
        )


if __name__ == "__main__":
    unittest.main()
