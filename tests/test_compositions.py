"""Range-composed piecewise schemes built from registered candidates."""

import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.interpretations import AnchorRecord, Interpretation
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import CheckpointStore
from robot_localization.segments import RebootBoundary
from robot_localization.sources import RebootScenarioSource
from robot_localization.types import ImuObservation


SENSOR = "rear_wheels"


def make_pipeline(with_imu=False):
    source = RebootScenarioSource()
    pipeline = LocalizationPipeline()
    for observation in source.observations():
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
    if with_imu:
        for step in range(0, 101):
            stamp = step * 0.1
            pipeline.add_observation(
                ImuObservation(
                    time=stamp,
                    sensor="imu",
                    sequence=step,
                    yaw_rate=0.0,
                    forward_acceleration=0.0,
                    arrival_time=stamp,
                )
            )
        pipeline.process_until(11.0)
    anchors = source.anchors()
    pipeline.stage_reboot_segments(
        SENSOR,
        boundaries=[
            RebootBoundary(20, True),
            RebootBoundary(40, False, (7.7, 8.3)),
        ],
        anchors={0: anchors[0], 1: anchors[1]},
    )
    pipeline.publish_staged_delay()
    return source, pipeline


def register_candidates(pipeline, source, confirmed_second=True):
    anchors = source.anchors()
    second = (
        RebootBoundary(40, True)
        if confirmed_second
        else RebootBoundary(40, False, (7.7, 8.3))
    )
    pipeline.register_interpretation(
        "full-a",
        SENSOR,
        boundaries=[RebootBoundary(20, True), second],
        anchors={0: anchors[0], 1: anchors[1], 2: anchors[2]},
    )
    shifted = {
        0: anchors[0],
        1: anchors[1],
        2: [(stamp, corrected - 0.2) for stamp, corrected in anchors[2]],
    }
    pipeline.register_interpretation(
        "shifted-c",
        SENSOR,
        boundaries=[RebootBoundary(20, True), second],
        anchors=shifted,
    )


class ComposeSchemeTests(unittest.TestCase):
    def test_pieces_keep_source_provenance_and_live_clock_untouched(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        raw_before = [(o.sensor, o.sequence, o.time) for o in pipeline.received]
        scheme = pipeline.compose_scheme(
            "combo-1",
            SENSOR,
            [
                ("full-a", (0, 20)),
                ("full-a", (21, 40)),
                ("shifted-c", (41, 50)),
            ],
        )
        self.assertEqual(scheme.status, "pending")
        self.assertEqual([piece.source for piece in scheme.pieces],
                         ["full-a", "full-a", "shifted-c"])
        self.assertEqual(scheme.pieces[2].sequence_range, (41, 50))
        first, second = scheme.boundaries
        self.assertEqual((first.after_sequence, first.left_source, first.right_source),
                         (20, "full-a", "full-a"))
        self.assertTrue(first.confirmed)
        self.assertEqual((second.after_sequence, second.left_source, second.right_source),
                         (40, "full-a", "shifted-c"))
        self.assertTrue(second.confirmed)
        self.assertEqual(scheme.contradictions, ())
        # Registering a scheme never publishes it.
        live = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertFalse(live.segments[2].calibrated)
        # Raw observations and receive order are untouched.
        self.assertEqual(
            [(o.sensor, o.sequence, o.time) for o in pipeline.received], raw_before)
        with self.assertRaises(ValueError):
            pipeline.compose_scheme("combo-1", SENSOR, [("full-a", (0, 20))])

    def test_uncovered_ranges_stay_unknown(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        scheme = pipeline.compose_scheme(
            "combo-gap", SENSOR, [("full-a", (0, 20)), ("full-a", (31, 40))])
        self.assertEqual(scheme.unknown_ranges(), ((21, 30),))
        self.assertEqual(
            pipeline._composition_unknown_ranges(scheme), ((21, 30), (41, 50)))
        clock = scheme.as_clock()
        self.assertIn(1, clock.uncalibrated_segments())

    def test_overlapping_ranges_reported_with_interval(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        scheme = pipeline.compose_scheme(
            "combo-overlap", SENSOR, [("full-a", (0, 25)), ("full-a", (21, 40))])
        kinds = {item.kind for item in scheme.contradictions}
        self.assertIn("range_overlap", kinds)
        overlap = next(
            item for item in scheme.contradictions if item.kind == "range_overlap")
        self.assertEqual(overlap.sequence_interval, (21, 25))
        self.assertEqual(len(overlap.evidence), 2)

    def test_reboot_breakpoint_inside_piece_reported(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        scheme = pipeline.compose_scheme(
            "combo-spanning", SENSOR, [("full-a", (0, 40))])
        kinds = {item.kind for item in scheme.contradictions}
        self.assertIn("reboot_inside_piece", kinds)
        item = next(
            entry for entry in scheme.contradictions
            if entry.kind == "reboot_inside_piece")
        self.assertEqual(item.sequence_interval, (20, 21))
        self.assertEqual(item.evidence[0]["after_sequence"], 20)

    def test_non_monotonic_stamps_reported_with_evidence(self):
        source, pipeline = make_pipeline()
        anchors = source.anchors()
        pipeline.register_interpretation(
            "mono",
            SENSOR,
            boundaries=[RebootBoundary(40, True)],
            anchors={0: anchors[0], 1: anchors[2]},
        )
        scheme = pipeline.compose_scheme("combo-mono", SENSOR, [("mono", (0, 40))])
        kinds = {item.kind for item in scheme.contradictions}
        self.assertIn("non_monotonic_stamp", kinds)
        item = next(
            entry for entry in scheme.contradictions
            if entry.kind == "non_monotonic_stamp")
        self.assertEqual(item.sequence_interval, (20, 21))
        stamps = [entry["stamp"] for entry in item.evidence]
        self.assertGreater(stamps[0], stamps[1])

    def test_shared_anchor_conflict_across_unconfirmed_boundary(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source, confirmed_second=False)
        scheme = pipeline.compose_scheme(
            "combo-shared", SENSOR, [("full-a", (21, 40)), ("shifted-c", (41, 50))])
        kinds = {item.kind for item in scheme.contradictions}
        self.assertIn("shared_anchor_conflict", kinds)
        item = next(
            entry for entry in scheme.contradictions
            if entry.kind == "shared_anchor_conflict")
        self.assertEqual(item.sequence_interval, (40, 41))
        self.assertEqual(
            {entry["source"] for entry in item.evidence}, {"full-a", "shifted-c"})

    def test_contradicted_scheme_cannot_be_adopted(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        pipeline.compose_scheme("combo-bad", SENSOR, [("full-a", (0, 40))])
        with self.assertRaises(RuntimeError) as caught:
            pipeline.adopt_composition("combo-bad")
        self.assertIn("reboot_inside_piece", str(caught.exception))
        self.assertEqual(pipeline.compositions["combo-bad"].status, "pending")


class OrderingAndAdoptionTests(unittest.TestCase):
    def test_pending_schemes_compared_then_one_adopted(self):
        source, pipeline = make_pipeline(with_imu=True)
        register_candidates(pipeline, source)
        raw_before = [(o.sensor, o.sequence, o.time) for o in pipeline.received]
        pipeline.compose_scheme(
            "combo-a", SENSOR,
            [("full-a", (0, 20)), ("full-a", (21, 40)), ("full-a", (41, 50))])
        pipeline.compose_scheme(
            "combo-b", SENSOR,
            [("full-a", (0, 20)), ("full-a", (21, 40)), ("shifted-c", (41, 50))])
        comparison = pipeline.compare_composition_ordering(["combo-a", "combo-b"])
        self.assertTrue(comparison.changes)
        for change in comparison.changes:
            self.assertEqual(change.first[0], SENSOR)
            self.assertEqual(change.second[0], "imu")
            orders = dict(change.orders)
            self.assertNotEqual(orders["combo-a"], orders["combo-b"])
        # Comparing is read-only: both schemes stay pending.
        self.assertEqual(pipeline.compositions["combo-a"].status, "pending")
        self.assertEqual(pipeline.compositions["combo-b"].status, "pending")
        report = pipeline.adopt_composition("combo-a")
        self.assertEqual(report.sensor, SENSOR)
        self.assertEqual(pipeline.compositions["combo-a"].status, "adopted")
        self.assertEqual(pipeline.compositions["combo-b"].status, "pending")
        live = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertTrue(live.segments[2].calibrated)
        # Adoption creates a new version; receive order and raw stamps stay.
        self.assertEqual(
            [(o.sensor, o.sequence, o.time) for o in pipeline.received], raw_before)
        with self.assertRaises(RuntimeError):
            pipeline.adopt_composition("combo-a")
        pipeline.discard_composition("combo-b")
        self.assertNotIn("combo-b", pipeline.compositions)


class LocalRecheckTests(unittest.TestCase):
    def test_local_anchor_rechecks_only_its_range(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        pipeline.compose_scheme(
            "combo-1", SENSOR, [("full-a", (0, 20)), ("full-a", (21, 40))])
        recheck = pipeline.register_composition_anchor(
            "combo-1", "seg0-extra", 0, (2.06, 2.0), source="field-survey")
        self.assertEqual(recheck.rechecked_ranges, ((0, 20),))
        self.assertEqual(recheck.retained_ranges, ((21, 40),))
        self.assertEqual(recheck.unknown_ranges, ((41, 50),))
        self.assertEqual(recheck.contradictions, ())
        record = pipeline.anchor_records["seg0-extra"]
        self.assertEqual(record.source, "field-survey")
        self.assertEqual(record.applies_to, (0, 20))
        # A contradictory local anchor is reported only for its own range.
        recheck = pipeline.register_composition_anchor(
            "combo-1", "seg1-bad", 1, (3.9, 3.0))
        self.assertEqual(recheck.rechecked_ranges, ((21, 40),))
        self.assertEqual(recheck.retained_ranges, ((0, 20),))
        kinds = {item.kind for item in recheck.contradictions}
        self.assertIn("non_monotonic_corrected", kinds)
        for item in recheck.contradictions:
            self.assertEqual(item.piece_indices, (1,))


class CompositionPersistenceTests(unittest.TestCase):
    def test_save_and_reopen_keeps_provenance_and_decisions(self):
        source, pipeline = make_pipeline()
        register_candidates(pipeline, source)
        pipeline.compose_scheme(
            "combo-1", SENSOR,
            [("full-a", (0, 20)), ("full-a", (21, 40)), ("shifted-c", (41, 50))])
        pipeline.compose_scheme("combo-bad", SENSOR, [("full-a", (0, 40))])
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
                anchor_records=checkpoint.anchor_records,
                pending_interpretations=checkpoint.pending_interpretations,
                compositions=checkpoint.compositions,
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
                anchor_records=[
                    AnchorRecord.from_dict(item)
                    for item in metadata["anchor_records"]
                ],
                pending_interpretations=[
                    Interpretation.from_dict(item)
                    for item in metadata["pending_interpretations"]
                ],
                compositions=metadata["compositions"],
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        self.assertEqual(set(resumed.compositions), {"combo-1", "combo-bad"})
        scheme = resumed.compositions["combo-1"]
        self.assertEqual(scheme.status, "pending")
        self.assertEqual([piece.source for piece in scheme.pieces],
                         ["full-a", "full-a", "shifted-c"])
        self.assertEqual(scheme.boundaries[1].right_source, "shifted-c")
        bad = resumed.compositions["combo-bad"]
        self.assertIn(
            "reboot_inside_piece", {item.kind for item in bad.contradictions})
        with self.assertRaises(RuntimeError):
            resumed.adopt_composition("combo-bad")
        resumed.adopt_composition("combo-1")
        self.assertEqual(resumed.compositions["combo-1"].status, "adopted")


if __name__ == "__main__":
    unittest.main()

