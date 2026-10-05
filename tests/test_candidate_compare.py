"""Multi-candidate filtering, side-by-side comparison and persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.candidate_compare import (
    CandidateFilter,
    ComparisonSelection,
)
from robot_localization.interpretations import AnchorRecord, Interpretation
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import CheckpointStore
from robot_localization.segments import RebootBoundary
from robot_localization.sources import RebootScenarioSource


SENSOR = "rear_wheels"
BOUNDARIES = [
    RebootBoundary(20, True),
    RebootBoundary(40, False, (7.7, 8.3)),
]


def make_pipeline(segment_anchors):
    source = RebootScenarioSource()
    pipeline = LocalizationPipeline()
    for observation in source.observations():
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
    pipeline.stage_reboot_segments(
        SENSOR,
        boundaries=BOUNDARIES,
        anchors=segment_anchors,
    )
    pipeline.publish_staged_delay()
    return source, pipeline


def full_anchors(source):
    anchors = source.anchors()
    return {0: anchors[0], 1: anchors[1], 2: anchors[2]}


def shifted_anchors(source, delta):
    anchors = full_anchors(source)
    anchors[2] = [(stamp, corrected + delta) for stamp, corrected in anchors[2]]
    return anchors


def registered_pipeline():
    source = RebootScenarioSource()
    known = source.anchors()
    _, pipeline = make_pipeline({0: known[0], 1: known[1][:1], 2: known[2]})
    pipeline.register_interpretation(
        "full-a", SENSOR, boundaries=BOUNDARIES, anchors=full_anchors(source)
    )
    pipeline.register_interpretation(
        "full-b",
        SENSOR,
        boundaries=BOUNDARIES,
        anchors=shifted_anchors(source, -0.2),
    )
    pipeline.register_late_anchor(
        "seg1-end",
        SENSOR,
        1,
        known[1][1],
        applies_to=(25, 35),
        arrival_time=9.5,
    )
    return source, pipeline


class CandidateFilterTests(unittest.TestCase):
    def test_filter_by_status_anchor_source_and_observation_range(self):
        _, pipeline = registered_pipeline()
        pending = pipeline.filtered_interpretations(
            CandidateFilter(statuses=("pending",))
        )
        self.assertEqual(
            {entry.name for entry in pending},
            {"full-a", "full-b", "late-anchor:seg1-end"},
        )
        by_anchor = pipeline.filtered_interpretations(
            CandidateFilter(anchor_ids=("seg1-end",))
        )
        self.assertEqual(
            [entry.name for entry in by_anchor], ["late-anchor:seg1-end"]
        )
        # The partial adoption only covers sequences 25..35.
        late_only = pipeline.filtered_interpretations(
            CandidateFilter(sequence_range=(25, 35))
        )
        self.assertIn("late-anchor:seg1-end", {entry.name for entry in late_only})
        ranged = pipeline.filtered_interpretations(
            CandidateFilter(sequence_range=(41, 50))
        )
        self.assertEqual(
            {entry.name for entry in ranged}, {"full-a", "full-b"}
        )
        # Status filter reflects explicit adoption.
        pipeline.adopt_interpretation("full-a")
        still_pending = pipeline.filtered_interpretations(
            CandidateFilter(statuses=("pending",))
        )
        self.assertEqual(
            {entry.name for entry in still_pending},
            {"full-b", "late-anchor:seg1-end"},
        )
        adopted = pipeline.filtered_interpretations(
            CandidateFilter(statuses=("adopted",))
        )
        self.assertEqual([entry.name for entry in adopted], ["full-a"])

    def test_stored_filter_is_used_by_default(self):
        _, pipeline = registered_pipeline()
        pipeline.set_candidate_filter(CandidateFilter(anchor_ids=("seg1-end",)))
        entries = pipeline.filtered_interpretations()
        self.assertEqual([entry.name for entry in entries], ["late-anchor:seg1-end"])


class CandidateComparisonTests(unittest.TestCase):
    def test_three_way_split_of_overlapping_observations(self):
        _, pipeline = registered_pipeline()
        comparison = pipeline.compare_candidates(["full-a", "full-b"])
        # Segment 2 anchors shifted: those observations are time conflicts.
        conflict_sequences = {item.sequence for item in comparison.time_conflicts}
        self.assertTrue(conflict_sequences)
        self.assertTrue(all(sequence > 40 for sequence in conflict_sequences))
        for item in comparison.time_conflicts:
            times = dict(item.corrected_times)
            self.assertAlmostEqual(times["full-a"] - times["full-b"], 0.2)
            self.assertAlmostEqual(item.time_spread, 0.2)
        # Segments 0 and 1 share anchors: identical time and basis.
        identical_sequences = {item.sequence for item in comparison.identical}
        self.assertTrue(identical_sequences)
        self.assertTrue(all(sequence <= 40 for sequence in identical_sequences))
        self.assertFalse(comparison.same_time_different_basis)
        counts = comparison.counts()
        self.assertEqual(
            counts["time_conflicts"] + counts["identical"],
            len(pipeline.interpretation_comparison("full-a").decisions),
        )

    def test_same_time_different_basis_group(self):
        _, pipeline = registered_pipeline()
        comparison = pipeline.compare_candidates(["full-a", "late-anchor:seg1-end"])
        grouped = {
            item.sequence: item
            for item in comparison.same_time_different_basis
        }
        # Outside the adopted range the partial candidate falls back to the
        # published base: segment 0 keeps the same time on different basis.
        self.assertTrue(any(sequence <= 20 for sequence in grouped))
        for item in grouped.values():
            times = dict(item.corrected_times)
            self.assertAlmostEqual(times["full-a"], times["late-anchor:seg1-end"])
            basis = dict(item.basis)
            self.assertNotEqual(basis["full-a"], basis["late-anchor:seg1-end"])
        self.assertTrue(comparison.time_conflicts)

    def test_conflicts_only_export_keeps_counts(self):
        _, pipeline = registered_pipeline()
        comparison = pipeline.compare_candidates(
            ["full-a", "full-b"], conflicts_only=True
        )
        exported = comparison.as_dict()
        self.assertEqual(set(exported["groups"]), {"time_conflicts"})
        self.assertGreater(exported["counts"]["identical"], 0)
        json.dumps(exported)  # export must stay JSON-serializable
        full = pipeline.compare_candidates(["full-a", "full-b"]).as_dict()
        self.assertIn("identical", full["groups"])
        self.assertIn("same_time_different_basis", full["groups"])

    def test_drift_assumptions_and_anchor_basis_listed(self):
        _, pipeline = registered_pipeline()
        comparison = pipeline.compare_candidates(
            ["full-a", "late-anchor:seg1-end"]
        )
        summaries = {item.name: item for item in comparison.candidates}
        full_a = summaries["full-a"]
        self.assertEqual(len(full_a.drift_assumptions), 3)
        segment_two = full_a.drift_assumptions[2]
        self.assertEqual(segment_two["anchor_count"], 2)
        self.assertNotEqual(segment_two["drift_rate"], 0.0)
        late = summaries["late-anchor:seg1-end"]
        self.assertEqual(
            [item["anchor_id"] for item in late.anchor_basis], ["seg1-end"]
        )
        self.assertEqual(late.anchor_basis[0]["applies_to"], [25, 35])
        self.assertEqual(late.anchor_basis[0]["arrival_time"], 9.5)
        self.assertEqual(full_a.anchor_basis, ())

    def test_comparison_is_read_only(self):
        _, pipeline = registered_pipeline()
        calibration_before = pipeline.calibration.as_dict()
        raw_before = {(o.sensor, o.sequence): o.time for o in pipeline.received}
        statuses_before = {
            name: interp.status for name, interp in pipeline.interpretations.items()
        }
        outputs_before = [state.time for state in pipeline.all_outputs()]
        pipeline.select_for_comparison(["full-a", "full-b"], conflicts_only=True)
        pipeline.filtered_interpretations(CandidateFilter(statuses=("pending",)))
        self.assertEqual(pipeline.calibration.as_dict(), calibration_before)
        self.assertEqual(
            {(o.sensor, o.sequence): o.time for o in pipeline.received}, raw_before
        )
        self.assertEqual(
            {name: interp.status for name, interp in pipeline.interpretations.items()},
            statuses_before,
        )
        self.assertEqual(
            [state.time for state in pipeline.all_outputs()], outputs_before
        )

    def test_selection_validation(self):
        _, pipeline = registered_pipeline()
        with self.assertRaises(ValueError):
            pipeline.compare_candidates(["full-a"])
        with self.assertRaises(KeyError):
            pipeline.compare_candidates(["full-a", "missing"])
        with self.assertRaises(ValueError):
            ComparisonSelection(("full-a", "full-a"))
        pipeline.register_interpretation(
            "imu-guess", "imu", boundaries=[], anchors={0: [(0.0, 0.0)]}
        )
        with self.assertRaises(ValueError):
            pipeline.compare_candidates(["full-a", "imu-guess"])


class ComparisonPersistenceTests(unittest.TestCase):
    def test_filter_and_selection_survive_save_and_restore(self):
        _, pipeline = registered_pipeline()
        pipeline.set_candidate_filter(CandidateFilter(statuses=("pending",)))
        pipeline.select_for_comparison(["full-a", "full-b"], conflicts_only=True)
        before = pipeline.compare_selected()
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
                candidate_filter=checkpoint.candidate_filter,
                comparison_selection=checkpoint.comparison_selection,
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
                candidate_filter=CandidateFilter.from_dict(
                    metadata["candidate_filter"]
                ),
                comparison_selection=ComparisonSelection.from_dict(
                    metadata["comparison_selection"]
                ),
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        self.assertEqual(resumed.candidate_filter, pipeline.candidate_filter)
        self.assertEqual(
            resumed.comparison_selection, pipeline.comparison_selection
        )
        self.assertEqual(
            [entry.name for entry in resumed.filtered_interpretations()],
            [entry.name for entry in pipeline.filtered_interpretations()],
        )
        after = resumed.compare_selected()
        self.assertEqual(after.names, before.names)
        self.assertEqual(after.counts(), before.counts())
        self.assertTrue(after.conflicts_only)
        before_times = {
            (item.sensor, item.sequence): dict(item.corrected_times)
            for item in before.time_conflicts
        }
        for item in after.time_conflicts:
            self.assertEqual(
                dict(item.corrected_times),
                before_times[(item.sensor, item.sequence)],
            )
        # Resuming the view still adopts nothing.
        self.assertTrue(
            all(interp.status == "pending" for interp in resumed.interpretations.values())
        )


if __name__ == "__main__":
    unittest.main()
