"""Filtering, set comparison and saved views for clock interpretations."""

import json
import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.interpretations import AnchorRecord, Interpretation
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import CheckpointStore
from robot_localization.segments import RebootBoundary
from robot_localization.sources import RebootScenarioSource


SENSOR = "rear_wheels"


def make_pipeline(segment_anchors):
    source = RebootScenarioSource()
    pipeline = LocalizationPipeline()
    for observation in source.observations():
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
    pipeline.stage_reboot_segments(
        SENSOR,
        boundaries=[
            RebootBoundary(source.REBOOTS[0][1], True),
            RebootBoundary(source.REBOOTS[1][1], False, (7.7, 8.3)),
        ],
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


def densified_anchors(source):
    """Same corrected times everywhere, but backed by extra midpoint anchors."""
    anchors = full_anchors(source)
    for index, points in anchors.items():
        (s0, c0), (s1, c1) = points
        anchors[index] = [(s0, c0), ((s0 + s1) / 2, (c0 + c1) / 2), (s1, c1)]
    return anchors


def boundaries():
    return [
        RebootBoundary(20, True),
        RebootBoundary(40, False, (7.7, 8.3)),
    ]


def register_three(pipeline, source):
    pipeline.register_anchor(
        "gps-seg2", SENSOR, 2, source.anchors()[2][0], source="gps"
    )
    pipeline.register_interpretation(
        "full-base",
        SENSOR,
        boundaries=boundaries(),
        anchors=full_anchors(source),
        anchor_ids=("gps-seg2",),
    )
    pipeline.register_interpretation(
        "full-shifted",
        SENSOR,
        boundaries=boundaries(),
        anchors=shifted_anchors(source, -0.2),
    )
    pipeline.register_late_anchor(
        "manual-seg1",
        SENSOR,
        1,
        (
            (source.anchors()[1][0][0] + source.anchors()[1][1][0]) / 2,
            (source.anchors()[1][0][1] + source.anchors()[1][1][1]) / 2,
        ),
        applies_to=(25, 35),
        arrival_time=9.5,
        source="manual",
    )


class FilterInterpretationTests(unittest.TestCase):
    def test_filter_by_status_anchor_source_and_sequence_range(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        all_names = {entry.name for entry in pipeline.filter_interpretations()}
        self.assertEqual(
            all_names, {"full-base", "full-shifted", "late-anchor:manual-seg1"}
        )
        pending = {
            entry.name
            for entry in pipeline.filter_interpretations(status="pending")
        }
        self.assertEqual(pending, all_names)
        gps = {
            entry.name
            for entry in pipeline.filter_interpretations(anchor_source="gps")
        }
        self.assertEqual(gps, {"full-base"})
        manual = {
            entry.name
            for entry in pipeline.filter_interpretations(anchor_source="manual")
        }
        self.assertEqual(manual, {"late-anchor:manual-seg1"})
        # Observation range: the partial candidate only covers 25..35.
        late_window = {
            entry.name
            for entry in pipeline.filter_interpretations(sequence_range=(26, 30))
        }
        self.assertEqual(late_window, all_names)
        early_window = {
            entry.name
            for entry in pipeline.filter_interpretations(sequence_range=(1, 10))
        }
        self.assertEqual(early_window, {"full-base", "full-shifted"})
        # Status changes are reflected after an explicit adoption.
        pipeline.adopt_interpretation("full-base")
        adopted = {
            entry.name
            for entry in pipeline.filter_interpretations(status="adopted")
        }
        self.assertEqual(adopted, {"full-base"})
        still_pending = {
            entry.name
            for entry in pipeline.filter_interpretations(status="pending")
        }
        self.assertEqual(still_pending, {"full-shifted", "late-anchor:manual-seg1"})


class SetComparisonTests(unittest.TestCase):
    def test_three_way_grouping_and_candidate_summaries(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        pipeline.register_interpretation(
            "full-densified",
            SENSOR,
            boundaries=boundaries(),
            anchors=densified_anchors(source),
        )
        comparison = pipeline.compare_interpretation_set(
            ["full-base", "full-shifted", "full-densified"]
        )
        self.assertEqual(comparison.sensor, SENSOR)
        self.assertEqual(
            comparison.names, ("full-base", "full-shifted", "full-densified")
        )
        # Shifted segment-2 anchors move corrected times there only.
        shifted_sequences = {item.sequence for item in comparison.time_differs}
        self.assertTrue(shifted_sequences)
        self.assertTrue(all(sequence > 40 for sequence in shifted_sequences))
        for item in comparison.time_differs:
            base, shifted, _ = item.corrected_times
            self.assertAlmostEqual(base - shifted, 0.2)
            self.assertGreater(item.max_time_spread, 0.1)
        # Densified anchors retime nothing but rest on a different basis.
        same_time = {
            item.sequence for item in comparison.same_time_different_basis
        }
        self.assertTrue(same_time)
        self.assertTrue(all(sequence <= 40 for sequence in same_time))
        # Nothing is fully identical: the densified basis differs everywhere.
        self.assertEqual(comparison.identical, ())
        self.assertEqual(comparison.identical_count, 0)
        # Candidate summaries expose drift assumptions and anchor basis.
        summaries = {item.name: item for item in comparison.candidates}
        self.assertEqual(
            set(summaries), {"full-base", "full-shifted", "full-densified"}
        )
        base_summary = summaries["full-base"]
        self.assertEqual(len(base_summary.drift_assumptions), 3)
        self.assertEqual(base_summary.anchor_basis[0]["source"], "gps")
        densified = summaries["full-densified"]
        self.assertEqual(densified.drift_assumptions[0]["anchor_count"], 3)
        for summary in summaries.values():
            for segment in summary.drift_assumptions:
                self.assertIn("implied_drift", segment)

    def test_identical_group_and_conflicts_only(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        pipeline.register_interpretation(
            "full-twin",
            SENSOR,
            boundaries=boundaries(),
            anchors=full_anchors(source),
        )
        comparison = pipeline.compare_interpretation_set(
            ["full-base", "full-twin", "full-shifted"]
        )
        # Segments 0/1 agree in time and basis between base and twin, but the
        # shifted candidate uses different segment-2 anchors only, so earlier
        # segments are fully identical across all three.
        identical_sequences = {item.sequence for item in comparison.identical}
        self.assertTrue(identical_sequences)
        self.assertTrue(all(sequence <= 40 for sequence in identical_sequences))
        self.assertEqual(comparison.identical_count, len(comparison.identical))
        conflicts = pipeline.compare_interpretation_set(
            ["full-base", "full-twin", "full-shifted"], conflicts_only=True
        )
        self.assertTrue(conflicts.conflicts_only)
        self.assertEqual(conflicts.identical, ())
        self.assertEqual(conflicts.identical_count, len(identical_sequences))
        self.assertEqual(
            {item.sequence for item in conflicts.time_differs},
            {item.sequence for item in comparison.time_differs},
        )

    def test_partial_candidate_overlap_is_the_adopted_range(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline(
            {0: source.anchors()[0], 1: source.anchors()[1][:1], 2: source.anchors()[2]}
        )
        register_three(pipeline, source)
        comparison = pipeline.compare_interpretation_set(
            ["full-base", "late-anchor:manual-seg1"]
        )
        covered = (
            {item.sequence for item in comparison.time_differs}
            | {item.sequence for item in comparison.same_time_different_basis}
            | {item.sequence for item in comparison.identical}
        )
        self.assertEqual(covered, set(range(25, 36)))

    def test_export_writes_json_and_needs_two_names(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        with self.assertRaises(ValueError):
            pipeline.compare_interpretation_set(["full-base"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            comparison = pipeline.export_interpretation_set(
                ["full-base", "full-shifted"], path, conflicts_only=True
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["names"], ["full-base", "full-shifted"])
        self.assertTrue(payload["conflicts_only"])
        self.assertEqual(
            payload["counts"]["time_differs"], len(comparison.time_differs)
        )
        self.assertEqual(payload["groups"]["identical"], [])
        # The in-memory export is JSON-serializable as well.
        json.dumps(comparison.as_dict(), ensure_ascii=False)

    def test_comparison_never_adopts_or_rewrites(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        raw_before = {(o.sensor, o.sequence): o.time for o in pipeline.received}
        calibration_before = pipeline.calibration.as_dict()
        sealed_before = [state.time for state in pipeline.sealed_outputs]
        outputs_before = [state.time for state in pipeline.outputs]
        pipeline.filter_interpretations(status="pending")
        pipeline.compare_interpretation_set(["full-base", "full-shifted"])
        pipeline.save_comparison_view(
            filters={"status": "pending"},
            selection=["full-base", "full-shifted"],
        )
        pipeline.resume_comparison_view()
        self.assertEqual(pipeline.calibration.as_dict(), calibration_before)
        self.assertEqual(
            {(o.sensor, o.sequence): o.time for o in pipeline.received}, raw_before
        )
        self.assertEqual(
            [state.time for state in pipeline.sealed_outputs], sealed_before
        )
        self.assertEqual([state.time for state in pipeline.outputs], outputs_before)
        self.assertTrue(
            all(
                interp.status == "pending"
                for interp in pipeline.interpretations.values()
            )
        )


class SavedComparisonViewTests(unittest.TestCase):
    def test_view_survives_checkpoint_save_and_restore(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline({0: source.anchors()[0], 1: source.anchors()[1]})
        register_three(pipeline, source)
        pipeline.save_comparison_view(
            filters={"sensor": SENSOR, "status": "pending"},
            selection=["full-base", "full-shifted"],
            conflicts_only=True,
        )
        before = pipeline.resume_comparison_view()
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
                comparison_view=checkpoint.comparison_view,
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
                comparison_view=metadata["comparison_view"],
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        after = resumed.resume_comparison_view()
        self.assertEqual(after["filters"], before["filters"])
        self.assertEqual(after["selection"], ["full-base", "full-shifted"])
        self.assertTrue(after["conflicts_only"])
        self.assertEqual(after["matches"], before["matches"])
        self.assertIsNotNone(after["comparison"])
        self.assertEqual(
            after["comparison"].identical_count,
            before["comparison"].identical_count,
        )
        self.assertEqual(
            {item.sequence for item in after["comparison"].time_differs},
            {item.sequence for item in before["comparison"].time_differs},
        )
        # Reopening changes nothing: candidates stay pending and adoptable.
        self.assertEqual(resumed.interpretations["full-base"].status, "pending")
        report = resumed.adopt_interpretation("full-base")
        self.assertEqual(report.sensor, SENSOR)


if __name__ == '__main__':
    unittest.main()
