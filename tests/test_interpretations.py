"""Multiple reset explanations, late anchors and anchor retraction."""

import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.interpretations import AnchorRecord, Interpretation
from robot_localization.pipeline import LocalizationPipeline
from robot_localization.recording import CheckpointStore
from robot_localization.segments import RebootBoundary
from robot_localization.sources import RebootScenarioSource
from robot_localization.types import WheelObservation


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


class MultipleInterpretationTests(unittest.TestCase):
    def test_boundary_drift_candidates_compared_without_auto_override(self):
        source = RebootScenarioSource()
        known = source.anchors()
        _, pipeline = make_pipeline({0: known[0], 1: known[1]})
        original = {(o.sensor, o.sequence): o.time for o in pipeline.received}
        comparison_a = pipeline.register_interpretation(
            "reset-window-a",
            SENSOR,
            boundaries=[
                RebootBoundary(20, True),
                RebootBoundary(40, False, (7.7, 8.3)),
            ],
            anchors=full_anchors(source),
        )
        comparison_b = pipeline.register_interpretation(
            "reset-window-b",
            SENSOR,
            boundaries=[
                RebootBoundary(20, True),
                RebootBoundary(40, False, (7.8, 8.4)),
            ],
            anchors=shifted_anchors(source, -0.2),
        )
        # Registering never publishes: segment 2 stays uncalibrated live.
        live = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertFalse(live.segments[2].calibrated)
        # Both candidates remain registered and pending.
        entries = {
            entry.name: entry for entry in pipeline.compare_interpretations(SENSOR)
        }
        self.assertEqual(set(entries), {"reset-window-a", "reset-window-b"})
        self.assertTrue(all(entry.status == "pending" for entry in entries.values()))
        # Same identities, different corrected times and downstream effects.
        times_a = {d.sequence: d.corrected_time for d in comparison_a.decisions}
        times_b = {d.sequence: d.corrected_time for d in comparison_b.decisions}
        self.assertEqual(set(times_a), set(times_b))
        self.assertAlmostEqual(times_a[45] - times_b[45], 0.2)
        self.assertTrue(comparison_a.position_changes)
        self.assertEqual(
            {c.identity for c in comparison_a.position_changes},
            {c.identity for c in comparison_b.position_changes},
        )
        self.assertEqual(
            entries["reset-window-a"].comparison.recomputable_range,
            comparison_a.recomputable_range,
        )
        # Raw time, sequence and identity are untouched.
        self.assertEqual(
            {(o.sensor, o.sequence): o.time for o in pipeline.received}, original
        )
        # A later registration can never replace an earlier one implicitly.
        with self.assertRaises(ValueError):
            pipeline.register_interpretation(
                "reset-window-a",
                SENSOR,
                boundaries=[
                    RebootBoundary(20, True),
                    RebootBoundary(40, False, (7.7, 8.3)),
                ],
                anchors=full_anchors(source),
            )

    def test_adopt_is_always_explicit(self):
        source = RebootScenarioSource()
        known = source.anchors()
        _, pipeline = make_pipeline({0: known[0], 1: known[1]})
        boundaries = [
            RebootBoundary(20, True),
            RebootBoundary(40, False, (7.7, 8.3)),
        ]
        pipeline.register_interpretation(
            "first", SENSOR, boundaries=boundaries, anchors=full_anchors(source)
        )
        pipeline.register_interpretation(
            "second",
            SENSOR,
            boundaries=boundaries,
            anchors=shifted_anchors(source, -0.2),
        )
        # The most recently registered candidate is not applied automatically.
        self.assertFalse(
            pipeline.calibration.reboot_clock_for(SENSOR).segments[2].calibrated
        )
        report = pipeline.adopt_interpretation("first")
        self.assertEqual(report.sensor, SENSOR)
        clock = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertTrue(clock.segments[2].calibrated)
        self.assertEqual(pipeline.interpretations["first"].status, "adopted")
        self.assertEqual(pipeline.interpretations["second"].status, "pending")
        with self.assertRaises(RuntimeError):
            pipeline.adopt_interpretation("first")


class LateAnchorTests(unittest.TestCase):
    def test_late_anchor_partial_adoption_keeps_other_observations(self):
        source = RebootScenarioSource()
        known = source.anchors()
        _, pipeline = make_pipeline(
            {0: known[0], 1: known[1][:1], 2: known[2]}
        )
        baseline = {
            o.sequence: pipeline.calibration.corrected_time(o)
            for o in pipeline.received
        }
        comparison = pipeline.register_late_anchor(
            "seg1-end",
            SENSOR,
            1,
            known[1][1],
            applies_to=(25, 35),
            arrival_time=9.5,
        )
        interpretation = pipeline.interpretations["late-anchor:seg1-end"]
        self.assertEqual(interpretation.adopted_sequences, tuple(range(25, 36)))
        shifts = {d.sequence: d.time_shift for d in comparison.decisions}
        for sequence, shift in shifts.items():
            if 25 <= sequence <= 35:
                continue
            self.assertAlmostEqual(shift, 0.0)
        self.assertTrue(any(abs(shifts[s]) > 1e-9 for s in range(25, 36)))
        # Identical raw stamps on both sides of the reset stay distinct.
        first = WheelObservation(0.25, SENSOR, 21, 1.0, 0.0)
        later = WheelObservation(0.25, SENSOR, 41, 1.0, 0.0)
        self.assertNotAlmostEqual(
            pipeline.calibration.corrected_time(first),
            pipeline.calibration.corrected_time(later),
        )
        # Adopting retimes only the chosen range; everything else keeps version.
        pipeline.adopt_interpretation("late-anchor:seg1-end")
        for observation in pipeline.received:
            corrected = pipeline.calibration.corrected_time(observation)
            if 25 <= observation.sequence <= 35:
                self.assertNotAlmostEqual(corrected, baseline[observation.sequence])
            else:
                self.assertAlmostEqual(corrected, baseline[observation.sequence])
        times = [state.time for state in pipeline.outputs]
        self.assertEqual(times, sorted(times))


class AnchorRetractionTests(unittest.TestCase):
    def test_retract_anchor_lists_dependents_and_keeps_others(self):
        source = RebootScenarioSource()
        known = source.anchors()
        _, pipeline = make_pipeline({0: known[0], 1: known[1]})
        pipeline.register_anchor("seg2-start", SENSOR, 2, known[2][0])
        boundaries = [
            RebootBoundary(20, True),
            RebootBoundary(40, False, (7.7, 8.3)),
        ]
        pipeline.register_interpretation(
            "uses-anchor",
            SENSOR,
            boundaries=boundaries,
            anchors=full_anchors(source),
            anchor_ids=("seg2-start",),
        )
        pipeline.register_interpretation(
            "independent",
            SENSOR,
            boundaries=boundaries,
            anchors=full_anchors(source),
        )
        retraction = pipeline.retract_anchor("seg2-start")
        self.assertEqual(retraction.dependent_interpretations, ("uses-anchor",))
        self.assertEqual(
            pipeline.interpretations["uses-anchor"].status, "anchor_retracted"
        )
        with self.assertRaises(RuntimeError):
            pipeline.adopt_interpretation("uses-anchor")
        self.assertEqual(pipeline.interpretations["independent"].status, "pending")
        report = pipeline.adopt_interpretation("independent")
        self.assertEqual(report.sensor, SENSOR)
        clock = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertTrue(clock.segments[2].calibrated)
        with self.assertRaises(KeyError):
            pipeline.retract_anchor("seg2-start")


class InterpretationCheckpointTests(unittest.TestCase):
    def test_pending_candidates_survive_save_and_restore(self):
        source = RebootScenarioSource()
        known = source.anchors()
        _, pipeline = make_pipeline(
            {0: known[0], 1: known[1][:1], 2: known[2]}
        )
        pipeline.register_late_anchor(
            "seg1-end",
            SENSOR,
            1,
            known[1][1],
            applies_to=(25, 35),
            arrival_time=9.5,
        )
        boundaries = [
            RebootBoundary(20, True),
            RebootBoundary(40, False, (7.7, 8.3)),
        ]
        pipeline.register_interpretation(
            "full-rewrite",
            SENSOR,
            boundaries=boundaries,
            anchors=full_anchors(source),
        )
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
            )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        # Pending candidates and their adoption scope are still traceable.
        self.assertEqual(set(resumed.interpretations), set(pipeline.interpretations))
        restored_interp = resumed.interpretations["late-anchor:seg1-end"]
        self.assertEqual(restored_interp.status, "pending")
        self.assertEqual(restored_interp.adopted_sequences, tuple(range(25, 36)))
        self.assertEqual(restored_interp.anchor_ids, ("seg1-end",))
        self.assertEqual(resumed.anchor_records["seg1-end"].applies_to, (25, 35))
        # Ranges and downstream adoption basis match the pre-save comparison.
        before = {
            d.sequence: d.candidate_range
            for d in pipeline.interpretation_comparison("late-anchor:seg1-end").decisions
        }
        after = {
            d.sequence: d.candidate_range
            for d in resumed.interpretation_comparison("late-anchor:seg1-end").decisions
        }
        self.assertEqual(set(before), set(after))
        for sequence in before:
            self.assertAlmostEqual(before[sequence][0], after[sequence][0])
            self.assertAlmostEqual(before[sequence][1], after[sequence][1])
        # Dependency tracking still works after reopening.
        retraction = resumed.retract_anchor("seg1-end")
        self.assertEqual(
            retraction.dependent_interpretations, ("late-anchor:seg1-end",)
        )
        # The other candidate is unaffected and can still be adopted.
        report = resumed.adopt_interpretation("full-rewrite")
        self.assertEqual(report.sensor, SENSOR)


if __name__ == '__main__':
    unittest.main()
