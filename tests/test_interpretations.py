import tempfile
import unittest
from pathlib import Path

from robot_localization.calibration import DelayCalibration
from robot_localization.pipeline import (
    Checkpoint,
    Interpretation,
    LocalizationPipeline,
    RegisteredAnchor,
)
from robot_localization.recording import CheckpointStore
from robot_localization.segments import RebootBoundary, SegmentedSensorClock
from robot_localization.sources import RebootScenarioSource


SENSOR = "rear_wheels"


def reboot_pipeline(calibration: DelayCalibration | None = None) -> LocalizationPipeline:
    source = RebootScenarioSource()
    pipeline = LocalizationPipeline(calibration=calibration)
    for observation in source.observations():
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
    return pipeline


def partial_calibration(source: RebootScenarioSource) -> DelayCalibration:
    """Reboot clock where segment 1 has only a single anchor (weak drift)."""
    anchors = source.anchors()
    clock = SegmentedSensorClock.from_anchors(
        SENSOR,
        {0: anchors[0], 1: anchors[1][:1], 2: anchors[2]}.items(),
        [RebootBoundary(20, True), RebootBoundary(40, True)],
    )
    return DelayCalibration().with_reboot_clock(clock)


class SequenceRangeInterpretationTests(unittest.TestCase):
    def test_multiple_boundary_interpretations_compared_without_overwrite(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline()
        original = {(o.sensor, o.sequence): o.time for o in pipeline.received}

        def boundaries(cut: int) -> list[RebootBoundary]:
            return [
                RebootBoundary(source.REBOOTS[0][1], True),
                RebootBoundary(cut, False, (7.7, 8.3), (40, 44)),
            ]

        narrow = pipeline.register_interpretation(
            "cut-40", SENSOR, boundaries=boundaries(40), segment_anchors=source.anchors()
        )
        wide = pipeline.register_interpretation(
            "cut-44", SENSOR, boundaries=boundaries(44), segment_anchors=source.anchors()
        )
        # Both candidates stay pending; the later one did not overwrite the first.
        self.assertEqual(
            set(pipeline.interpretation_comparisons()), {"cut-40", "cut-44"}
        )
        with self.assertRaises(ValueError):
            pipeline.register_interpretation(
                "cut-40", SENSOR, boundaries=boundaries(40),
                segment_anchors=source.anchors(),
            )
        # The reset sequence window is reported, not resolved to an instant.
        view = next(item for item in narrow.segment_views if item.index == 2)
        self.assertEqual(view.sequence_window, (40, 44))
        self.assertEqual(view.status, "boundary_uncertain")
        # The two cuts disagree on which segment sequences 41/42 belong to,
        # so their corrected times differ per interpretation.
        narrow_times = {
            item.sequence: item.corrected_time for item in narrow.decisions
        }
        wide_times = {
            item.sequence: item.corrected_time for item in wide.decisions
        }
        self.assertNotAlmostEqual(narrow_times[41], wide_times[41])
        self.assertNotAlmostEqual(narrow_times[42], wide_times[42])
        self.assertAlmostEqual(narrow_times[30], wide_times[30])
        # Downstream trajectory effect is attributable per identity.
        self.assertTrue(narrow.position_changes)
        self.assertEqual(
            {change.identity for change in narrow.position_changes},
            {change.identity for change in wide.position_changes},
        )
        # Nothing was published: raw stamps, identities and calibration stay.
        self.assertIsNone(pipeline.calibration.reboot_clock_for(SENSOR))
        self.assertEqual(
            {(o.sensor, o.sequence): o.time for o in pipeline.received}, original
        )
        # Publishing is an explicit named choice, never the latest candidate.
        report = pipeline.publish_interpretation("cut-40")
        self.assertEqual(report.sensor, SENSOR)
        self.assertIn("cut-44", pipeline.interpretation_comparisons())
        clock = pipeline.calibration.reboot_clock_for(SENSOR)
        self.assertEqual(clock.boundaries[1].after_sequence, 40)
        self.assertEqual(clock.boundaries[1].sequence_range, (40, 44))
        self.assertEqual(clock.segment_index_for(41), 2)


class LateAnchorPartialAdoptionTests(unittest.TestCase):
    def test_late_anchor_interpretation_adopts_only_listed_observations(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(partial_calibration(source))
        # Identical raw stamps exist on both sides of the reset.
        stamps = {}
        for observation in pipeline.received:
            stamps.setdefault(observation.time, []).append(observation.sequence)
        duplicated = [
            sequences for sequences in stamps.values() if len(sequences) > 1
        ]
        self.assertTrue(duplicated)

        def baseline_time(sequence: int) -> float:
            observation = next(
                item for item in pipeline.received if item.sequence == sequence
            )
            return pipeline.calibration.corrected_time(observation)

        # A late anchor for segment 1 arrives after the duplicates were seen.
        record = pipeline.register_anchor(
            SENSOR,
            1,
            (source.stamp_for(6.0, 1), 6.0),
            late=True,
            note="恢复后补传的段内锚点",
        )
        self.assertTrue(record.late)
        adopted = [(SENSOR, 25), (SENSOR, 30)]
        comparison = pipeline.register_interpretation(
            "late-anchor", SENSOR, anchor_ids=[record.anchor_id], adopt=adopted
        )
        decisions = {item.sequence: item for item in comparison.decisions}
        # Adopted observations move to the two-anchor segment explanation.
        self.assertNotAlmostEqual(
            decisions[25].corrected_time, decisions[25].baseline_corrected_time
        )
        # Every other observation keeps its original corrected time.
        for sequence, decision in decisions.items():
            if sequence in {25, 30}:
                continue
            self.assertAlmostEqual(
                decision.corrected_time, decision.baseline_corrected_time
            )
            self.assertAlmostEqual(decision.time_shift, 0.0)
        baseline_25 = baseline_time(25)
        baseline_26 = baseline_time(26)
        pipeline.publish_interpretation("late-anchor")
        # After publishing, non-adopted observations still use the old version.
        self.assertAlmostEqual(
            pipeline.calibration.corrected_time(
                next(item for item in pipeline.received if item.sequence == 26)
            ),
            baseline_26,
        )
        self.assertNotAlmostEqual(
            pipeline.calibration.corrected_time(
                next(item for item in pipeline.received if item.sequence == 25)
            ),
            baseline_25,
        )
        # Raw stamps and identities are untouched by the partial adoption.
        self.assertEqual(
            pipeline.calibration.adopted_identities(),
            frozenset(adopted),
        )

    def test_withdraw_anchor_lists_dependent_interpretations(self) -> None:
        source = RebootScenarioSource()
        pipeline = reboot_pipeline(partial_calibration(source))
        anchor_a = pipeline.register_anchor(
            SENSOR, 1, (source.stamp_for(6.0, 1), 6.0), late=True
        )
        anchor_b = pipeline.register_anchor(
            SENSOR, 2, (source.stamp_for(9.0, 2), 9.0)
        )
        pipeline.register_interpretation(
            "uses-a", SENSOR, anchor_ids=[anchor_a.anchor_id]
        )
        pipeline.register_interpretation(
            "uses-both",
            SENSOR,
            anchor_ids=[anchor_a.anchor_id, anchor_b.anchor_id],
        )
        pipeline.register_interpretation(
            "uses-b", SENSOR, anchor_ids=[anchor_b.anchor_id]
        )
        published_before = pipeline.calibration
        withdrawal = pipeline.withdraw_anchor(anchor_a.anchor_id)
        self.assertEqual(
            set(withdrawal.dropped_interpretations), {"uses-a", "uses-both"}
        )
        self.assertEqual(withdrawal.anchor_id, anchor_a.anchor_id)
        # The independent interpretation and the published clock stay as they were.
        self.assertEqual(set(pipeline.interpretation_comparisons()), {"uses-b"})
        self.assertIs(pipeline.calibration, published_before)
        with self.assertRaises(ValueError):
            pipeline.withdraw_anchor(anchor_a.anchor_id)


class InterpretationCheckpointTests(unittest.TestCase):
    def test_pending_candidates_survive_save_and_reopen(self) -> None:
        source = RebootScenarioSource()
        base_clock = partial_calibration(source).reboot_clock_for(SENSOR)
        calibration = partial_calibration(source).with_reboot_clock(
            base_clock.replacing(
                {},
                {1: RebootBoundary(40, False, (7.7, 8.3), (38, 42))},
            )
        )
        pipeline = reboot_pipeline(calibration)
        pipeline.seal_until(3.0)
        record = pipeline.register_anchor(
            SENSOR, 1, (source.stamp_for(6.0, 1), 6.0, 0.3), late=True
        )
        adopted = [(SENSOR, 25), (SENSOR, 30)]
        pipeline.register_interpretation(
            "late-anchor", SENSOR, anchor_ids=[record.anchor_id], adopt=adopted
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
                registered_anchors=[
                    item.as_dict() for item in checkpoint.registered_anchors
                ],
                pending_interpretations=[
                    item.as_dict() for item in checkpoint.pending_interpretations
                ],
            )
            state, observations, metadata = CheckpointStore(path).load_with_metadata()
        restored_checkpoint = Checkpoint(
            state=state,
            event_count=checkpoint.event_count,
            arrival_horizon=metadata["arrival_horizon"],
            seal_time=metadata["seal_time"],
            calibration=DelayCalibration(metadata["clock_models"]),
            visible_observations=observations,
            sealed_outputs=metadata["sealed_outputs"],
            rejected_observations=metadata["rejected_observations"],
            registered_anchors=tuple(
                RegisteredAnchor.from_dict(item)
                for item in metadata["registered_anchors"]
            ),
            pending_interpretations=tuple(
                Interpretation.from_dict(item)
                for item in metadata["pending_interpretations"]
            ),
        )
        resumed = LocalizationPipeline()
        resumed.restore_checkpoint(restored_checkpoint)
        # The reset window and the pending candidate are still traceable.
        clock = resumed.calibration.reboot_clock_for(SENSOR)
        self.assertEqual(clock.boundaries[1].sequence_range, (38, 42))
        self.assertIn(record.anchor_id, resumed.registered_anchors)
        self.assertTrue(resumed.registered_anchors[record.anchor_id].late)
        interpretations = resumed.interpretation_comparisons()
        self.assertEqual(set(interpretations), {"late-anchor"})
        interpretation = resumed.interpretations["late-anchor"]
        self.assertEqual(interpretation.adopted_identities, frozenset(adopted))
        comparison = interpretations["late-anchor"]
        # Ranges and downstream adoption basis are recomputed after reopening.
        decisions = {item.sequence: item for item in comparison.decisions}
        self.assertIsNotNone(decisions[25].candidate_range)
        self.assertGreater(
            decisions[25].candidate_range[1], decisions[25].candidate_range[0]
        )
        adopted_changes = {
            change.identity for change in comparison.position_changes
        }
        self.assertIn((SENSOR, 25), adopted_changes)
        # Publishing after reopen only moves the adopted identities.
        baseline_26 = resumed.calibration.corrected_time(
            next(item for item in resumed.received if item.sequence == 26)
        )
        resumed.publish_interpretation("late-anchor")
        self.assertAlmostEqual(
            resumed.calibration.corrected_time(
                next(item for item in resumed.received if item.sequence == 26)
            ),
            baseline_26,
        )
        self.assertEqual(
            resumed.calibration.adopted_identities(), frozenset(adopted)
        )


if __name__ == "__main__":
    unittest.main()
