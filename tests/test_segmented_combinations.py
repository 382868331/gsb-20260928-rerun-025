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

from tests.test_interpretations import SENSOR, full_anchors, make_pipeline, shifted_anchors


def boundaries():
    return [
        RebootBoundary(20, True),
        RebootBoundary(40, False, (7.7, 8.3)),
    ]


def ordered_alt_anchors(source):
    anchors = full_anchors(source)
    anchors[2] = [(stamp, corrected + 0.05) for stamp, corrected in anchors[2]]
    return anchors


class SegmentedCombinationTests(unittest.TestCase):
    def test_combine_ranges_keeps_boundary_sources_and_changes_order(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline(
            {0: source.anchors()[0], 1: source.anchors()[1]}
        )
        pipeline.add_observation(ImuObservation(8.25, "imu", 1, arrival_time=10.15, yaw_rate=0.0, forward_acceleration=0.0))
        pipeline.process_until(10.2)
        pipeline.register_interpretation(
            "base", SENSOR, boundaries=boundaries(), anchors=full_anchors(source)
        )
        pipeline.register_interpretation(
            "shifted",
            SENSOR,
            boundaries=boundaries(),
            anchors=ordered_alt_anchors(source),
        )
        comparison = pipeline.register_segmented_combination(
            "mixed",
            SENSOR,
            [("base", 0, 40), ("shifted", 41, 50)],
        )
        clock = pipeline.interpretations["mixed"].clock
        self.assertEqual(
            [(item.source_name, item.first_sequence, item.last_sequence) for item in clock.ranges],
            [("base", 0, 40), ("shifted", 41, 50)],
        )
        self.assertEqual(comparison.reboot_contradictions, ())
        effects = pipeline.compare_combination_order_effects("mixed")
        affected = {effect.identity: effect for effect in effects}
        self.assertIn((SENSOR, 41), affected)
        self.assertIn(("imu", 1), affected)
        relations = {
            (sensor, sequence): relation
            for sensor, sequence, relation
            in affected[(SENSOR, 41)].changed_relations
        }
        self.assertEqual(relations[("imu", 1)], "before→after")
        report = pipeline.adopt_interpretation("mixed")
        self.assertEqual(report.sensor, SENSOR)
        self.assertEqual(pipeline.interpretations["base"].status, "pending")
        self.assertEqual(pipeline.interpretations["shifted"].status, "pending")

    def test_boundary_conflict_blocks_adoption_without_smoothing(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline(
            {0: source.anchors()[0], 1: source.anchors()[1]}
        )
        pipeline.register_interpretation(
            "base", SENSOR, boundaries=boundaries(), anchors=full_anchors(source)
        )
        pipeline.register_interpretation(
            "reversed",
            SENSOR,
            boundaries=boundaries(),
            anchors=shifted_anchors(source, -1.0),
        )
        comparison = pipeline.register_segmented_combination(
            "bad",
            SENSOR,
            [("base", 0, 40), ("reversed", 41, 50)],
        )
        kinds = {item["kind"] for item in comparison.reboot_contradictions}
        self.assertIn("boundary_time_reversal", kinds)
        conflict = next(
            item for item in comparison.reboot_contradictions
            if item["kind"] == "boundary_time_reversal"
        )
        self.assertEqual(conflict["sources"], ["base", "reversed"])
        self.assertEqual(
            conflict["boundary_between_sequences"], [40, 41]
        )
        self.assertEqual(conflict["identities"], [[SENSOR, 40], [SENSOR, 41]])
        with self.assertRaises(ValueError):
            pipeline.adopt_interpretation("bad")
        self.assertEqual(pipeline.interpretations["bad"].status, "pending")

    def test_supplement_anchor_rechecks_only_affected_range_and_survives_restore(self):
        source = RebootScenarioSource()
        _, pipeline = make_pipeline(
            {0: source.anchors()[0], 1: source.anchors()[1]}
        )
        pipeline.register_interpretation(
            "base", SENSOR, boundaries=boundaries(), anchors=full_anchors(source)
        )
        pipeline.register_interpretation(
            "shifted",
            SENSOR,
            boundaries=boundaries(),
            anchors=ordered_alt_anchors(source),
        )
        pipeline.register_segmented_combination(
            "mixed",
            SENSOR,
            [("base", 0, 40), ("shifted", 41, 50)],
        )
        stamp = source.stamp_for(9.0, 2)
        pipeline.supplement_combination_anchor(
            "mixed",
            "new-seg2",
            2,
            (stamp, 9.0),
            applies_to=(41, 50),
            arrival_time=10.1,
            source="manual-review",
        )
        version_name = "mixed:new-seg2"
        interp = pipeline.interpretations[version_name]
        self.assertIn("new-seg2", interp.anchor_ids)
        range_item = interp.clock.ranges[1]
        self.assertEqual(range_item.anchor_ids, ("new-seg2",))
        self.assertEqual(
            [anchor.stamp for anchor in range_item.clock.segments[2].anchors],
            sorted(anchor.stamp for anchor in range_item.clock.segments[2].anchors),
        )
        comparison = pipeline.interpretation_comparison(version_name)
        self.assertEqual(comparison.reboot_contradictions, ())
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
            restored = type(checkpoint)(
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
        resumed.restore_checkpoint(restored)
        resumed_clock = resumed.interpretations[version_name].clock
        self.assertEqual(
            [(item.source_name, item.first_sequence, item.last_sequence) for item in resumed_clock.ranges],
            [("base", 0, 40), ("shifted", 41, 50)],
        )
        self.assertEqual(resumed_clock.ranges[1].anchor_ids, ("new-seg2",))
        self.assertEqual(
            resumed.interpretation_comparison(version_name).reboot_contradictions,
            (),
        )


if __name__ == "__main__":
    unittest.main()
