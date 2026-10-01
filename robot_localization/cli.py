from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .calibration import DelayCalibration
from .pipeline import (
    Checkpoint,
    ClockComparison,
    ClockRevisionReport,
    DelayComparison,
    LocalizationPipeline,
    RecomputeReport,
)
from .recording import CheckpointStore, JsonlRecorder
from .sources import SyntheticScenarioSource


TRUE_DELAYS = {"rear_wheels": 0.30, "imu": 0.12}
INITIAL_GUESS = {"rear_wheels": 0.10, "imu": 0.12}
SEAL_AT = 5.0
WHEEL_CLOCK_DRIFT = 0.02


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline robot localization sample")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = SyntheticScenarioSource(
        fixed_delays=TRUE_DELAYS,
        clock_drifts={"rear_wheels": WHEEL_CLOCK_DRIFT},
    )
    messages = source.observations()
    JsonlRecorder(args.output_dir / "messages.jsonl").write_many(messages)

    truth = source.ground_truth()

    # Baseline: messages fused as if every transport delay were zero.
    uncorrected = LocalizationPipeline()
    for observation in messages:
        uncorrected.add_observation(observation)
        uncorrected.process_until(observation.observed_at())

    # Corrected online run with an initial (partly wrong) fixed delay guess.
    pipeline = LocalizationPipeline(
        calibration=DelayCalibration(INITIAL_GUESS)
    )
    recompute_events: list[dict[str, object]] = []
    delay_comparison: dict[str, object] | None = None
    discarded_revision: dict[str, object] | None = None
    clock_draft: dict[str, object] | None = None
    clock_publish: dict[str, object] | None = None
    saved_checkpoint = None
    boundary_at_checkpoint = None
    for index, observation in enumerate(messages):
        pipeline.add_observation(observation)
        pipeline.process_until(observation.observed_at())
        # Before sealing, the user refines the wheel delay from live evidence.
        # Only the currently visible interval is recomputed.
        if (
            not recompute_events
            and observation.observed_at() >= TRUE_DELAYS["rear_wheels"] + 2.5
        ):
            report = pipeline.set_sensor_delay(
                "rear_wheels", TRUE_DELAYS["rear_wheels"]
            )
            recompute_events.append(_describe_report(report))
        # Seal the first 5 s of corrected time once it has been observed.
        if (
            pipeline.seal_time is None
            and observation.observed_at() >= SEAL_AT + TRUE_DELAYS["rear_wheels"]
        ):
            pipeline.seal_until(SEAL_AT)
        # Persist while later messages are still hidden, then keep ingesting.
        if index == len(messages) - 25:
            saved_checkpoint = pipeline.save_checkpoint()
            boundary_at_checkpoint = pipeline.visible_data_boundary()

    comparison = pipeline.stage_sensor_delay("rear_wheels", 0.45)
    delay_comparison = _describe_comparison(comparison)
    discarded_revision = {
        "sensor": "rear_wheels",
        "candidate_delay": 0.45,
        "published": False,
        "message": "候选校正尚未发布即撤销，当前轨迹和已发布延迟保持不变",
    }
    pipeline.discard_staged_delay()

    # The wheel clock drifts gradually (not a constant offset). Stage a full
    # offset-plus-drift revision, inspect the side-by-side explanation, then
    # save while the draft is still unpublished and reopen the same plan.
    clock_comparison = pipeline.stage_clock_revision(
        "rear_wheels", TRUE_DELAYS["rear_wheels"], WHEEL_CLOCK_DRIFT, 0.0
    )
    clock_draft = _describe_clock_comparison(clock_comparison)
    draft_checkpoint = pipeline.save_checkpoint()
    resumed_for_draft = LocalizationPipeline()
    resumed_for_draft.restore_checkpoint(draft_checkpoint)
    draft_after_restore = resumed_for_draft.pending_clock_revision()
    clock_report = resumed_for_draft.publish_staged_clock_revision()
    clock_publish = _describe_clock_report(clock_report)
    # Adopt the resumed pipeline state for the emitted trajectory so the
    # output reflects the published drifting-clock revision.
    pipeline = resumed_for_draft

    rows = []
    for point in truth:
        state = _state_at(pipeline, point.time)
        raw = _state_at(uncorrected, point.time)
        if state is None:
            continue
        rows.append(
            {
                "time": point.time,
                "x": state.x,
                "y": state.y,
                "heading": state.heading,
                "x_error": state.x - point.x,
                "y_error": state.y - point.y,
                "heading_error": _angle_diff(state.heading, point.heading),
                "position_95_radius": state.position_95_radius,
                "heading_95": state.heading_95,
                "degraded_reasons": "|".join(sorted(state.degraded_reasons)),
                "raw_position_error": (
                    "" if raw is None else math.hypot(raw.x - point.x, raw.y - point.y)
                ),
                "sealed": state.time <= SEAL_AT + 1e-9,
            }
        )
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = saved_checkpoint or pipeline.save_checkpoint()
    checkpoint_store = CheckpointStore(args.output_dir / "checkpoint.json")
    checkpoint_store.save(
        checkpoint.state,
        checkpoint.visible_observations,
        calibration=pipeline.calibration.as_dict(),
        arrival_horizon=checkpoint.arrival_horizon,
        seal_time=pipeline.seal_time,
        sealed_outputs=checkpoint.sealed_outputs,
        rejected_observations=checkpoint.rejected_observations,
        clock_models=pipeline.calibration.clock_models(),
        staged_plan=checkpoint.staged_plan,
    )

    # Restore must present exactly the same visible data boundary: a message
    # received after the checkpoint stays hidden until its arrival time.
    loaded_state, loaded_observations, metadata = checkpoint_store.load_with_metadata()
    loaded_checkpoint = Checkpoint(
        state=loaded_state,
        event_count=len(metadata["sealed_outputs"]),
        arrival_horizon=metadata["arrival_horizon"],
        seal_time=metadata["seal_time"],
        calibration=_calibration_from_metadata(metadata),
        visible_observations=loaded_observations,
        sealed_outputs=metadata["sealed_outputs"],
        rejected_observations=metadata["rejected_observations"],
        staged_plan=_staged_plan_from_metadata(
            metadata.get("staged_clock_plan")
        ),
    )
    restored = LocalizationPipeline()
    restored.restore_checkpoint(loaded_checkpoint)
    boundary_after = restored.visible_data_boundary()

    metrics = _metrics(rows, pipeline)
    raw_rows = [row for row in rows if row["raw_position_error"] != ""]
    raw_rmse = math.sqrt(
        sum(float(row["raw_position_error"]) ** 2 for row in raw_rows) / len(raw_rows)
    )
    metrics["rmse_position_uncorrected"] = raw_rmse
    metrics["delay_recompute"] = recompute_events
    metrics["delay_setting_comparison"] = delay_comparison
    metrics["discarded_unpublished_revision"] = discarded_revision
    metrics["clock_revision_comparison"] = clock_draft
    metrics["clock_revision_published"] = clock_publish
    metrics["clock_draft_resumed_after_reopen"] = {
        "pending_before_publish": list(draft_after_restore) if draft_after_restore else None,
        "continued_same_plan": draft_after_restore
        == (
            "rear_wheels",
            TRUE_DELAYS["rear_wheels"],
            WHEEL_CLOCK_DRIFT,
            0.0,
        ),
    }
    metrics["visible_boundary_before_restore"] = boundary_at_checkpoint
    metrics["visible_boundary_after_restore"] = boundary_after
    metrics["restore_preserves_boundary"] = boundary_at_checkpoint == boundary_after
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _state_at(pipeline: LocalizationPipeline, time: float):
    eligible = [
        state for state in pipeline.all_outputs() if abs(state.time - time) < 0.06
    ]
    return eligible[-1] if eligible else None


def _calibration_from_metadata(metadata: dict) -> DelayCalibration:
    models = metadata.get("clock_models") or []
    if not models:
        return DelayCalibration(metadata.get("fixed_delays", {}))
    calibration = DelayCalibration()
    for item in models:
        calibration = calibration.with_clock(
            item["sensor"],
            float(item.get("delay", 0.0)),
            float(item.get("drift", 0.0)),
            float(item.get("epoch", 0.0)),
        )
    return calibration


def _staged_plan_from_metadata(payload) -> tuple[str, float, float, float] | None:
    if not payload:
        return None
    return (
        payload["sensor"],
        float(payload["delay"]),
        float(payload["drift"]),
        float(payload.get("epoch", 0.0)),
    )


def _metrics(rows, pipeline: LocalizationPipeline) -> dict:
    rmse_corrected = math.sqrt(
        sum(row["x_error"] ** 2 + row["y_error"] ** 2 for row in rows) / len(rows)
    )
    within_95 = sum(
        1
        for row in rows
        if math.hypot(row["x_error"], row["y_error"]) <= row["position_95_radius"]
    )
    return {
        "messages": len(pipeline.received),
        "rmse_position_corrected": rmse_corrected,
        "final_abs_heading_error": abs(rows[-1]["heading_error"]),
        "final_position_95_radius": rows[-1]["position_95_radius"],
        "position_95_coverage": within_95 / len(rows),
        "degraded_samples": sum(1 for row in rows if row["degraded_reasons"]),
        "sealed_output_count": len(pipeline.sealed_outputs),
        "fixed_delays": pipeline.calibration.as_dict(),
    }


def _describe_report(report: RecomputeReport) -> dict[str, object]:
    return {
        "sensor": report.sensor,
        "old_delay": report.old_delay,
        "new_delay": report.new_delay,
        "affected_interval": [report.affected_from, report.affected_to],
        "recomputed_from": report.recomputed_from,
        "sealed_blocked_times": list(report.sealed_blocked),
        "boundary_state_time": report.boundary_state_time,
        "adopted_observations": list(report.adopted_observations),
        "rejected_observations": [
            _describe_decision(item) for item in report.rejected_observations
        ],
    }


def _describe_comparison(comparison: DelayComparison) -> dict[str, object]:
    return {
        "sensor": comparison.sensor,
        "recomputable_range": list(comparison.recomputable_range or []),
        "sealed_history": list(comparison.sealed_history or []),
        "boundary_state_time": comparison.boundary_state_time,
        "settings": {
            "baseline": _describe_setting(comparison.baseline),
            "candidate": _describe_setting(comparison.candidate),
        },
        "observations": [
            _describe_decision(item) for item in comparison.decisions
        ],
    }


def _describe_setting(setting) -> dict[str, object]:
    return {
        "delay": setting.delay,
        "corrected_interval": list(setting.corrected_interval or []),
        "affected_observations": list(setting.affected_sequences),
        "open_observations": list(setting.open_sequences),
        "sealed_observations": list(setting.sealed_sequences),
        "rejected_observations": list(setting.rejected_sequences),
    }


def _describe_decision(decision) -> dict[str, object]:
    return {
        "sensor": decision.sensor,
        "sequence": decision.sequence,
        "stamp": decision.stamp,
        "arrival_time": decision.arrival_time,
        "candidate_corrected_time": decision.corrected_time,
        "classification": decision.classification,
        "reason": decision.reason,
    }


def _describe_clock_setting(setting) -> dict[str, object]:
    return {
        "delay": setting.delay,
        "drift": setting.drift,
        "epoch": setting.epoch,
        "corrected_interval": list(setting.corrected_interval or []),
        "affected_observations": list(setting.affected_sequences),
        "open_observations": list(setting.open_sequences),
        "sealed_observations": list(setting.sealed_sequences),
        "rejected_observations": list(setting.rejected_sequences),
    }


def _describe_position_change(change) -> dict[str, object]:
    return {
        "sensor": change.sensor,
        "sequence": change.sequence,
        "baseline_corrected_time": change.baseline_time,
        "candidate_corrected_time": change.candidate_time,
        "time_shift": change.time_shift,
        "position_delta": change.position_delta,
        "simultaneous_with": [list(item) for item in change.simultaneous_with],
        "ordering_swapped_with": [list(item) for item in change.ordering_swapped_with],
        "cause": change.cause,
    }


def _describe_uncertain_interval(interval) -> dict[str, object]:
    return {
        "from_time": interval.from_time,
        "to_time": interval.to_time,
        "gap": interval.gap,
        "reason": interval.reason,
    }


def _describe_clock_comparison(comparison: ClockComparison) -> dict[str, object]:
    return {
        "sensor": comparison.sensor,
        "recomputable_range": list(comparison.recomputable_range or []),
        "sealed_history": list(comparison.sealed_history or []),
        "boundary_state_time": comparison.boundary_state_time,
        "settings": {
            "baseline": _describe_clock_setting(comparison.baseline),
            "candidate": _describe_clock_setting(comparison.candidate),
        },
        "observations": [
            _describe_decision(item) for item in comparison.decisions
        ],
        "position_changes": [
            _describe_position_change(item) for item in comparison.position_changes
        ],
        "uncertain_intervals": [
            _describe_uncertain_interval(item) for item in comparison.uncertain_intervals
        ],
        "simultaneous_groups": [
            [list(identity) for identity in group]
            for group in comparison.simultaneous_groups
        ],
        "applied_identities": [list(item) for item in comparison.applied_identities],
    }


def _describe_clock_report(report: ClockRevisionReport) -> dict[str, object]:
    return {
        "sensor": report.sensor,
        "old_delay": report.old_delay,
        "new_delay": report.new_delay,
        "old_drift": report.old_drift,
        "new_drift": report.new_drift,
        "epoch": report.epoch,
        "boundary_state_time": report.boundary_state_time,
        "adopted_observations": list(report.adopted_observations),
        "rejected_observations": [
            _describe_decision(item) for item in report.rejected_observations
        ],
        "position_changes": [
            _describe_position_change(item) for item in report.position_changes
        ],
        "uncertain_intervals": [
            _describe_uncertain_interval(item) for item in report.uncertain_intervals
        ],
        "simultaneous_groups": [
            [list(identity) for identity in group]
            for group in report.simultaneous_groups
        ],
        "applied_identities": [list(item) for item in report.applied_identities],
        "identity_count": len(report.applied_identities),
        "identity_unique": len(set(report.applied_identities))
        == len(report.applied_identities),
    }


def _angle_diff(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    raise SystemExit(main())
