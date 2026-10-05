from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .calibration import DelayCalibration
from .pipeline import Checkpoint, DelayComparison, LocalizationPipeline, RecomputeReport
from .recording import CheckpointStore, JsonlRecorder
from .sources import SyntheticScenarioSource


TRUE_DELAYS = {"rear_wheels": 0.30, "imu": 0.12}
INITIAL_GUESS = {"rear_wheels": 0.10, "imu": 0.12}
SEAL_AT = 5.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline robot localization sample")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = SyntheticScenarioSource(fixed_delays=TRUE_DELAYS)
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
    saved_checkpoint = None
    boundary_at_checkpoint = None
    resume_comparison = None
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
        if index == len(messages) - 30 and pipeline.staged_calibration is None:
            resume_comparison = pipeline.stage_sensor_clock(
                "rear_wheels",
                anchors=[(0.0, -TRUE_DELAYS["rear_wheels"]), (10.0, 9.55)],
            )
        if index == len(messages) - 25:
            saved_checkpoint = pipeline.save_checkpoint()
            boundary_at_checkpoint = pipeline.visible_data_boundary()

    comparison = pipeline.stage_sensor_clock(
        "rear_wheels",
        anchors=[(0.0, -TRUE_DELAYS["rear_wheels"]), (10.0, 9.55)],
    )
    delay_comparison = _describe_comparison(comparison)
    discarded_revision = {
        "sensor": "rear_wheels",
        "candidate_clock": pipeline.staged_calibration.clock_for("rear_wheels").as_dict(),
        "published": False,
        "message": "候选校正尚未发布即撤销，当前轨迹和已发布延迟保持不变",
    }
    pipeline.discard_staged_delay()

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
        staged_calibration=(
            checkpoint.staged_calibration.as_dict()
            if checkpoint.staged_calibration is not None
            else None
        ),
        staged_sensor=checkpoint.staged_sensor,
        comparison_view=checkpoint.comparison_view,
        compositions=checkpoint.compositions,
    )

    # Restore must present exactly the same visible data boundary: a message
    # received after the checkpoint stays hidden until its arrival time.
    loaded_state, loaded_observations, metadata = checkpoint_store.load_with_metadata()
    loaded_checkpoint = Checkpoint(
        state=loaded_state,
        event_count=len(metadata["sealed_outputs"]),
        arrival_horizon=metadata["arrival_horizon"],
        seal_time=metadata["seal_time"],
        calibration=DelayCalibration(metadata["clock_models"]),
        visible_observations=loaded_observations,
        sealed_outputs=metadata["sealed_outputs"],
        rejected_observations=metadata["rejected_observations"],
        staged_calibration=(
            DelayCalibration(metadata["staged_calibration"])
            if metadata["staged_calibration"]
            else None
        ),
        staged_sensor=metadata["staged_sensor"],
        comparison_view=metadata["comparison_view"],
        compositions=metadata["compositions"],
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
    metrics["visible_boundary_before_restore"] = boundary_at_checkpoint
    metrics["visible_boundary_after_restore"] = boundary_after
    metrics["restore_preserves_boundary"] = boundary_at_checkpoint == boundary_after
    metrics["resume_staged_calibration"] = (
        _describe_comparison(resume_comparison)
        if resume_comparison is not None
        else None
    )
    metrics["restored_staged_sensor"] = restored.staged_sensor
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
        "clock_models": pipeline.calibration.as_dict(),
        "fixed_delays": {
            sensor: model.delay
            for sensor, model in pipeline.calibration.clock_items()
        },
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
        "timeline": [
            {
                "identity": list(item.identity),
                "stamp": item.stamp,
                "arrival_time": item.arrival_time,
                "baseline_time": item.baseline_time,
                "candidate_time": item.candidate_time,
                "time_shift": item.time_shift,
                "baseline_order": item.baseline_order,
                "candidate_order": item.candidate_order,
                "order_delta": item.order_delta,
                "classification": item.classification,
                "reason": item.reason,
            }
            for item in comparison.timeline
        ],
        "position_changes": [
            {
                "identity": list(item.identity),
                "baseline_time": item.baseline_time,
                "candidate_time": item.candidate_time,
                "baseline_position": [item.baseline_x, item.baseline_y],
                "candidate_position": [item.candidate_x, item.candidate_y],
            }
            for item in comparison.position_changes
        ],
        "uncertain_intervals": [
            {
                "start": item.start,
                "end": item.end,
                "reason": item.reason,
            }
            for item in comparison.uncertain_intervals
        ],
    }


def _describe_setting(setting) -> dict[str, object]:
    return {
        "delay": setting.delay,
        "clock": setting.clock.as_dict() if setting.clock is not None else None,
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
        "baseline_corrected_time": decision.baseline_corrected_time,
        "candidate_corrected_time": decision.corrected_time,
        "time_shift": decision.time_shift,
        "classification": decision.classification,
        "reason": decision.reason,
    }


def _angle_diff(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    raise SystemExit(main())
