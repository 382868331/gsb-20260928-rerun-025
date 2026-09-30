from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .calibration import DelayCalibration
from .pipeline import LocalizationPipeline, RecomputeReport, SealedHistoryError
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
    sealed_attempt: dict[str, object] | None = None
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

    # The user tries once more with a delay that touches the sealed interval,
    # to demonstrate that sealed outputs cannot be silently replaced.
    try:
        pipeline.set_sensor_delay("rear_wheels", 0.45)
    except SealedHistoryError as error:
        sealed_attempt = _describe_sealed_error(error)

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
    )

    # Restore must present exactly the same visible data boundary: a message
    # received after the checkpoint stays hidden until its arrival time.
    restored = LocalizationPipeline()
    restored.restore_checkpoint(checkpoint)
    boundary_after = restored.visible_data_boundary()

    metrics = _metrics(rows, pipeline)
    raw_rows = [row for row in rows if row["raw_position_error"] != ""]
    raw_rmse = math.sqrt(
        sum(float(row["raw_position_error"]) ** 2 for row in raw_rows) / len(raw_rows)
    )
    metrics["rmse_position_uncorrected"] = raw_rmse
    metrics["delay_recompute"] = recompute_events
    metrics["sealed_rejected_update"] = sealed_attempt
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
    }


def _describe_sealed_error(error: SealedHistoryError) -> dict[str, object]:
    return {
        "sensor": error.sensor,
        "attempted_interval": list(error.affected_interval),
        "sealed_times": error.sealed_times,
        "message": str(error),
    }


def _angle_diff(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    raise SystemExit(main())
