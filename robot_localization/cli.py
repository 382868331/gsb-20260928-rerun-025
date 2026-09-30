from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from dataclasses import asdict

from .evaluation import compare_with_truth
from .pipeline import LocalizationPipeline
from .recording import CheckpointStore, JsonlRecorder
from .sources import SyntheticScenarioSource, TimedDelayedScenarioSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline robot localization sample")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--timed-calibration", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.timed_calibration:
        return _run_timed_calibration(args.output_dir)

    source = SyntheticScenarioSource()
    messages = source.observations()
    JsonlRecorder(args.output_dir / "messages.jsonl").write_many(messages)

    pipeline = LocalizationPipeline()
    horizon = 0.0
    for arrival_index, observation in enumerate(messages):
        pipeline.add_observation(observation)
        if observation.time <= 9.8:
            horizon = max(horizon, observation.time)
            pipeline.process_until(horizon)
        if arrival_index == 60:
            checkpoint = pipeline.save_checkpoint()
            CheckpointStore(args.output_dir / "checkpoint.json").save(
                checkpoint.state,
                pipeline.checkpoint_observations(),
            )
    pipeline.process_until(10.0)

    truth = source.ground_truth()
    rows = []
    for point in truth:
        state = _state_at(pipeline, point.time)
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
            }
        )
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rmse = math.sqrt(sum(row["x_error"] ** 2 + row["y_error"] ** 2 for row in rows) / len(rows))
    final_heading_error = abs(rows[-1]["heading_error"])
    metrics = {
        "messages": len(messages),
        "rmse_position": rmse,
        "final_abs_heading_error": final_heading_error,
        "final_position_95_radius": rows[-1]["position_95_radius"],
        "degraded_samples": sum(1 for row in rows if row["degraded_reasons"]),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def _run_timed_calibration(output_dir: Path) -> int:
    source = TimedDelayedScenarioSource()
    messages = source.observations()
    JsonlRecorder(output_dir / "messages.jsonl").write_many(messages)

    def run(delay: float, checkpoint_path: Path | None = None):
        pipeline = LocalizationPipeline(sensor_delays={"rear_wheels": delay})
        for index, observation in enumerate(messages):
            pipeline.add_observation(observation)
            arrival = (
                observation.arrival_time
                if observation.arrival_time is not None
                else observation.time
            )
            if arrival <= 10.0:
                pipeline.process_until(arrival)
            if checkpoint_path is not None and abs(arrival - 4.1) < 0.06:
                checkpoint = pipeline.save_checkpoint()
                CheckpointStore(checkpoint_path).save(
                    checkpoint.state,
                    pipeline.checkpoint_observations(),
                    {
                        "horizon": pipeline.horizon,
                        "visible_arrival": pipeline.visible_arrival,
                        "sealed_horizon": pipeline.sealed_horizon,
                        "sensor_delays": pipeline.sensor_delays,
                    },
                )
        return pipeline

    uncorrected = run(0.0)
    corrected = run(0.5, output_dir / "checkpoint.json")
    truth = source.ground_truth()
    uncorrected_metrics = compare_with_truth(uncorrected.outputs, truth)
    corrected_metrics = compare_with_truth(corrected.outputs, truth)
    metrics = {
        "uncorrected": asdict(uncorrected_metrics),
        "corrected": asdict(corrected_metrics),
        "revision_records": [asdict(item) for item in corrected.revision_records],
        "sealed_outputs": len(corrected.sealed_outputs),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def _state_at(pipeline: LocalizationPipeline, time: float):
    eligible = [state for state in pipeline.outputs if abs(state.time - time) < 0.006]
    return eligible[-1] if eligible else None


def _angle_diff(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


if __name__ == "__main__":
    raise SystemExit(main())
