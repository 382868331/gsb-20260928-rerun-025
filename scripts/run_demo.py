"""离线回放演示：构造样例 -> 融合 -> 评估 -> 检查点恢复对比 -> 落盘。

用法: python -m scripts.run_demo [--out runs/demo]
不读取真实设备。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from robotloc.evaluate import evaluate
from robotloc.pipeline import LocalizationPipeline
from robotloc.recording import (
    JsonlRecorder,
    estimate_to_dict,
    message_to_dict,
    read_messages,
)
from robotloc.scenario import GroundTruth, ScenarioConfig, make_ground_truth, make_messages
from robotloc.sources import InMemorySource, ReplaySource


def run_full(
    messages, initial_pose=(0.0, 0.0, 0.0)
) -> tuple[LocalizationPipeline, object]:
    pipe = LocalizationPipeline(initial_pose=initial_pose)
    result = pipe.run(InMemorySource(messages))
    return pipe, result


def run_resume_from_checkpoint(messages, split_arrival: float) -> object:
    """在 split_arrival 处保存检查点，恢复后只投递之后到达的消息。"""
    early = [m for m in messages if m.received_at <= split_arrival + 1e-12]
    pipe1 = LocalizationPipeline()
    pipe1.run(InMemorySource(early))
    ckpt = Path("/tmp/robotloc_ckpt_demo.json")
    pipe1.save(ckpt)

    pipe2 = LocalizationPipeline()
    pipe2.load(ckpt)
    remaining = pipe2.filter_unprocessed(
        [m for m in messages if m.received_at > split_arrival + 1e-12]
    )
    return pipe2.run(InMemorySource(remaining)), pipe2


def write_trajectory_csv(path: Path, estimates) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["t", "x", "y", "theta", "pos_1sigma_m", "heading_1sigma_deg", "fresh"]
        )
        for e in estimates:
            writer.writerow([
                f"{e.t:.4f}", f"{e.x:.5f}", f"{e.y:.5f}", f"{e.theta:.6f}",
                f"{e.position_sigma:.5f}",
                f"{np.degrees(e.heading_sigma):.4f}", int(e.fresh),
            ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = make_ground_truth()
    cfg = ScenarioConfig()
    messages = make_messages(gt, cfg)

    # 1) 记录原始消息（测量时刻 vs 到达时刻都保留）
    rec = JsonlRecorder(out / "messages.jsonl")
    rec.write_many(message_to_dict(m) for m in messages)

    # 2) 一次性完整回放
    pipe_full, result_full = run_full(messages)
    metrics = evaluate(result_full.estimates, gt)
    pose_rec = JsonlRecorder(out / "poses_full.jsonl")
    pose_rec.write_many(estimate_to_dict(e) for e in result_full.estimates)
    write_trajectory_csv(out / "trajectory.csv", result_full.estimates)

    # 3) 从记录文件回放（走 ReplaySource，不直接用内存对象）
    pipe_replay, result_replay = run_full(
        list(ReplaySource(read_messages(out / "messages.jsonl")).messages())
    )

    # 4) 检查点保存/恢复：在 t=10s 处切开
    (resume_result, _pipe2) = run_resume_from_checkpoint(messages, 10.0)

    max_diff = max(
        np.abs(
            np.array([a.as_array() for a in result_full.estimates])
            - np.array([b.as_array() for b in result_replay.estimates])
        ).max(),
        0.0,
    )

    # 恢复后的估计与完整回放逐点对比（恢复段起点之后）
    full_by_t = {round(e.t, 9): e for e in result_full.estimates}
    diffs = []
    for e in resume_result.estimates:
        f = full_by_t.get(round(e.t, 9))
        if f is not None:
            diffs.append(np.abs(e.as_array() - f.as_array()))
    resume_max_diff = float(np.max(diffs)) if diffs else float("nan")

    summary = {
        "n_messages": len(messages),
        "n_estimates": len(result_full.estimates),
        "metrics": metrics.as_dict(),
        "late_accepted": result_full.late_accepted,
        "late_dropped": result_full.late_dropped,
        "gap_windows": result_full.gap_windows,
        "replay_max_pose_diff": max_diff,
        "resume_max_pose_diff": resume_max_diff,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
