"""记录与回放：消息和位姿输出统一落 JSONL。

- 消息记录保存测量时刻 ``t`` 与到达时刻 ``received_at``，两者分离，
  所以迟到 / 断流场景可以原样重放；
- 输出记录带 ``fresh`` 标志，断流期间的纯推演结果可与观测更新区分；
- numpy 数组转成嵌套 list，文件不依赖 pickle。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import ImuObservation, PoseEstimate, WheelObservation


def message_to_dict(msg: WheelObservation | ImuObservation) -> dict:
    if isinstance(msg, WheelObservation):
        return {
            "kind": "wheel",
            "t": msg.t,
            "received_at": msg.received_at,
            "left_counts": msg.left_counts,
            "right_counts": msg.right_counts,
            "dt_counts": msg.dt_counts,
            "wheel_radius": msg.wheel_radius,
            "wheel_base": msg.wheel_base,
            "counts_per_revolution": msg.counts_per_revolution,
            "seq": msg.seq,
        }
    return {
        "kind": "imu",
        "t": msg.t,
        "received_at": msg.received_at,
        "yaw_rate": msg.yaw_rate,
        "yaw": msg.yaw,
        "seq": msg.seq,
    }


def estimate_to_dict(est: PoseEstimate) -> dict:
    return {
        "kind": "pose",
        "t": est.t,
        "x": est.x,
        "y": est.y,
        "theta": est.theta,
        "covariance": np.asarray(est.covariance, dtype=float).tolist(),
        "fresh": est.fresh,
    }


class JsonlRecorder:
    """追加写 JSONL；同一文件可分段写（检查点后续跑）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def write_many(self, records: Iterable[dict]) -> None:
        for rec in records:
            self.write(rec)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_messages(path: str | Path) -> list[dict]:
    """只读取消息记录（供 ReplaySource 使用）。"""
    return [r for r in read_jsonl(path) if r.get("kind") in ("wheel", "imu")]


def read_poses(path: str | Path) -> list[PoseEstimate]:
    out: list[PoseEstimate] = []
    for r in read_jsonl(path):
        if r.get("kind") != "pose":
            continue
        out.append(
            PoseEstimate(
                t=r["t"],
                x=r["x"],
                y=r["y"],
                theta=r["theta"],
                covariance=np.array(r["covariance"], dtype=float),
                fresh=r["fresh"],
            )
        )
    return out
