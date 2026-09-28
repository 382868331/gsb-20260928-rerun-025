"""传感器接入层：把“设备”与估计核心隔开。

本轮只用离线/虚构数据源（不读取真实设备）。真实设备接入时实现
:class:`SensorSource` 协议即可，pipeline 与 estimator 无需改动。
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol

from .types import ImuObservation, WheelObservation


class SensorSource(Protocol):
    """传感器源协议：按到达顺序产出消息（可乱序、可缺失）。"""

    def messages(self) -> Iterable[WheelObservation | ImuObservation]: ...


class ReplaySource:
    """记录回放源：严格按消息的“到达时刻”逐条吐消息。

    与实时源的差别仅在于它来自文件；它同样不会提前释放尚未到达的
    消息，因此回放结果与在线处理一致。
    """

    def __init__(self, records: Iterable[dict]):
        self._records = list(records)

    def messages(self) -> Iterator[WheelObservation | ImuObservation]:
        ordered = sorted(
            self._records, key=lambda r: (r["received_at"], r.get("seq", 0))
        )
        for rec in ordered:
            yield _record_to_message(rec)


class InMemorySource:
    """内存消息源（仿真器直接灌入消息），按到达顺序输出。"""

    def __init__(self, messages: Iterable[WheelObservation | ImuObservation]):
        self._messages = sorted(
            messages, key=lambda m: (m.received_at, m.seq)
        )

    def messages(self) -> list[WheelObservation | ImuObservation]:
        return list(self._messages)


def _record_to_message(rec: dict) -> WheelObservation | ImuObservation:
    if rec["kind"] == "wheel":
        return WheelObservation(
            t=rec["t"],
            received_at=rec["received_at"],
            left_counts=rec["left_counts"],
            right_counts=rec["right_counts"],
            dt_counts=rec["dt_counts"],
            wheel_radius=rec.get("wheel_radius", 0.1),
            wheel_base=rec.get("wheel_base", 0.5),
            counts_per_revolution=rec.get("counts_per_revolution", 1024.0),
            seq=rec.get("seq", 0),
        )
    if rec["kind"] == "imu":
        return ImuObservation(
            t=rec["t"],
            received_at=rec["received_at"],
            yaw_rate=rec["yaw_rate"],
            yaw=rec.get("yaw"),
            seq=rec.get("seq", 0),
        )
    raise ValueError(f"未知消息类型: {rec.get('kind')!r}")
