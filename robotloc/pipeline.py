"""因果处理管线。

设计约束（对应需求）：

1. **不提前使用未来观测**：处理按消息“到达时刻” ``received_at`` 推进，
   只有 ``received_at <= now`` 且测量时刻 ``t <= now`` 的消息才送入 EKF。
2. **迟到消息**：测量时刻早于已提交水位时，回滚到其之前最近的检查点，
   把迟到消息与该区间原消息一起按时间序重放。重放是纯函数式重算，
   所以结果与“该消息一开始就准时到达”完全一致。
3. **断流不谎称精度**：到达间隔超过 ``gap_timeout`` 时，以
   ``dead_reckon_dt`` 步长做纯推演，输出 ``fresh=False``；EKF 过程噪声
   使协方差持续增长。观测恢复时产生 ``fresh=True`` 的更新。
4. **检查点恢复**：检查点 = 估计器全部状态（JSON 序列化）。保存后恢复
   继续处理，与一次性完整回放逐位一致（由测试保证）。
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from .estimator import EkfConfig, EkfLocalizer
from .sources import SensorSource
from .types import ImuObservation, PoseEstimate, WheelObservation

Message = WheelObservation | ImuObservation


@dataclass
class PipelineConfig:
    checkpoint_dt: float = 0.5          # 检查点间隔（数据时间，秒）
    gap_timeout: float = 0.35           # 超过此时长无消息 -> 断流
    dead_reckon_dt: float = 0.1         # 断流期纯推演步长
    max_late_seconds: float = 30.0      # 超出覆盖范围的迟到消息被放弃


@dataclass
class RunResult:
    estimates: list[PoseEstimate] = field(default_factory=list)
    late_accepted: int = 0
    late_dropped: int = 0
    gap_windows: list[tuple[float, float]] = field(default_factory=list)


class LocalizationPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        ekf_config: EkfConfig | None = None,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.config = config or PipelineConfig()
        self._initial_pose = initial_pose
        self._ekf_config = ekf_config or EkfConfig()
        self.ekf = EkfLocalizer(
            config=self._ekf_config, initial_pose=initial_pose
        )
        # 检查点: (提交水位, EKF 快照, 已处理消息数)
        self._checkpoints: list[tuple[float, dict, int]] = []
        self._processed: list[Message] = []
        self._committed_t: float | None = None
        self._last_checkpoint_t: float | None = None
        # 时刻 -> 输出（dict 保证重放时覆盖旧值，最终按 t 排序）
        self._outputs: dict[float, PoseEstimate] = {}
        self.late_accepted = 0
        self.late_dropped = 0
        self.gap_windows: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    def run(self, source: SensorSource) -> RunResult:
        messages = sorted(source.messages(), key=lambda m: (m.received_at, m.seq))
        if not messages:
            return RunResult()
        arrivals = sorted({m.received_at for m in messages})
        horizon_end = max(m.t for m in messages)
        prev_arrival = arrivals[0]

        for now in arrivals:
            gap_ticks = self._gap_ticks(prev_arrival, now)
            if gap_ticks:
                self.gap_windows.append((prev_arrival, now))
            due, late = self._partition(messages, now)
            for msg in late:
                if now - msg.t > self.config.max_late_seconds:
                    self.late_dropped += 1
                    continue
                self._rollback_and_replay(msg, messages, now, gap_ticks)
                self.late_accepted += 1
            if due or (self._committed_t is not None and gap_ticks):
                self._replay_events(due, gap_ticks, mark_processed=True)

            self._committed_t = now
            self._maybe_checkpoint(now)
            prev_arrival = now

        # 数据覆盖范围末端：若最后一个到达时刻之后仍有覆盖，补纯推演
        if self._committed_t is not None:
            tail = self._gap_ticks(prev_arrival, horizon_end)
            if tail:
                self._replay_events([], tail, mark_processed=False)
        return RunResult(
            estimates=[self._outputs[k] for k in sorted(self._outputs)],
            late_accepted=self.late_accepted,
            late_dropped=self.late_dropped,
            gap_windows=list(self.gap_windows),
        )

    # ------------------------------------------------------------------
    def _partition(
        self, messages: list[Message], now: float
    ) -> tuple[list[Message], list[Message]]:
        """把本时刻到达且尚未处理的消息分成 准时 / 迟到 两组。"""
        due: list[Message] = []
        late: list[Message] = []
        done = {(type(m).__name__, m.seq, round(m.t, 12)) for m in self._processed}
        for m in messages:
            if m.received_at > now:
                break
            key = (type(m).__name__, m.seq, round(m.t, 12))
            if key in done:
                continue
            if self._committed_t is not None and m.t < self._committed_t - 1e-12:
                late.append(m)
            elif m.t <= now + 1e-12:
                due.append(m)
        due.sort(key=lambda m: (m.t, m.seq))
        late.sort(key=lambda m: (m.t, m.seq))
        return due, late

    def _gap_ticks(self, a: float, b: float) -> list[float]:
        """断流区间 (a, b] 内对齐网格的纯推演时刻；非断流返回空。"""
        if b - a <= self.config.gap_timeout:
            return []
        step = self.config.dead_reckon_dt
        ticks: list[float] = []
        t = a
        while t + step <= b + 1e-12:
            t += step
            ticks.append(round(t, 12))
        return ticks

    def _replay_events(
        self,
        due: list[Message],
        ticks: list[float],
        *,
        mark_processed: bool,
    ) -> None:
        """按时间序混合处理消息与断流推演刻度；同时刻消息优先（记 fresh）。"""
        events: list[tuple[float, int, object]] = [
            (round(m.t, 12), 0, m) for m in due
        ] + [(round(t, 12), 1, t) for t in ticks]
        # 同一消息可能与刻度同刻：消息在前；同刻多条消息按 seq
        events.sort(key=lambda e: (e[0], e[1], getattr(e[2], "seq", 0)))
        for _, kind, payload in events:
            if kind == 0:
                self._apply_and_emit(payload)
                if mark_processed:
                    self._processed.append(payload)
            else:
                key = round(payload, 12)
                if key not in self._outputs:
                    self._emit_dead_reckon(payload)

    def _rollback_and_replay(
        self,
        late_msg: Message,
        messages: list[Message],
        now: float,
        gap_ticks: list[float],
    ) -> None:
        # 选择回滚锚点：测量时刻 <= 迟到消息时刻的最近检查点
        anchor_idx = bisect_right(
            [cp[0] for cp in self._checkpoints], late_msg.t + 1e-12
        ) - 1
        if anchor_idx >= 0:
            cp_t, snap, n_done = self._checkpoints[anchor_idx]
            self.ekf.restore(snap)
            kept = self._processed[:n_done]
            replay_old = self._processed[n_done:]
            rollback_to = cp_t
        else:
            self.ekf = EkfLocalizer(
                config=self._ekf_config,
                initial_pose=self._initial_pose,
                initial_heading_std=self.ekf.initial_heading_std,
                initial_velocity_std=self.ekf.initial_velocity_std,
            )
            kept = []
            replay_old = list(self._processed)
            rollback_to = -1e300

        # 重放区间：所有 received_at <= now、测量时刻落在 (anchor, now] 的消息
        cutoff = now + 1e-12
        replay_new = [
            m for m in messages
            if m.received_at <= cutoff and rollback_to < m.t <= cutoff
        ]
        merged = sorted(
            replay_old + replay_new + [late_msg], key=lambda m: (m.t, m.seq)
        )
        # 去重（late_msg 可能本来就在 replay_new 集合里不会发生，
        # 但 replay_old 与 replay_new 之间可能重叠）
        uniq: dict[tuple, Message] = {}
        for m in merged:
            uniq[(type(m).__name__, m.seq, round(m.t, 12))] = m

        self._processed = kept
        replay_msgs = [
            v for _, v in sorted(
                ((mm.t, mm) for mm in uniq.values()),
                key=lambda kv: (kv[0], kv[1].seq),
            )
        ]
        # 丢弃锚点之后的输出（观测更新与断流推演都重算）
        for t in [t for t in self._outputs if t > rollback_to + 1e-12]:
            del self._outputs[t]
        # 只重放锚点之后的断流刻度
        ticks = [t for t in gap_ticks if t > rollback_to + 1e-12]
        self._replay_events(replay_msgs, ticks, mark_processed=True)
        self._committed_t = now

    def _emit_dead_reckon(self, t: float) -> None:
        est = self.ekf.estimate(t, fresh=False)
        self._outputs[round(t, 12)] = est

    def _apply_and_emit(self, msg: Message) -> None:
        if isinstance(msg, WheelObservation):
            self.ekf.update_wheel(msg)
        elif isinstance(msg, ImuObservation):
            self.ekf.update_imu(msg)
        else:
            raise TypeError(f"未知消息类型: {type(msg)!r}")
        est = self.ekf.estimate(msg.t, fresh=True)
        self._outputs[round(msg.t, 12)] = est

    # ------------------------------------------------------------------
    def _maybe_checkpoint(self, now: float) -> None:
        if (
            self._last_checkpoint_t is None
            or now - self._last_checkpoint_t >= self.config.checkpoint_dt - 1e-12
        ):
            self._checkpoints.append(
                (now, self.ekf.checkpoint(), len(self._processed))
            )
            self._last_checkpoint_t = now
            cutoff = now - self.config.max_late_seconds
            kept = [cp for cp in self._checkpoints if cp[0] >= cutoff]
            self._checkpoints = kept or self._checkpoints[-1:]

    # ------------------------------------------------------------------
    # 显式保存/恢复（“保存检查点后恢复计算”）
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "ekf": self.ekf.checkpoint(),
            "committed_t": self._committed_t,
            "last_checkpoint_t": self._last_checkpoint_t,
            "processed_keys": [
                [type(m).__name__, m.seq, m.t] for m in self._processed
            ],
            "checkpoints": [
                [cp_t, snap, n] for cp_t, snap, n in self._checkpoints
            ],
            "counters": {
                "late_accepted": self.late_accepted,
                "late_dropped": self.late_dropped,
            },
            "gap_windows": self.gap_windows,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ekf.restore(payload["ekf"])
        self._committed_t = payload["committed_t"]
        self._last_checkpoint_t = payload["last_checkpoint_t"]
        self._checkpoints = [
            (float(cp_t), snap, int(n))
            for cp_t, snap, n in payload["checkpoints"]
        ]
        self.late_accepted = payload["counters"]["late_accepted"]
        self.late_dropped = payload["counters"]["late_dropped"]
        self.gap_windows = [tuple(w) for w in payload["gap_windows"]]
        # 已处理消息以 (类型名, seq, t) 记录；恢复后由消息源避免重复投递
        self._processed = []
        self._processed_keys = {
            (k[0], int(k[1]), float(k[2]))
            for k in payload["processed_keys"]
        }

    def filter_unprocessed(self, messages: list[Message]) -> list[Message]:
        """恢复后使用：剔除保存点之前已处理过的消息。"""
        keys = getattr(self, "_processed_keys", None)
        if keys is None:
            return list(messages)
        return [
            m for m in messages
            if (type(m).__name__, m.seq, float(m.t)) not in keys
        ]
