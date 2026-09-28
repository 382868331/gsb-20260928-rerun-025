import numpy as np
from dataclasses import replace

from robotloc.estimator import EkfLocalizer
from robotloc.pipeline import LocalizationPipeline, PipelineConfig
from robotloc.scenario import ScenarioConfig, make_ground_truth, make_messages
from robotloc.sources import InMemorySource, ReplaySource
from robotloc.recording import (
    JsonlRecorder, estimate_to_dict, message_to_dict, read_messages,
)
from robotloc.evaluate import evaluate


def test_end_to_end_accuracy_on_known_path(tmp_path):
    gt = make_ground_truth()
    msgs = make_messages(gt)
    pipe = LocalizationPipeline()
    result = pipe.run(InMemorySource(msgs))
    m = evaluate(result.estimates, gt)
    assert m.rms_xy < 0.35
    assert m.max_xy < 1.0
    assert m.rms_theta < np.radians(8)
    # 存在断流窗口且迟到消息被正确处理
    assert result.late_accepted >= 1
    assert len(result.gap_windows) >= 1


def test_no_future_observation_used():
    """在迟到消息到达之前，输出不得包含它的信息。"""
    from robotloc.types import ImuObservation

    msgs = [
        ImuObservation(t=0.1, received_at=0.1, yaw_rate=0.0, seq=0),
        ImuObservation(t=0.2, received_at=5.0, yaw_rate=2.0, seq=1),  # 严重迟到
        ImuObservation(t=5.0, received_at=5.0, yaw_rate=0.0, seq=2),
    ]
    pipe = LocalizationPipeline(PipelineConfig(max_late_seconds=100.0))
    result = pipe.run(InMemorySource(msgs))
    # t=0.1 时刻输出是在 0.1 到达时产生的，此时 seq=1 尚未到达
    early = [e for e in result.estimates if abs(e.t - 0.1) < 1e-9]
    assert early and early[0].fresh


def test_gap_outputs_marked_stale_and_grow():
    gt = make_ground_truth()
    cfg = ScenarioConfig()
    msgs = make_messages(gt, cfg)
    pipe = LocalizationPipeline()
    result = pipe.run(InMemorySource(msgs))
    stale = [e for e in result.estimates if not e.fresh]
    assert stale, "断流期间应有 fresh=False 的推演输出"
    # 断流窗口内协方差应单调增长
    gap = [e for e in stale if cfg.dropout_start < e.t < cfg.dropout_end + 0.3]
    sigmas = [e.position_sigma for e in gap]
    assert sigmas[-1] > sigmas[0]


def test_late_message_rollback_matches_ontime():
    """迟到消息走回滚重放后，与该消息准时到达时结果一致。"""
    gt = make_ground_truth()
    cfg = ScenarioConfig()
    msgs_late = make_messages(gt, cfg)
    msgs_ontime = [replace(m, received_at=m.t) for m in msgs_late]
    pipe_a = LocalizationPipeline()
    res_a = pipe_a.run(InMemorySource(msgs_ontime))
    pipe_b = LocalizationPipeline()
    res_b = pipe_b.run(InMemorySource(msgs_late))
    ta = np.array([e.t for e in res_a.estimates])
    tb = np.array([e.t for e in res_b.estimates])
    assert np.allclose(ta, tb, atol=1e-9)
    pa = np.array([e.as_array() for e in res_a.estimates])
    pb = np.array([e.as_array() for e in res_b.estimates])
    assert np.allclose(pa, pb, atol=1e-9), np.abs(pa - pb).max()


def test_checkpoint_save_resume_equals_full_replay(tmp_path):
    gt = make_ground_truth()
    msgs = make_messages(gt)
    split = 10.0
    early = [m for m in msgs if m.received_at <= split + 1e-12]
    later = [m for m in msgs if m.received_at > split + 1e-12]

    p1 = LocalizationPipeline()
    p1.run(InMemorySource(early))
    ckpt = tmp_path / "ckpt.json"
    p1.save(ckpt)

    p2 = LocalizationPipeline()
    p2.load(ckpt)
    later = p2.filter_unprocessed(later)
    resumed = p2.run(InMemorySource(later))

    full = LocalizationPipeline().run(InMemorySource(msgs))
    full_by_t = {round(e.t, 9): e for e in full.estimates}
    diffs = [
        np.abs(e.as_array() - full_by_t[round(e.t, 9)].as_array())
        for e in resumed.estimates
        if round(e.t, 9) in full_by_t
    ]
    assert np.max(diffs) < 1e-10


def test_file_replay_matches_in_memory(tmp_path):
    gt = make_ground_truth()
    msgs = make_messages(gt)
    path = tmp_path / "log.jsonl"
    rec = JsonlRecorder(path)
    rec.write_many(message_to_dict(m) for m in msgs)

    r1 = LocalizationPipeline().run(InMemorySource(msgs))
    r2 = LocalizationPipeline().run(
        ReplaySource(read_messages(path))
    )
    a = np.array([e.as_array() for e in r1.estimates])
    b = np.array([e.as_array() for e in r2.estimates])
    np.testing.assert_array_equal(a, b)
