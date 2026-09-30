# 离线机器人定位核心

这是不读取真实设备的 Python 离线核心。传感器接入、状态估计和记录回放分别位于 `sources.py`、`estimator.py` 和 `recording.py`，`pipeline.py` 负责按到达时间因果接收、按校正后的观测时间融合。地图匹配暂未加入，后续可作为独立测量源接入。

## 运行

```bash
python -m robot_localization.cli --output-dir outputs
```

输出：

- `messages.jsonl`：按到达顺序记录的虚构轮速与 IMU 消息。
- `trajectory.csv`：位置、朝向、误差、95% 误差半径和降级原因。
- `metrics.json`：已知路径的汇总误差。
- `checkpoint.json`：示例中途检查点。

## 因果与迟到

每条消息同时保留戳记观测时间 `time` 和墙钟到达时间 `arrival_time`。`process_until(t)` 只放行到达时间不晚于 `t` 的消息，融合时再按每路固定延迟把戳记校正为真实观测时间（`stamp - delay`，见 `calibration.py`）。未来消息即使已进入接收队列也不会提前生效。

迟到或恢复后补传的旧消息不会被打到当前时间上：系统在开放区间内按校正时间因果重放历史，再把实时状态向到达时钟预测，因此位置不会瞬间跳回旧时刻。

## 封存与延迟修订

`seal_until(t)` 之前的输出是封存历史，数值不再变化。用户通过 `set_sensor_delay(sensor, delay)` 修改某一路固定延迟时，系统只重算封存边界之后受影响的开放轨迹区间并返回 `RecomputeReport`（传感器、旧/新延迟、受影响区间、实际重算起点）。若修订会覆盖封存区间，则抛出 `SealedHistoryError` 并列出受阻的封存时刻，封存输出保持原值，不会被静默替换。

轮速断流、IMU 断流和疑似打滑都会写入 `degraded_reasons`。断流期间协方差继续增长，因此误差范围会扩大，不会声称精度保持不变。

检查点保存到达边界（`arrival_horizon`）、封存边界、各路延迟和当时已可见的消息。恢复后可见数据边界与保存时一致：更晚到达的消息仍然隐藏，必须等到其到达时间才放行。

默认示例同时运行零延迟（未校正）基线与延迟校正运行，`metrics.json` 输出两者的位置 RMSE、95% 半径覆盖率、重算记录、被封存拒绝的修订以及恢复前后的可见边界；`trajectory.csv` 增加 `raw_position_error` 与 `sealed` 列。

## 测试

```bash
python -m unittest -v
```
