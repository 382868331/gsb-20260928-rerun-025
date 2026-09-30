# 离线机器人定位核心

这是不读取真实设备的 Python 离线核心。传感器接入、状态估计和记录回放分别位于 `sources.py`、`estimator.py` 和 `recording.py`，`pipeline.py` 负责按观测时间因果处理。地图匹配暂未加入，后续可作为独立测量源接入。

## 运行

```bash
python -m robot_localization.cli --output-dir outputs
```

多传感器定时校正示例：

```bash
python -m robot_localization.cli --timed-calibration --output-dir outputs
```

输出：

- `messages.jsonl`：按到达顺序记录的虚构轮速与 IMU 消息。
- `trajectory.csv`：位置、朝向、误差、95% 误差半径和降级原因。
- `metrics.json`：已知路径的汇总误差。
- `checkpoint.json`：示例中途检查点。

## 因果与迟到

`LocalizationPipeline.process_until(t)` 只使用观测时间不晚于 `t` 的消息。未来消息即使已经进入接收队列，也不会提前影响状态。迟到的旧消息到达时，系统回滚到早于该消息的最近检查点，并重放后续事件；结果与一次性按观测时间完整回放比较。

带时间偏移的观测同时保留两个字段：`time` 是传感器观测戳，`arrival_time` 是到达记录器的时刻；融合使用 `time - sensor_delays[sensor]` 得到的校正时间，不覆写任何原始字段。可见性同时受到达边界和校正观测边界约束。

可调用 `seal_outputs_through(t)` 封存已发布轨迹。之后修改固定延迟只重算封存点之后的受影响区间；落入封存区间的迟到或补传观测仍保留在 `received` 中，但不会被融合，也不会静默替换旧输出。原因、请求区间、实际重算区间和封存区间记录在 `revision_records`。

持续断流期间继续预测并标记降级、扩大协方差。恢复后的突发补传只从封存点重建到当前到达时刻，并在该时刻输出状态，因此当前估计不会瞬间跳回旧观测时刻。

检查点保存 `horizon`、`visible_arrival`、`sealed_horizon` 和各路 `sensor_delays`，恢复后只看到保存时已经可见的数据；完整消息文件和可见检查点消息的边界不混用。

`--timed-calibration` 使用已知轨迹生成固定延迟、乱序、断流补传和打滑观测，并在 `metrics.json` 中对照校正前后 RMSE、平均 95% 半径及覆盖率。

轮速断流、IMU 断流和疑似打滑都会写入 `degraded_reasons`。断流期间协方差继续增长，因此误差范围会扩大，不会声称精度保持不变。

## 测试

```bash
python -m unittest -v
```
