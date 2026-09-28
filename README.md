# 离线机器人定位核心

这是不读取真实设备的 Python 离线核心。传感器接入、状态估计和记录回放分别位于 `sources.py`、`estimator.py` 和 `recording.py`，`pipeline.py` 负责按观测时间因果处理。地图匹配暂未加入，后续可作为独立测量源接入。

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

`LocalizationPipeline.process_until(t)` 只使用观测时间不晚于 `t` 的消息。未来消息即使已经进入接收队列，也不会提前影响状态。迟到的旧消息到达时，系统回滚到早于该消息的最近检查点，并重放后续事件；结果与一次性按观测时间完整回放比较。

轮速断流、IMU 断流和疑似打滑都会写入 `degraded_reasons`。断流期间协方差继续增长，因此误差范围会扩大，不会声称精度保持不变。

## 测试

```bash
python -m unittest -v
```
