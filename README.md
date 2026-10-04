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

每条消息同时保留戳记观测时间 `time` 和墙钟到达时间 `arrival_time`。`process_until(t)` 只放行到达时间不晚于 `t` 的消息，融合时再按每路时钟模型把戳记校正为轨迹时间（`stamp - offset(stamp)`，见 `calibration.py`）。固定延迟只是单点锚点；两点或更多锚点形成分段线性模型，可表达逐渐偏移。未来消息即使已进入接收队列也不会提前生效。

迟到或恢复后补传的旧消息不会被打到当前时间上：系统在开放区间内按校正时间因果重放历史，再把实时状态向到达时钟预测，因此位置不会瞬间跳回旧时刻。

## 封存与时钟校时

`seal_until(t)` 之前的输出是封存历史，数值不再变化。校时先用 `stage_sensor_clock(sensor, anchors=...)`（固定延迟仍可用 `stage_sensor_delay`）生成 `DelayComparison`：两套解释并排列出每条 `(sensor, sequence)` 观测的原始戳记、到达时间、原校正时间、候选校正时间、时间移动量及全局顺序变化。确认后调用 `publish_staged_delay()`，系统从封存边界状态恢复估计器，只因果重放开放区间；调用 `discard_staged_delay()` 可撤销尚未发布的校正，轨迹、已封存输出和观测原始 `time` 均保持不变。

校时只改变观测在轨迹时间轴上的位置，不改变 `(sensor, sequence)` 身份。不同来源在候选时间下同刻时，按 `(corrected_time, sensor, sequence)` 稳定应用；交换先后时也按新顺序各应用一次，不会因为序号或到达时间变化而重复应用、丢弃或合并观测。`position_changes` 保留每个身份在原解释和候选解释下的对应状态位置，`timeline` 明确哪些移动来自校时、哪些仅顺序受其他观测影响。

开放区间重估后，`uncertain_intervals` 列出仍然超过传感器超时阈值的空窗及原因；这些位置变化来自缺少观测后的预测，而不是校时本身。

观测跨越封存边界时逐条给出原因：旧、新校正时间都在边界后的观测参与开放重算；都在边界前的观测属于不可修改历史；候选校时会把开放观测移入封存区间时拒绝该观测并记录原因，而不是回写旧位置；候选校时把封存观测移到边界后时，则从同一个封存锚点开始纳入开放重算。封存之后才到达、却校正到边界之前的迟到消息同样只记录拒用原因，不进入融合。

运行过程中可以继续向前封存；新封存只是把当时开放输出追加到历史。检查点保存封存输出、封存边界、已发布时钟模型、到达边界、可见消息、拒用观测以及尚未发布的同一套校时草稿和目标传感器。文件重开后以封存边界状态恢复开放尾段并继续同一草稿；草稿可继续查看、发布或撤回，而不会在保存期间自动改写已封存轨迹。

轮速断流、IMU 断流和疑似打滑都会写入 `degraded_reasons`。断流期间协方差继续增长，因此误差范围会扩大，不会声称精度保持不变。

## 设备重启与分段校时

设备在采集途中重启会把设备钟重新置零，同一原始时间戳会在不同采集段重复出现。系统以 `(sensor, sequence)` 作为观测身份，并以重启边界之间的序号划分采集段（`segments.py`）：相同原始戳记落在不同段时，经各自段的锚点映射到不同轨迹时间，绝不会被当成同一次观测。

- `stage_reboot_segments(sensor, boundaries=..., anchors=...)` 并排比较整套“重启边界 + 各段校时”解释；`RebootBoundary(after_sequence, confirmed, reset_window)` 可表示明确重启记录，也可只给出轨迹时间上的推测窗口而不伪造精确重启时刻。
- 每段使用自己的 `SegmentAnchor(stamp, corrected_time, uncertainty)`。只有一个可靠锚点时状态标记为 `single_anchor`；跨段锚点范围矛盾时给出 `reboot_contradictions` 并保留各自范围，`DelayComparison.segment_views` 逐段列出依据、序号、戳记区间、校正时间区间和不能确定的原因。
- `stage_reboot_segment()` / `confirm_reboot_segment()` 只修改指定段；确认后仅在仍开放且受影响的区间重估，撤回另一段未确认草稿不会撤销早先的确认。未校时段保持身份映射并标记 `uncalibrated`，不会把上一段或上一次重启前的校时参数静默沿用过去。
- 段间交错、同刻与跨过封存边界的观测沿用逐条 `ObservationDecision`（参与开放重算、封存不变、拒绝移入封存、重新纳入开放），`TimelinePosition` 增加 `segment_index`；观测的原始 `time`、身份和历史解释均不改写，迟到观测按校正后的段内轨迹时间因果重放。
- 分段模型（`type: reboot_segments`、各段锚点、边界确认状态与推测窗口）随检查点保存；恢复后继续同一套解释方案，可继续比较、逐段确认或撤回。

`RebootScenarioSource` 生成含缓慢漂移、两次归零和迟到观测的记录；`tests/test_reboot_segments.py` 覆盖同戳不同段、单锚点/推测窗口范围、跨段矛盾、交错与迟到排序、封存逐条决策、方案比较、逐段确认互不影响以及保存恢复。

## 多解释候选与迟到锚点

当重启只能确定在某个序号范围内时，`RebootBoundary(sequence_range=(low, high))` 记录候选切点范围，名义切点保持显式；`register_interpretation(name, sensor, boundaries=..., segment_anchors=..., anchor_ids=..., adopt=...)` 把每一套“边界 + 漂移锚点”解释登记为未决候选并立即给出与当前校时的 `DelayComparison` 并排比较（校正时刻变化、`position_changes` 的下游轨迹影响）。候选只登记不发布，同名候选拒绝覆盖，发布必须显式 `publish_interpretation(name)`，绝不会被最后一条候选自动顶替；观测的原始 `time`、`sequence` 与身份始终不变。

重置两侧已出现相同原始时间后补到的锚点用 `register_anchor(..., late=True)` 登记，再以 `adopt=[(sensor, sequence), ...]` 生成只采用部分观测的新解释：被采用的身份按候选段内模型校正，其余观测维持原版本（`DelayCalibration` 的 selective 形式随 `as_dict` 持久化）。`withdraw_anchor(anchor_id)` 返回 `AnchorWithdrawal`，列出并移除所有引用该锚点的未决解释，已发布校时不受影响。

已登记锚点与未决解释随检查点保存（`registered_anchors`、`pending_interpretations`）；重开后比较结果、不确定范围与下游采用依据会按恢复后的校时重算，可继续比较、发布或撤回。`tests/test_interpretations.py` 覆盖序号范围多候选比较、迟到锚点部分采用、锚点撤回依赖清单以及保存重开后的可追溯性。

默认示例同时运行零延迟（未校正）基线与时钟校正运行，`metrics.json` 输出两者的位置 RMSE、95% 半径覆盖率、已发布重算记录、两套时钟解释的逐身份并排比较、未发布校时的撤销记录、开放区间不确定空窗以及恢复前后的可见边界；`trajectory.csv` 增加 `raw_position_error` 与 `sealed` 列。

## 测试

```bash
python -m unittest -v
```
