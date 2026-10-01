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

`seal_until(t)` 之前的输出是封存历史，数值不再变化。延迟修订先用 `stage_sensor_delay(sensor, delay)` 生成 `DelayComparison`：两套设置并排列出校正区间、受影响观测序号、仍可重算的开放观测、不能修改的封存观测，以及封存边界所依据的 `boundary_state_time`。确认后调用 `publish_staged_delay()`，系统从封存边界状态恢复估计器，只因果重放开放区间；调用 `discard_staged_delay()` 可撤销尚未发布的校正，轨迹和当前延迟均保持不变。

迟到观测跨越封存边界时逐条给出原因：旧、新校正时间都在边界后的观测参与开放重算；都在边界前的观测属于不可修改历史；候选延迟会把开放观测移入封存区间时拒绝该观测并记录原因，而不是回写旧位置；候选延迟把封存观测移到边界后时，则从同一个封存锚点开始纳入开放重算。封存之后才到达、却校正到边界之前的迟到消息同样只记录拒用原因，不进入融合。

运行过程中可以继续向前封存；新封存只是把当时开放输出追加到历史。检查点保存封存输出、封存边界、已发布延迟、到达边界、可见消息及拒用观测。文件重开后以封存边界状态恢复开放尾段，再继续按到达时间接收消息，因此同一条轨迹的观测时间、到达时间和封存时间含义保持一致。

轮速断流、IMU 断流和疑似打滑都会写入 `degraded_reasons`。断流期间协方差继续增长，因此误差范围会扩大，不会声称精度保持不变。

## 逐渐漂移的时钟校时

固定延迟只能表示恒定偏移，但某台传感器的时钟可能逐渐走快或走慢。`calibration.py` 的每路时钟模型因此是线性的：`corrected = epoch + (stamp - epoch)/(1 + drift) - delay`，其中 `drift=0` 时严格退化为原有的固定延迟校正；`stamp` 与 `arrival_time` 始终原样保留，任何校时都不会改写观测的原始时间戳或观测身份 `(sensor, sequence)`。

用户针对一段采集记录调用 `stage_clock_revision(sensor, delay, drift, epoch)` 生成 `ClockComparison`，与原时间解释并排展示：两套设置各自的 `delay/drift/epoch`、校正区间、受影响/开放/封存/被拒观测序号，以及封存边界 `boundary_state_time`。确认后用 `publish_staged_clock_revision()` 从封存锚点只重放开放区间，用 `discard_staged_delay()` 撤回未发布方案。

- **跨边界观测逐条说明**：旧、新校正时间都在边界后的参与开放重算（`open_recomputed`）；都在边界前的保持封存原值（`sealed_unchanged`）；候选模型会把开放观测移入封存区间的拒绝并记录原因（`crossing_rejected`），绝不回写封存轨迹；把封存观测移到边界后的则从同一封存锚点纳入开放重算（`open_adopted`）。
- **同刻与换序不丢身份**：开放事件按校正时间排序，同刻按传感器名与序号确定性打破平局。`simultaneous_groups` 列出变成同刻的身份组，`position_changes[].ordering_swapped_with` 列出交换先后的身份；`applied_identities` 去重保存所有真正应用的 `(sensor, sequence)`，不会因同刻或换序而重复应用或丢失任何一条观测。
- **位置变化归因**：`position_changes` 给出每条受影响观测的原/新校正时间、时间位移、相对原解释在同一轨迹时刻的位置变化及原因；`uncertain_intervals` 单独标出没有任何观测的区间和该传感器断流区间，说明这些位置仍因缺少观测而不确定，不能归因于校时。
- **未处理完的方案可重开续做**：未发布草稿随检查点保存（检查点新增 `staged_plan`，文件检查点新增 `staged_clock_plan` 与每路 `clock_models`）。重开后 `pending_clock_revision()` 返回同一方案并可直接发布；封存推进时草稿不丢失，只是按新边界刷新说明。再次封存或撤回未发布校时后，已封存轨迹与原始时间戳均不被改写。

默认示例同时运行零延迟（未校正）基线与延迟校正运行，`metrics.json` 输出两者的位置 RMSE、95% 半径覆盖率、已发布重算记录、两套延迟设置的并排比较、未发布校正的撤销记录以及恢复前后的可见边界；`trajectory.csv` 增加 `raw_position_error` 与 `sealed` 列。

默认示例的轮速时钟带 `0.02` 的线性漂移；`metrics.json` 额外输出漂移校时的并排比较（`clock_revision_comparison`）、未发布草稿保存重开后续做同一方案（`clock_draft_resumed_after_reopen`）以及发布后的位置变化、缺测不确定区间、同刻/换序身份与去重应用身份（`clock_revision_published`）。

## 测试

```bash
python -m unittest -v
```
