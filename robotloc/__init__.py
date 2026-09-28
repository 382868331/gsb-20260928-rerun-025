"""离线机器人定位核心。

模块按职责拆分，均可独立替换：

- :mod:`robotloc.types`      传感器消息 / 状态的数据结构
- :mod:`robotloc.sources`    传感器接入（离线文件源等，不接触真实设备）
- :mod:`robotloc.estimator`  状态估计（轮速 + 惯性 EKF）
- :mod:`robotloc.recording`  记录与回放（JSONL）
- :mod:`robotloc.pipeline`   因果处理：迟到消息 / 断流 / 检查点
- :mod:`robotloc.mapping`    地图匹配（占位接口，本轮不实现）
"""

from .types import ImuObservation, PoseEstimate, WheelObservation

__all__ = ["ImuObservation", "WheelObservation", "PoseEstimate"]
