"""单臂几何上下文：ref/base 变换与 IK 工具链共用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArmFrameContext:
    """ref/base 与工具链参数（来自 RokaeRobot / ZMQ tool_info）。"""

    tool_end_pos: np.ndarray
    tool_ref_pos: np.ndarray
    base_frame_in_world: np.ndarray

    @classmethod
    def from_robot(cls, robot: Any) -> ArmFrameContext:
        """从 ``RokaeRobot`` 或兼容对象构造。"""
        return cls(
            tool_end_pos=_as_pose6(getattr(robot, "tool_end_pos", None)),
            tool_ref_pos=_as_pose6(getattr(robot, "tool_ref_pos", None)),
            base_frame_in_world=_as_pose6(getattr(robot, "base_frame_in_world", None)),
        )


def _as_pose6(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(6, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(6)
