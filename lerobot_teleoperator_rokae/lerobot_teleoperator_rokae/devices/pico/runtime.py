from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    CoreArmRuntime,
)


@dataclass
class PicoArmRuntime(CoreArmRuntime):
    session_start_pos_flan_in_base: np.ndarray | None = None
    session_start_ori_flan_in_base: np.ndarray | None = None
    last_gripper_trigger: float = 1.0
