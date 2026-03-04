from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_spacemouse")
@dataclass
class BiSpacemouseConfig(TeleoperatorConfig):
    # Device indices for left and right spacemouse devices
    left_device_index: Optional[int] = 0
    right_device_index: Optional[int] = 1
    # 上层参考笛卡尔速度上限（m/s 和 rad/s），会传入 BiInverseKinematicsProcessor
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
