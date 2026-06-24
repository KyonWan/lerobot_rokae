from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    # Device index for multiple spacemouse devices (0 for first, 1 for second, etc.)
    device_index: Optional[int] = 0
    # 上层参考笛卡尔速度上限（m/s 和 rad/s），会传入 build_arm_pipeline
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
