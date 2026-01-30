from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_spacemouse")
@dataclass
class BiSpacemouseConfig(TeleoperatorConfig):
    # Device indices for left and right spacemouse devices
    left_device_index: Optional[int] = 0
    right_device_index: Optional[int] = 1
