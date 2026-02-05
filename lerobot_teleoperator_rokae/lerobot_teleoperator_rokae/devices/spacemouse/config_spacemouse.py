from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    # Device index for multiple spacemouse devices (0 for first, 1 for second, etc.)
    device_index: Optional[int] = 0