from __future__ import annotations

from dataclasses import dataclass

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    CoreArmRuntime,
)


@dataclass
class SpaceMouseArmRuntime(CoreArmRuntime):
    button_prev: int = 0
