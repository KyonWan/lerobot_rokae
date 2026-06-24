"""Monorepo-safe entry: register robot configs when this project dir is on sys.path."""

from .lerobot_robot_rokae.devices.bi_rokae_robot.config_bi_rokae_robot import (  # noqa: F401
    BiRokaeRobotConfig,
)
from .lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import RokaeRobotConfig  # noqa: F401
