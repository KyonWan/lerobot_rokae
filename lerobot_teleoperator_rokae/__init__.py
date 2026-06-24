"""Monorepo-safe entry: register teleop configs when this project dir is on sys.path."""

from .lerobot_teleoperator_rokae import (  # noqa: F401
    BiSpacemouseConfig,
    PicoConfig,
    PicoSingleConfig,
    SpacemouseConfig,
)
