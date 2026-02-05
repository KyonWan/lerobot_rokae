#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from enum import Enum

class ControlMode(str, Enum):
    JOINT_POSITION = "joint_position"
    CARTETIAN_POSITION = "cartesian_position"
    JOINT_IMPEDNACE = "joint_impedance"
    CARTESIAN_IMPEDNACE = "cartesian_impedance"


class CallbackMode(str, Enum):
    JOINT_POS = "joint_pos"
    CART_POS = "cart_pos"
    CART_VEL = "cart_vel"


@RobotConfig.register_subclass("rokae_robot")
@dataclass
class RokaeRobotConfig(RobotConfig):
    # basic params
    joint_num: int = 6
    # control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    # callback_mode: CallbackMode = CallbackMode.CART_VEL

    control_mode: ControlMode = ControlMode.JOINT_POSITION
    callback_mode: CallbackMode = CallbackMode.JOINT_POS

    # ZMQ server address
    # TCP mode: "tcp://127.0.0.1:5555" (works on all platforms, supports cross-network)
    # IPC mode (faster, Unix/Linux only): "ipc:///tmp/rokae_server_5555"
    # Note: Windows does not support IPC, will automatically use TCP
    # Default: Auto-generated based on OS (TCP on Windows, IPC on Unix/Linux)
    zmq_address: str | None = None

    # ZMQ server port (only used if zmq_address is None)
    # Default: 5555
    zmq_port: int = 5555

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
