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
    # ip to connet the robot
    host_ip: str = "192.168.21.1"
    robot_ip: str = "192.168.21.10"
    
    # basic params
    joint_num: int = 6
    # control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    # callback_mode: CallbackMode = CallbackMode.CART_VEL

    control_mode: ControlMode = ControlMode.JOINT_POSITION
    callback_mode: CallbackMode = CallbackMode.JOINT_POS

    # Server port for HTTP API (default 5000)
    server_port: int = 5000

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
