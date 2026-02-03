#!/usr/bin/env python
 
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from ..rokae_single_arm.config_rokae_robot import ControlMode, CallbackMode


@RobotConfig.register_subclass("bi_rokae_robot")
@dataclass
class BiRokaeRobotConfig(RobotConfig):
    # Server ports for HTTP API (each arm needs its own server on different port)
    left_server_port: int = 5000
    right_server_port: int = 5001
    
    # Basic params
    left_joint_num: int = 7
    right_joint_num: int = 7
    
    # Control modes
    left_control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    left_callback_mode: CallbackMode = CallbackMode.CART_VEL
    right_control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    right_callback_mode: CallbackMode = CallbackMode.CART_VEL

    # Cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
