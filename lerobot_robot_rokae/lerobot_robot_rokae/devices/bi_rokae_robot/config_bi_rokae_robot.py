#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from ..rokae_single_arm.config_rokae_robot import ControlMode, CallbackMode


@RobotConfig.register_subclass("bi_rokae_robot")
@dataclass
class BiRokaeRobotConfig(RobotConfig):
    id: str | None = "AR"
    # Server ports for HTTP API (each arm needs its own server on different port)
    left_server_port: int = 5000
    right_server_port: int = 5001

    # Communication protocol: "http" or "zmq" (default "zmq")
    protocol: str = "zmq"

    # ZMQ server addresses (only used if protocol="zmq")
    # Left arm: port 5000 -> ZMQ port 5555
    # Right arm: port 5001 -> ZMQ port 5556
    # Note: Windows does not support IPC, will automatically use TCP
    #       Unix/Linux will use IPC for better performance
    left_zmq_address: str | None = None
    right_zmq_address: str | None = None

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
