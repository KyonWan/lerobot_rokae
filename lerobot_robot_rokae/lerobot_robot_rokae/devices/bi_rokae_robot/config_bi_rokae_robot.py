#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from ..rokae_robot.config_rokae_robot import ControlMode, CallbackMode


@RobotConfig.register_subclass("bi_rokae_robot")
@dataclass
class BiRokaeRobotConfig(RobotConfig):
    id: str | None = "AR"
    # ZMQ server addresses
    # TCP mode: "tcp://127.0.0.1:5555" (works on all platforms, supports cross-network)
    # IPC mode (faster, Unix/Linux only): "ipc:///tmp/rokae_server_5555"
    # Note: Windows does not support IPC, will automatically use TCP
    #       Unix/Linux will use IPC for better performance
    left_zmq_address: str | None = None
    right_zmq_address: str | None = None

    # ZMQ server ports (only used if zmq_address is None)
    # Default: left=5555, right=5556
    left_zmq_port: int = 5555
    right_zmq_port: int = 5556

    # Basic params
    left_joint_num: int = 7
    right_joint_num: int = 7

    # 控制循环频率（Hz）。在录制脚本中会由 dataset.fps 自动覆盖，并传播到左右单臂的 RokaeRobotConfig。
    control_loop_fps: int | None = None

    # Control modes
    left_control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    left_callback_mode: CallbackMode = CallbackMode.CART_VEL
    right_control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDNACE
    right_callback_mode: CallbackMode = CallbackMode.CART_VEL

    # rokae_algo 运动学初始化参数（joint_pos 模式下 BiInverseKinematicsProcessor 使用）
    left_rbv: list[float] = field(default_factory=list)
    right_rbv: list[float] = field(default_factory=list)
    left_min_joint: list[float] = field(default_factory=list)
    left_max_joint: list[float] = field(default_factory=list)
    right_min_joint: list[float] = field(default_factory=list)
    right_max_joint: list[float] = field(default_factory=list)

    # Cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
