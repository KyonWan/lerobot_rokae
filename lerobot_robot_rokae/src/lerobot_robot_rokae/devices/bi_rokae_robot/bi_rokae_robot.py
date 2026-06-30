import logging
import platform
import time
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from .config_bi_rokae_robot import BiRokaeRobotConfig
from ..rokae_robot.rokae_robot import RokaeRobot
from ..rokae_robot.config_rokae_robot import RokaeRobotConfig

logger = logging.getLogger(__name__)


class BiRokaeRobot(Robot):
    """
    Bimanual Rokae robot consisting of two single-arm robots.
    Each arm is controlled independently via its own server.
    """

    config_class = BiRokaeRobotConfig
    name = "bi_rokae_robot"

    def __init__(self, config: BiRokaeRobotConfig):
        super().__init__(config)
        self.cfg = config

        # 自动生成 ZMQ 地址（如果未指定）
        left_zmq_address = config.left_zmq_address
        if left_zmq_address is None:
            left_zmq_port = getattr(config, "left_zmq_port", 5555)
            if platform.system() == "Windows":
                left_zmq_address = f"tcp://127.0.0.1:{left_zmq_port}"
            else:
                left_zmq_address = f"ipc:///tmp/rokae_server_{left_zmq_port}"

        right_zmq_address = config.right_zmq_address
        if right_zmq_address is None:
            right_zmq_port = getattr(config, "right_zmq_port", 5556)
            if platform.system() == "Windows":
                right_zmq_address = f"tcp://127.0.0.1:{right_zmq_port}"
            else:
                right_zmq_address = f"ipc:///tmp/rokae_server_{right_zmq_port}"

        # Create left arm config
        left_arm_config = RokaeRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            control_mode=config.left_control_mode,
            zmq_address=left_zmq_address,
            control_loop_fps=getattr(config, "control_loop_fps", None),
            cameras={},  # Cameras are shared at the bimanual level
        )

        # Create right arm config
        right_arm_config = RokaeRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            control_mode=config.right_control_mode,
            zmq_address=right_zmq_address,
            control_loop_fps=getattr(config, "control_loop_fps", None),
            cameras={},  # Cameras are shared at the bimanual level
        )

        self.left_arm = RokaeRobot(left_arm_config)
        self.right_arm = RokaeRobot(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _left_robot_ft(self) -> dict[str, type]:
        return {
            **{f"left_joint_pos{i}": float for i in range(self.left_arm.joint_num)},
            **{f"left_cart_pos{i}": float for i in range(6)},
            "left_psi": float,
            "left_gripper_pos": float
        }

    @property
    def _right_robot_ft(self) -> dict[str, type]:
        return {
            **{f"right_joint_pos{i}": float for i in range(self.right_arm.joint_num)},
            **{f"right_cart_pos{i}": float for i in range(6)},
            "right_psi": float,
            "right_gripper_pos": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cfg.cameras[cam].height, self.cfg.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._left_robot_ft, **self._right_robot_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._left_robot_ft, **self._right_robot_ft}

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected 
            and self.right_arm.is_connected 
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Get left arm observation and add "left_" prefix
        left_obs = self.left_arm.get_observation()
        for key, value in left_obs.items():
            if not key.startswith("left_") and key not in self.cameras:
                obs_dict[f"left_{key}"] = value
            elif key in self.cameras:
                # Camera data from left arm (if any)
                obs_dict[f"left_{key}"] = value

        # Get right arm observation and add "right_" prefix
        right_obs = self.right_arm.get_observation()
        for key, value in right_obs.items():
            if not key.startswith("right_") and key not in self.cameras:
                obs_dict[f"right_{key}"] = value
            elif key in self.cameras:
                # Camera data from right arm (if any)
                obs_dict[f"right_{key}"] = value

        # Add shared cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:

        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value 
            for key, value in action.items() 
            if key.startswith("left_")
        }

        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value 
            for key, value in action.items() 
            if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        if not self.is_connected:
            return

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def reset_position(self):
        """重置左右臂到拖拽位姿"""
        self.left_arm.reset_position()
        self.right_arm.reset_position()
    
    def set_gripper_states(self, left_gripper_pos: int, right_gripper_pos: int) -> None:
        """
        直接设置左右夹爪状态，不影响机械臂动作。
        
        Args:
            left_gripper_pos: 左夹爪状态，0=关闭，1=打开
            right_gripper_pos: 右夹爪状态，0=关闭，1=打开
        """
        self.left_arm.set_gripper_state(left_gripper_pos)
        self.right_arm.set_gripper_state(right_gripper_pos)
