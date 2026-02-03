import logging
import time
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .config_rokae_robot import RokaeRobotConfig, ControlMode, CallbackMode
from rokae_python_wrapper.rokae_client import RokaeClient
import numpy as np

logger = logging.getLogger(__name__)


class RokaeRobot(Robot):
    config_class = RokaeRobotConfig
    name = "rokae_robot"

    def __init__(self, config:RokaeRobotConfig):
        super().__init__(config)
        self.cfg = config
        self.joint_num = config.joint_num
        self.cameras = make_cameras_from_configs(config.cameras)
        # Support configurable port for multiple arms
        port = getattr(config, 'server_port', 5000)
        base_url = f"http://127.0.0.1:{port}"
        self.client = RokaeClient(base_url=base_url)

    @property
    def _tele_robot_ft(self) -> dict[str, type]:
        return {**{f"cart_pos{i}": float for i in range(self.joint_num)}, "gripper_pos": float}

    @property
    def _robot_ft(self) -> dict[str, type]:
        return {**{f"joint_pos{i}": float for i in range(self.joint_num)}, "gripper_pos": float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cfg.cameras[cam].height, self.cfg.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._robot_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._robot_ft

    @cached_property
    def tele_action_features(self) -> dict[str, type]:
        return self._tele_robot_ft

    @property
    def is_connected(self) -> bool:
        return self.client.is_in_realtime_loop() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        if not self.client.is_in_realtime_loop():
            self.client.start_realtime_loop(self.cfg.control_mode.value, self.cfg.callback_mode.value)

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        state = self.client.get_state(["joint_pos_cmd", "gripper_pos"])
        obs_dict = {**{f"joint_pos{i}": state["joint_pos_cmd"][i] for i in range(self.joint_num)}, "gripper_pos": state["gripper_pos"][0]}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.cfg.callback_mode == CallbackMode.JOINT_POS:
            robot_action = np.array([action[f"joint_pos{i}"] for i in range(self.joint_num)])
            self.client.set_target_joint_pos(robot_action)
        elif self.cfg.callback_mode == CallbackMode.CART_POS:
            robot_action = np.array([action[f"cart_pos{i}"] for i in range(6)]) # todo
            self.client.set_target_cart_pos(robot_action)
        elif self.cfg.callback_mode == CallbackMode.CART_VEL:
            robot_action = np.array([action[f"cart_vel{i}"] for i in range(6)]) # todo
            self.client.set_target_cart_vel(robot_action)

        if action["gripper_pos"] == 0:
            self.client.close_gripper()
        elif action["gripper_pos"] == 1:
            self.client.open_gripper()

        state = self.client.get_state(["joint_pos_cmd", "gripper_pos"])  # todo: use next time's joint position

        return {**{f"joint_pos{i}": state["joint_pos_cmd"][i] for i in range(self.joint_num)}, "gripper_pos": state["gripper_pos"][0]} # todo

    def disconnect(self):
        if not self.is_connected:
            return

        self.client.stop_realtime_loop()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def reset_position(self):
        """重置机器人到拖拽位姿"""
        return self.client.reset_position()
