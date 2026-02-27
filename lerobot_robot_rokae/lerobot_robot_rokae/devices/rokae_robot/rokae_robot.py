import logging
import platform
import time
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .config_rokae_robot import RokaeRobotConfig, ControlMode, CallbackMode
from rokae_python_wrapper.rokae_zmq_client import RokaeZmqClient, RokaeZmqClientError
import numpy as np

logger = logging.getLogger(__name__)
# # 设置 logger 级别为 DEBUG 以显示性能日志
# # 注意：如果根 logger 的 handler 级别是 INFO，需要确保 handler 也接受 DEBUG
# logger.setLevel(logging.DEBUG)
# # 如果没有 handler 或 handler 级别太高，添加一个 DEBUG 级别的 handler
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
# else:
#     # 确保现有 handler 也接受 DEBUG 级别
#     for handler in logger.handlers:
#         if handler.level > logging.DEBUG:
#             handler.setLevel(logging.DEBUG)


class RokaeRobot(Robot):
    config_class = RokaeRobotConfig
    name = "rokae_robot"

    def __init__(self, config:RokaeRobotConfig):
        super().__init__(config)
        self.cfg = config
        self.joint_num = config.joint_num
        self.cameras = make_cameras_from_configs(config.cameras)
        self.gripper_pos_cur = None

        # 使用 ZMQ 客户端
        if config.zmq_address:
            zmq_address = config.zmq_address
        else:
            # 自动生成地址：Windows 使用 TCP，Unix/Linux 使用 IPC
            zmq_port = getattr(config, "zmq_port", 5555)
            if platform.system() == "Windows":
                zmq_address = f"tcp://127.0.0.1:{zmq_port}"
            else:
                zmq_address = f"ipc:///tmp/rokae_server_{zmq_port}"
        self.client = RokaeZmqClient(address=zmq_address)
        logger.info(f"Using ZMQ client: {zmq_address}")

        tool_info = self.client.get_tool_info()
        self.tool_info = tool_info
        self.tool_mass = float(tool_info.get("mass"))
        self.tool_center_of_mass = np.array(tool_info.get("center_of_mass"), dtype=np.float64)
        self.tool_inertia_tensor = np.array(tool_info.get("inertia_tensor"), dtype=np.float64)
        self.tool_end_pos = np.array(tool_info.get("end_pos"), dtype=np.float64)
        self.tool_ref_pos = np.array(tool_info.get("ref_pos"), dtype=np.float64)
        self.base_frame_in_world = np.array(self.client.get_base_frame(), dtype=np.float64)

    @property
    def _robot_ft(self) -> dict[str, type]:
        return {**{f"joint_pos{i}": float for i in range(self.joint_num)}, **{f"cart_pos{i}": float for i in range(6)}, "psi": float, "gripper_pos": float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cfg.cameras[cam].height, self.cfg.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._robot_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**{f"joint_pos{i}": float for i in range(self.joint_num)}, "gripper_pos": float}

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

        # Read arm position
        start = time.perf_counter()
        state = self.client.get_state(
            ["joint_pos_cmd", "cart_pos_cmd", "psi", "gripper_pos"]
        )
        obs_dict = {
            **{f"joint_pos{i}": state["joint_pos_cmd"][i] for i in range(self.joint_num)},
            **{f"cart_pos{i}": state["cart_pos_cmd"][i] for i in range(6)},
            "psi": state["psi"],
            "gripper_pos": state["gripper_pos"][0],
        }
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
        # ZMQ 客户端使用一次 RPC 完成 set_target + gripper + get_state，减少往返延迟
        if hasattr(self.client, "send_action_and_get_state"):
            if self.cfg.callback_mode == CallbackMode.JOINT_POS:
                robot_action = np.array([action[f"joint_pos{i}"] for i in range(self.joint_num)])
                action_type = "joint_pos"
            elif self.cfg.callback_mode == CallbackMode.CART_POS:
                robot_action = np.array([action[f"cart_pos{i}"] for i in range(6)])
                action_type = "cart_pos"
            else:  # CART_VEL
                robot_action = np.array([action[f"cart_vel{i}"] for i in range(6)])
                action_type = "cart_vel"
            gripper_pos = float(action["gripper_pos"])
            state = self.client.send_action_and_get_state(
                action_type=action_type,
                action_value=robot_action,
                gripper_pos=gripper_pos,
                quantities=["joint_pos_cmd", "gripper_pos"],
            )
            self.gripper_pos_cur = gripper_pos
            return {
                **{f"joint_pos{i}": state["joint_pos_cmd"][i] for i in range(self.joint_num)},
                "gripper_pos": state["gripper_pos"][0],
            }
        # ZMQ 客户端回退逻辑（如果 send_action_and_get_state 不可用）
        # 这种情况不应该发生，因为 ZMQ 客户端总是支持 send_action_and_get_state
        # 但保留此逻辑作为安全回退
        if self.cfg.callback_mode == CallbackMode.JOINT_POS:
            robot_action = np.array([action[f"joint_pos{i}"] for i in range(self.joint_num)])
            self.client.set_target_joint_pos(robot_action)
        elif self.cfg.callback_mode == CallbackMode.CART_POS:
            robot_action = np.array([action[f"cart_pos{i}"] for i in range(6)])
            self.client.set_target_cart_pos(robot_action)
        elif self.cfg.callback_mode == CallbackMode.CART_VEL:
            robot_action = np.array([action[f"cart_vel{i}"] for i in range(6)])
            self.client.set_target_cart_vel(robot_action)

        if action["gripper_pos"] == 0 and self.gripper_pos_cur != 0:
            self.client.close_gripper()
        elif action["gripper_pos"] == 1 and self.gripper_pos_cur != 1:
            self.client.open_gripper()
        self.gripper_pos_cur = action["gripper_pos"]

        state = self.client.get_state(["joint_pos_cmd", "cart_pos_cmd", "psi", "gripper_pos"])

        return {**{f"joint_pos{i}": state["joint_pos_cmd"][i] for i in range(self.joint_num)}, "gripper_pos": state["gripper_pos"][0]}

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
    
    def set_gripper_state(self, gripper_pos: int) -> None:
        """
        直接设置夹爪状态，不影响机械臂动作。
        
        Args:
            gripper_pos: 夹爪状态，0=关闭，1=打开
        """
        if gripper_pos == 0:
            self.client.close_gripper()
        elif gripper_pos == 1:
            self.client.open_gripper()
        else:
            raise ValueError(f"Invalid gripper_pos: {gripper_pos}, must be 0 or 1")
        
        self.gripper_pos_cur = gripper_pos
