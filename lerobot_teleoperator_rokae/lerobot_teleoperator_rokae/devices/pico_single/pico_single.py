import logging
import threading
import time
from typing import Any

import meshcat.transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_pico_single import PicoSingleConfig

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

SIDE_MANIPULATOR_CONFIG = {
    "left": {
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
    },
    "right": {
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}


class PicoSingle(Teleoperator):
    """
    Pico 单臂遥操器。

    通过 `config.side` 选择左/右手柄作为输入，仅输出单臂 action 键：
    `target_x/y/z/wx/wy/wz` 与 `gripper_pos`。
    """

    config_class = PicoSingleConfig
    name = "pico_single"

    def __init__(self, config: PicoSingleConfig):
        super().__init__(config)
        self.xr_client = config.xr_client or XrClient()
        self.cfg = config
        self.side_config = SIDE_MANIPULATOR_CONFIG[self.cfg.side]

        self._is_connected = False
        self._stop_event = threading.Event()

        self._last_trigger_val = 1.0
        self.gripper_pos = self.cfg.open_position

        self.init_controller_xyz: np.ndarray | None = None
        self.init_controller_quat: np.ndarray | None = None
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.base_delta_xyz = np.zeros(3)
        self.base_delta_rot = np.zeros(3)
        self.was_active = False

        self.R_headset_world = R.from_euler("ZYX", self.cfg.R_headset_world, degrees=True).as_matrix()

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.init_controller_xyz = None
        self.init_controller_quat = None
        self.base_delta_xyz = np.zeros(3)
        self.base_delta_rot = np.zeros(3)
        self.was_active = False

        threading.Thread(target=self._start_pose_update, daemon=True).start()
        self._is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.")

    def reset_for_new_episode(self) -> None:
        """Clear accumulated delta targets to avoid replaying previous episode pose."""
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.base_delta_xyz = np.zeros(3)
        self.base_delta_rot = np.zeros(3)
        self.init_controller_xyz = None
        self.init_controller_quat = None
        self.was_active = False
        self._last_trigger_val = self.cfg.open_position
        self.gripper_pos = self.cfg.open_position

    def _start_pose_update(self) -> None:
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()
                self._update_pose_deltas_from_xr()
                elapsed = time.perf_counter() - start
                time.sleep(max(0, 1 / self.cfg.fps - elapsed))
            except Exception as exc:
                logger.error(f"Error in pose update thread: {exc}")

    def _update_pose_deltas_from_xr(self) -> None:
        xr_grip_val = self.xr_client.get_key_value_by_name(self.side_config["control_trigger"])
        active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

        trigger_val = self.xr_client.get_key_value_by_name(self.side_config["gripper_trigger"])
        if self.cfg.trigger_reverse:
            trigger_val = self.cfg.open_position - trigger_val
        if trigger_val < self.cfg.trigger_threshold:
            trigger_val = self.cfg.close_position
        else:
            trigger_val = self.cfg.open_position

        if self._last_trigger_val == 1 and trigger_val == 0:
            self.gripper_pos = (
                self.cfg.close_position
                if self.gripper_pos == self.cfg.open_position
                else self.cfg.open_position
            )
        self._last_trigger_val = trigger_val

        if active:
            if not self.was_active:
                self.init_controller_xyz = None
                self.init_controller_quat = None

            xr_pose = self.xr_client.get_pose_by_name(self.side_config["pose_source"])
            delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose)
            self.current_delta_xyz = self.base_delta_xyz + delta_xyz
            self.current_delta_rot = self.base_delta_rot + delta_rot_angle_axis
        else:
            if self.was_active:
                self.base_delta_xyz = self.current_delta_xyz.copy()
                self.base_delta_rot = self.current_delta_rot.copy()
                self.init_controller_xyz = None
                self.init_controller_quat = None

        self.was_active = active

    def _process_xr_pose(self, xr_pose: list[float]) -> tuple[np.ndarray, np.ndarray]:
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])
        controller_xyz = self.R_headset_world @ controller_xyz

        r_transform = np.eye(4)
        r_transform[:3, :3] = self.R_headset_world
        r_quat = tf.quaternion_from_matrix(r_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(r_quat, controller_quat),
            tf.quaternion_conjugate(r_quat),
        )

        if self.init_controller_xyz is None:
            self.init_controller_xyz = controller_xyz.copy()
            self.init_controller_quat = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.zeros(3)
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz) * self.cfg.xyz_scale_factor
            delta_rot = (
                quat_diff_as_angle_axis(self.init_controller_quat, controller_quat) * self.cfg.rot_scale_factor
            )
        return delta_xyz, delta_rot

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        return {
            "target_x": float(self.current_delta_xyz[0]),
            "target_y": float(self.current_delta_xyz[1]),
            "target_z": float(self.current_delta_xyz[2]),
            "target_wx": float(self.current_delta_rot[0]),
            "target_wy": float(self.current_delta_rot[1]),
            "target_wz": float(self.current_delta_rot[2]),
            "gripper_pos": float(self.gripper_pos),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._stop_event.set()
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")
