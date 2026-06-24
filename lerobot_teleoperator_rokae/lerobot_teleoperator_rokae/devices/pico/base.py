"""Shared implementation for Pico-based teleoperators."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import meshcat.transformations as tf
import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class PicoInputSpec:
    pose_source: str
    clutch_button: str
    gripper_trigger_source: str
    action_prefix: str = ""


@dataclass
class PicoArmState:
    init_controller_xyz: np.ndarray | None = None
    init_controller_quat: np.ndarray | None = None
    current_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    was_active: bool = False
    raw_gripper_trigger: float = 0.0

    def reset_pose(self) -> None:
        self.init_controller_xyz = None
        self.init_controller_quat = None
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.base_delta_xyz = np.zeros(3)
        self.base_delta_rot = np.zeros(3)
        self.was_active = False

    def reset_raw_gripper_trigger(self, raw_open_value: float) -> None:
        self.raw_gripper_trigger = raw_open_value


class PicoTeleopBase(Teleoperator):
    """Common XR polling, clutching, and pose-delta logic for Pico teleoperators."""

    def __init__(
        self,
        config: Any,
        *,
        input_specs: dict[str, PicoInputSpec],
        euler_sequence: str,
    ) -> None:
        super().__init__(config)
        self.xr_client = config.xr_client or XrClient()
        self.cfg = config
        self.input_specs = dict(input_specs)
        self.arm_states = {
            arm_name: PicoArmState() for arm_name in self.input_specs
        }
        for state in self.arm_states.values():
            state.reset_raw_gripper_trigger(self._raw_open_trigger_value)

        self._is_connected = False
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._update_thread: threading.Thread | None = None
        self.R_headset_world = R.from_euler(
            euler_sequence, self.cfg.R_headset_world, degrees=True
        ).as_matrix()

    @property
    def _raw_open_trigger_value(self) -> float:
        return 0.0 if self.cfg.trigger_reverse else 1.0

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
        if self.is_connected:
            return

        self._stop_event.clear()
        with self._state_lock:
            for state in self.arm_states.values():
                state.reset_pose()

        self._update_thread = threading.Thread(target=self._run_xr_update_loop, daemon=True)
        self._update_thread.start()
        self._is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.")

    def reset_for_new_episode(self) -> None:
        """Clear accumulated delta targets to avoid replaying previous episode pose."""
        with self._state_lock:
            for state in self.arm_states.values():
                state.reset_pose()
                state.reset_raw_gripper_trigger(self._raw_open_trigger_value)

    def _run_xr_update_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()
                self._update_inputs_from_xr()
                elapsed = time.perf_counter() - start
                time.sleep(max(0, 1 / self.cfg.fps - elapsed))
            except Exception as exc:
                logger.error(f"Error in pose update thread: {exc}")

    def _update_inputs_from_xr(self) -> None:
        with self._state_lock:
            for arm_name, input_spec in self.input_specs.items():
                state = self.arm_states[arm_name]
                self._update_input_from_xr(input_spec, state)

    def _update_input_from_xr(self, input_spec: PicoInputSpec, state: PicoArmState) -> None:
        xr_grip_val = self.xr_client.get_key_value_by_name(input_spec.clutch_button)
        active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

        if active:
            self._sample_raw_gripper_trigger(input_spec, state)
            if not state.was_active:
                state.init_controller_xyz = None
                state.init_controller_quat = None

            xr_pose = self.xr_client.get_pose_by_name(input_spec.pose_source)
            delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, state)
            state.current_delta_xyz = state.base_delta_xyz + delta_xyz
            # Rotvecs cannot be accumulated by vector addition across clutch cycles.
            state.current_delta_rot = (
                R.from_rotvec(delta_rot_angle_axis) * R.from_rotvec(state.base_delta_rot)
            ).as_rotvec()
        elif state.was_active:
            state.base_delta_xyz = state.current_delta_xyz.copy()
            state.base_delta_rot = state.current_delta_rot.copy()
            state.init_controller_xyz = None
            state.init_controller_quat = None

        state.was_active = active

    def _sample_raw_gripper_trigger(
        self, input_spec: PicoInputSpec, state: PicoArmState
    ) -> None:
        state.raw_gripper_trigger = self.xr_client.get_key_value_by_name(
            input_spec.gripper_trigger_source
        )

    def _process_xr_pose(
        self, xr_pose: list[float], state: PicoArmState
    ) -> tuple[np.ndarray, np.ndarray]:
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

        if state.init_controller_xyz is None:
            state.init_controller_xyz = controller_xyz.copy()
            state.init_controller_quat = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.zeros(3)
        else:
            delta_xyz = (
                controller_xyz - state.init_controller_xyz
            ) * self.cfg.xyz_scale_factor
            delta_rot = (
                quat_diff_as_angle_axis(state.init_controller_quat, controller_quat)
                * self.cfg.rot_scale_factor
            )
        return delta_xyz, delta_rot

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._stop_event.set()
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
        self._is_connected = False
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")
