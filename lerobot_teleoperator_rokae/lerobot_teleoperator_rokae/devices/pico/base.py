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
from rokae_python_wrapper.cart_fir_filter import CartPosFirFilter
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SERVER_CART_FIR_WINDOW_SIZE = 50
_SERVER_CONTROL_FPS = 1000.0
_ZERO_DELTA_POSE = np.zeros(6, dtype=np.float64)


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
    raw_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    was_active: bool = False
    raw_gripper_trigger: float = 0.0
    delta_filter: CartPosFirFilter | None = None

    def configure_delta_filter(self, window_sizes: list[int] | None) -> None:
        self.delta_filter = (
            None
            if window_sizes is None
            else CartPosFirFilter(init_pos=_ZERO_DELTA_POSE, window_sizes=window_sizes)
        )

    def reset_pose(self) -> None:
        self.init_controller_xyz = None
        self.init_controller_quat = None
        self.raw_delta_xyz = np.zeros(3)
        self.raw_delta_rot = np.zeros(3)
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.base_delta_xyz = np.zeros(3)
        self.base_delta_rot = np.zeros(3)
        self.was_active = False
        if self.delta_filter is not None:
            self.delta_filter.reset(_ZERO_DELTA_POSE)

    def reset_raw_gripper_trigger(self, raw_open_value: float) -> None:
        self.raw_gripper_trigger = raw_open_value

    def update_current_delta(self, raw_delta_xyz: np.ndarray, raw_delta_rot: np.ndarray) -> None:
        self.raw_delta_xyz = raw_delta_xyz.copy()
        self.raw_delta_rot = raw_delta_rot.copy()
        raw_delta_pose = np.concatenate([raw_delta_xyz, raw_delta_rot])
        if self.delta_filter is None:
            filtered_delta_pose = raw_delta_pose
        else:
            filtered_delta_pose = self.delta_filter.update(raw_delta_pose)
        self.current_delta_xyz = filtered_delta_pose[:3].copy()
        self.current_delta_rot = filtered_delta_pose[3:].copy()

    def sync_filter_to_current_delta(self) -> None:
        if self.delta_filter is None:
            return
        current_delta_pose = np.concatenate([self.current_delta_xyz, self.current_delta_rot])
        self.delta_filter.reset(current_delta_pose)


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
        self._filter_enabled = bool(getattr(self.cfg, "filter_enabled", True))
        self._filter_window_sizes = (
            self._resolve_filter_window_sizes() if self._filter_enabled else None
        )
        self.input_specs = dict(input_specs)
        self.arm_states = {
            arm_name: PicoArmState() for arm_name in self.input_specs
        }
        for state in self.arm_states.values():
            state.configure_delta_filter(self._filter_window_sizes)
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

    def _resolve_filter_window_sizes(self) -> list[int]:
        configured_window_sizes = getattr(self.cfg, "filter_window_sizes", None)
        if configured_window_sizes is None:
            fps = float(getattr(self.cfg, "fps", 60.0))
            scaled_window_size = max(
                1,
                int(round(_SERVER_CART_FIR_WINDOW_SIZE * fps / _SERVER_CONTROL_FPS)),
            )
            window_sizes = [scaled_window_size, scaled_window_size]
        else:
            window_sizes = [int(size) for size in configured_window_sizes]

        if len(window_sizes) != 2 or any(size <= 0 for size in window_sizes):
            raise ValueError(
                "filter_window_sizes must contain exactly two positive integers"
            )
        return window_sizes

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
                time.sleep(max(0.0, 1.0 / self.cfg.fps - elapsed))
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
            raw_delta_xyz = state.base_delta_xyz + delta_xyz
            # Rotvecs cannot be accumulated by vector addition across clutch cycles.
            raw_delta_rot = (
                R.from_rotvec(delta_rot_angle_axis) * R.from_rotvec(state.base_delta_rot)
            ).as_rotvec()
            state.update_current_delta(raw_delta_xyz, raw_delta_rot)
        elif state.was_active:
            state.base_delta_xyz = state.current_delta_xyz.copy()
            state.base_delta_rot = state.current_delta_rot.copy()
            state.sync_filter_to_current_delta()
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
