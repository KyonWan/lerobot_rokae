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
from rokae_python_wrapper.joint_fir_filter import SlideWindowFilter
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_SERVER_CART_FIR_WINDOW_SIZE = 50
_SERVER_CONTROL_FPS = 1000.0
_ZERO_DELTA_POSE = np.zeros(6, dtype=np.float64)
_ROT_EPS = 1e-12
_FILTER_SETTLE_EPS = 1e-6


def _continuous_rotvec_for_rotation(rotation: R, previous: np.ndarray) -> np.ndarray:
    """Return a rotvec for ``rotation`` using the branch closest to ``previous``."""
    rotvec = rotation.as_rotvec()
    previous = np.asarray(previous, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(rotvec))
    if norm < _ROT_EPS:
        return rotvec

    axis = rotvec / norm
    projected_prev = float(np.dot(previous, axis))
    k_center = int(round((projected_prev - norm) / (2.0 * np.pi)))
    candidates = [
        rotvec + (2.0 * np.pi * k) * axis
        for k in range(k_center - 2, k_center + 3)
    ]
    return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - previous)))


class PicoDeltaFirFilter:
    """FIR filter for cumulative Pico delta targets.

    Translation is filtered on increments. Rotation input/output is a rotvec target,
    and the filter smooths SO(3) increments using left-multiplied deltas to match
    the downstream flange/base delta application semantics.
    """

    def __init__(self, init_pos: np.ndarray, window_sizes: list[int]) -> None:
        if len(window_sizes) != 2:
            raise ValueError("window_sizes must contain exactly two values")

        init_pos = np.asarray(init_pos, dtype=np.float64).reshape(6)
        self._trans_filter1 = SlideWindowFilter(window_sizes[0])
        self._trans_filter2 = SlideWindowFilter(window_sizes[1])
        self._rot_filter1 = SlideWindowFilter(window_sizes[0])
        self._rot_filter2 = SlideWindowFilter(window_sizes[1])
        self._last_pos_in = init_pos.copy()
        self._last_pos_out = init_pos.copy()
        self._last_pos_out_stage1 = init_pos.copy()
        self._last_R_in = R.from_rotvec(init_pos[3:])
        self._last_R_out = R.from_rotvec(init_pos[3:])
        self._last_R_out_stage1 = R.from_rotvec(init_pos[3:])

    def update(self, delta_pose: np.ndarray) -> np.ndarray:
        delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)

        delta_trans_in = delta_pose[:3] - self._last_pos_in[:3]
        delta_trans_1 = self._trans_filter1.filter(delta_trans_in)
        delta_trans_out = self._trans_filter2.filter(delta_trans_1)
        trans_stage1 = self._last_pos_out_stage1[:3] + delta_trans_1
        trans_out = self._last_pos_out[:3] + delta_trans_out

        R_in = R.from_rotvec(delta_pose[3:])
        rotvec_delta_in = (R_in * self._last_R_in.inv()).as_rotvec()
        rotvec_delta_1 = self._rot_filter1.filter(rotvec_delta_in)
        rotvec_delta_out = self._rot_filter2.filter(rotvec_delta_1)
        R_stage1 = R.from_rotvec(rotvec_delta_1) * self._last_R_out_stage1
        R_out = R.from_rotvec(rotvec_delta_out) * self._last_R_out
        rot_stage1 = _continuous_rotvec_for_rotation(R_stage1, self._last_pos_out_stage1[3:])
        rot_out = _continuous_rotvec_for_rotation(R_out, self._last_pos_out[3:])

        pos_stage1 = np.concatenate([trans_stage1, rot_stage1])
        pos_out = np.concatenate([trans_out, rot_out])

        self._last_pos_in = delta_pose.copy()
        self._last_pos_out = pos_out.copy()
        self._last_pos_out_stage1 = pos_stage1.copy()
        self._last_R_in = R_in
        self._last_R_out = R_out
        self._last_R_out_stage1 = R_stage1
        return pos_out

    def reset(self, init_pos: np.ndarray) -> None:
        init_pos = np.asarray(init_pos, dtype=np.float64).reshape(6)
        self._trans_filter1.reset()
        self._trans_filter2.reset()
        self._rot_filter1.reset()
        self._rot_filter2.reset()
        self._last_pos_in = init_pos.copy()
        self._last_pos_out = init_pos.copy()
        self._last_pos_out_stage1 = init_pos.copy()
        self._last_R_in = R.from_rotvec(init_pos[3:])
        self._last_R_out = R.from_rotvec(init_pos[3:])
        self._last_R_out_stage1 = R.from_rotvec(init_pos[3:])


@dataclass(frozen=True)
class PicoInputSpec:
    pose_source: str
    clutch_button: str
    gripper_trigger_source: str
    action_prefix: str = ""


@dataclass
class PicoArmState:
    prev_controller_xyz: np.ndarray | None = None
    prev_controller_quat: np.ndarray | None = None
    raw_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_delta_rotation: R = field(default_factory=R.identity)
    current_delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_step_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_step_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    filter_settling: bool = False
    was_active: bool = False
    raw_gripper_trigger: float = 0.0
    delta_filter: PicoDeltaFirFilter | None = None

    def configure_delta_filter(self, window_sizes: list[int] | None) -> None:
        self.delta_filter = (
            None
            if window_sizes is None
            else PicoDeltaFirFilter(init_pos=_ZERO_DELTA_POSE, window_sizes=window_sizes)
        )

    def reset_pose(self) -> None:
        self.prev_controller_xyz = None
        self.prev_controller_quat = None
        self.raw_delta_xyz = np.zeros(3)
        self.raw_delta_rot = np.zeros(3)
        self.raw_delta_rotation = R.identity()
        self.current_delta_xyz = np.zeros(3)
        self.current_delta_rot = np.zeros(3)
        self.last_step_xyz = np.zeros(3)
        self.last_step_rot = np.zeros(3)
        self.filter_settling = False
        self.was_active = False
        if self.delta_filter is not None:
            self.delta_filter.reset(_ZERO_DELTA_POSE)

    def reset_raw_gripper_trigger(self, raw_open_value: float) -> None:
        self.raw_gripper_trigger = raw_open_value

    def integrate_raw_step(self, step_xyz: np.ndarray, step_rot: np.ndarray) -> None:
        self.last_step_xyz = np.asarray(step_xyz, dtype=np.float64).reshape(3).copy()
        self.last_step_rot = np.asarray(step_rot, dtype=np.float64).reshape(3).copy()
        self.raw_delta_xyz = self.raw_delta_xyz + self.last_step_xyz
        self.raw_delta_rotation = R.from_rotvec(self.last_step_rot) * self.raw_delta_rotation
        self.raw_delta_rot = _continuous_rotvec_for_rotation(
            self.raw_delta_rotation, self.raw_delta_rot
        )

    def update_current_delta_from_raw(self) -> None:
        raw_delta_pose = np.concatenate([self.raw_delta_xyz, self.raw_delta_rot])
        if self.delta_filter is None:
            filtered_delta_pose = raw_delta_pose
        else:
            filtered_delta_pose = self.delta_filter.update(raw_delta_pose)
        self.current_delta_xyz = filtered_delta_pose[:3].copy()
        self.current_delta_rot = filtered_delta_pose[3:].copy()
        self.filter_settling = self._filter_error_norm() > _FILTER_SETTLE_EPS

    def _filter_error_norm(self) -> float:
        trans_err = float(np.linalg.norm(self.raw_delta_xyz - self.current_delta_xyz))
        rot_err = float(
            (
                R.from_rotvec(self.current_delta_rot).inv()
                * R.from_rotvec(self.raw_delta_rot)
            ).magnitude()
        )
        return max(trans_err, rot_err)


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
            xr_pose = self.xr_client.get_pose_by_name(input_spec.pose_source)
            controller_xyz, controller_quat = self._transform_xr_pose(xr_pose)
            if state.prev_controller_xyz is None or state.prev_controller_quat is None:
                state.prev_controller_xyz = controller_xyz.copy()
                state.prev_controller_quat = controller_quat.copy()
                state.last_step_xyz = np.zeros(3)
                state.last_step_rot = np.zeros(3)
            else:
                step_xyz = (
                    controller_xyz - state.prev_controller_xyz
                ) * self.cfg.xyz_scale_factor
                step_rot = (
                    quat_diff_as_angle_axis(state.prev_controller_quat, controller_quat)
                    * self.cfg.rot_scale_factor
                )
                state.integrate_raw_step(step_xyz, step_rot)
                state.prev_controller_xyz = controller_xyz.copy()
                state.prev_controller_quat = controller_quat.copy()
            state.update_current_delta_from_raw()
        elif state.was_active:
            state.prev_controller_xyz = None
            state.prev_controller_quat = None
            state.last_step_xyz = np.zeros(3)
            state.last_step_rot = np.zeros(3)
            state.update_current_delta_from_raw()
        else:
            state.last_step_xyz = np.zeros(3)
            state.last_step_rot = np.zeros(3)
            state.update_current_delta_from_raw()

        state.was_active = active

    def _sample_raw_gripper_trigger(
        self, input_spec: PicoInputSpec, state: PicoArmState
    ) -> None:
        state.raw_gripper_trigger = self.xr_client.get_key_value_by_name(
            input_spec.gripper_trigger_source
        )

    def _transform_xr_pose(self, xr_pose: list[float]) -> tuple[np.ndarray, np.ndarray]:
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
        return controller_xyz, controller_quat

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
