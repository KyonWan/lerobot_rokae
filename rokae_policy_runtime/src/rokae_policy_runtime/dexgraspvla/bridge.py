#!/usr/bin/env python3
"""
Rokae 双臂机器人 <-> DexGraspVLA WebSocket policy server 桥接。

架构边界：
- 本 bridge 采集 Rokae 双臂状态与三路相机：external 裁下半幅后 resize 为 640x480；
  腕部 left_wrist/right_wrist 原分辨率送入策略（不 crop、不 resize）。
- DexGraspVLA 服务端运行在它自己的 conda 环境里，负责 mask / Planner / SAM / Cutie / controller。
- WebSocket 协议与 openpi_client.websocket_client_policy.WebsocketClientPolicy 对齐。
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_rokae.devices.bi_rokae_robot.bi_rokae_robot import BiRokaeRobot
from lerobot_robot_rokae.devices.bi_rokae_robot.config_bi_rokae_robot import BiRokaeRobotConfig
from lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import ControlMode
from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)

POLICY_IMAGE_WIDTH = 640
POLICY_IMAGE_HEIGHT = 480

BRIDGE_LATENCY_FIELDS = [
    "timestamp",
    "step",
    "chunk_index",
    "event",
    "build_observation_ms",
    "robot_get_observation_ms",
    "state_build_ms",
    "image_external_ms",
    "image_left_wrist_ms",
    "image_right_wrist_ms",
    "ws_roundtrip_ms",
    "response_parse_ms",
    "server_infer_ms",
    "server_parse_obs_ms",
    "server_planner_total_ms",
    "server_planner_instruction_ms",
    "server_planner_bbox_ms",
    "server_sam_seed_ms",
    "server_track_total_ms",
    "server_controller_ms",
    "action_shape",
    "execute_action_ms",
    "send_action_ms",
    "loop_elapsed_ms",
    "sleep_ms",
    "mode",
    "control_arm",
    "left_gripper",
    "right_gripper",
]


class CsvLatencyLogger:
    def __init__(self, path: str | Path | None, fieldnames: list[str]) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.fieldnames = fieldnames
        self._file = None
        self._writer: csv.DictWriter | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.path.exists() and self.path.stat().st_size > 0
            self._file = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
            if not exists:
                self._writer.writeheader()
                self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None or self._file is None:
            return
        clean = {key: row.get(key, "") for key in self.fieldnames}
        self._writer.writerow(clean)
        self._file.flush()


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class DexGraspVLABridge:
    """Run DexGraspVLA actions from a WebSocket policy server on a bimanual Rokae robot."""

    GRIPPER_CLOSE_THRESHOLD = 0.5

    def __init__(
        self,
        dexgrasp_host: str = "127.0.0.1",
        dexgrasp_port: int = 8008,
        task_prompt: str = "grasp the target object",
        mode: str = "autonomous",
        max_steps: int = 10000,
        control_frequency: int = 30,
        action_stride: int = 6,
        left_zmq_port: int = 5555,
        right_zmq_port: int = 5556,
        left_zmq_address: str | None = None,
        right_zmq_address: str | None = None,
        cam_high_serial: str | None = None,
        cam_left_wrist_serial: str | None = None,
        cam_right_wrist_serial: str | None = None,
        reset_on_start: bool = True,
        latency_log_file: str | Path | None = None,
        action_dim: int = 8,
        control_arm: str = "left",
    ) -> None:
        self.mode = mode
        self.max_steps = int(max_steps)
        self.control_frequency = int(control_frequency)
        self.dt = 1.0 / float(self.control_frequency)
        self.action_stride = int(max(1, action_stride))
        self.reset_on_start = bool(reset_on_start)
        self.task_prompt = task_prompt
        self.action_dim = int(action_dim)
        self.control_arm = str(control_arm)
        if self.action_dim not in (8, 16):
            raise ValueError(f"action_dim must be 8 or 16, got {self.action_dim}")
        if self.control_arm not in ("left", "right", "dual"):
            raise ValueError(f"control_arm must be left/right/dual, got {self.control_arm}")
        self.latency_logger = CsvLatencyLogger(latency_log_file, BRIDGE_LATENCY_FIELDS)
        if latency_log_file:
            logger.info("Bridge latency CSV: %s", Path(latency_log_file).expanduser().resolve())

        logger.info("Connecting DexGraspVLA WebSocket policy server %s:%s ...", dexgrasp_host, dexgrasp_port)
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=dexgrasp_host,
            port=dexgrasp_port,
        )
        logger.info("DexGraspVLA server metadata: %s", self.policy_client.get_server_metadata())

        cameras = self._build_cameras(
            cam_high_serial=cam_high_serial,
            cam_left_wrist_serial=cam_left_wrist_serial,
            cam_right_wrist_serial=cam_right_wrist_serial,
        )
        required_cameras = {"external", "left_wrist"} if self.action_dim == 8 else {"external", "left_wrist", "right_wrist"}
        missing = required_cameras - set(cameras)
        if missing:
            raise ValueError(
                "DexGraspVLA bridge camera config is incomplete. "
                f"Missing: {sorted(missing)}"
            )

        robot_config = BiRokaeRobotConfig(
            id="bi_rokae_dexgraspvla",
            left_zmq_address=left_zmq_address,
            right_zmq_address=right_zmq_address,
            left_zmq_port=left_zmq_port,
            right_zmq_port=right_zmq_port,
            left_control_mode=ControlMode.JOINT_IMPEDNACE,
            right_control_mode=ControlMode.JOINT_IMPEDNACE,
            control_loop_fps=self.control_frequency,
            cameras=cameras,
        )
        self.robot: BiRokaeRobot = make_robot_from_config(robot_config)
        self.robot.connect()

        self.left_joint_num = self.robot.left_arm.joint_num
        self.right_joint_num = self.robot.right_arm.joint_num

        self.episode_step = 0
        self._left_gripper_cmd: int | None = None
        self._right_gripper_cmd: int | None = None

    def _build_cameras(
        self,
        *,
        cam_high_serial: str | None,
        cam_left_wrist_serial: str | None,
        cam_right_wrist_serial: str | None,
    ) -> dict[str, Any]:
        cameras: dict[str, Any] = {}
        if cam_high_serial:
            if str(cam_high_serial).isdigit():
                cameras["external"] = RealSenseCameraConfig(
                    serial_number_or_name=cam_high_serial,
                    width=640,
                    height=480,
                    fps=60,
                    use_depth=False,
                )
            else:
                cameras["external"] = OrbbecCameraConfig(
                    serial_number_or_index=cam_high_serial,
                    width=640,
                    height=480,
                    fps=60,
                    use_depth=False,
                )
        if cam_left_wrist_serial:
            cameras["left_wrist"] = RealSenseCameraConfig(
                serial_number_or_name=cam_left_wrist_serial,
                width=640,
                height=480,
                fps=60,
                use_depth=False,
            )
        if cam_right_wrist_serial:
            if self.action_dim == 8:
                logger.info(
                    "action_dim=8 (left-arm-only): ignoring --cam_right_wrist_serial=%s",
                    cam_right_wrist_serial,
                )
            else:
                cameras["right_wrist"] = RealSenseCameraConfig(
                    serial_number_or_name=cam_right_wrist_serial,
                    width=640,
                    height=480,
                    fps=60,
                    use_depth=False,
                )
        return cameras

    def _obs_to_state(self, obs: dict[str, Any]) -> np.ndarray:
        left_joints = np.array(
            [obs[f"left_joint_pos{i}"] for i in range(self.left_joint_num)],
            dtype=np.float32,
        )
        left_gripper = np.array([obs["left_gripper_pos"]], dtype=np.float32)
        right_joints = np.array(
            [obs[f"right_joint_pos{i}"] for i in range(self.right_joint_num)],
            dtype=np.float32,
        )
        right_gripper = np.array([obs["right_gripper_pos"]], dtype=np.float32)
        left_state = np.concatenate([left_joints, left_gripper])
        right_state = np.concatenate([right_joints, right_gripper])
        if self.action_dim == 8:
            return left_state if self.control_arm == "left" else right_state
        return np.concatenate([left_joints, left_gripper, right_joints, right_gripper])

    @staticmethod
    def _crop_lower_half(image: np.ndarray) -> np.ndarray:
        h = int(image.shape[0])
        if h < 2:
            return image
        return image[h // 2 :, ...]

    @staticmethod
    def _image_rgb_uint8(image: np.ndarray, key: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Camera {key!r} must be HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _external_image_for_policy(
        self, obs: dict[str, Any], timings: dict[str, Any] | None = None
    ) -> np.ndarray:
        t0 = time.perf_counter()
        image = self._image_rgb_uint8(obs["external"], "external")
        image = self._crop_lower_half(image)
        h, w = image.shape[:2]
        if (w, h) != (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT):
            image = cv2.resize(
                image,
                (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        if timings is not None:
            timings["image_external_ms"] = _ms_since(t0)
        return image

    def _wrist_image_for_policy(
        self, obs: dict[str, Any], key: str, timings: dict[str, Any] | None = None
    ) -> np.ndarray:
        t0 = time.perf_counter()
        image = self._image_rgb_uint8(obs[key], key)
        if timings is not None:
            timings[f"image_{key}_ms"] = _ms_since(t0)
        return image

    def _build_policy_observation(self, timings: dict[str, Any] | None = None) -> dict[str, Any]:
        t_total = time.perf_counter()
        t_obs = time.perf_counter()
        obs = self.robot.get_observation()
        if timings is not None:
            timings["robot_get_observation_ms"] = _ms_since(t_obs)
        t_state = time.perf_counter()
        state = self._obs_to_state(obs)
        if timings is not None:
            timings["state_build_ms"] = _ms_since(t_state)
        observation = {
            "state": state,
            "images": {
                "external": self._external_image_for_policy(obs, timings),
                "left_wrist": self._wrist_image_for_policy(obs, "left_wrist", timings),
            },
            "prompt": self.task_prompt,
            "episode_step": int(self.episode_step),
        }
        if self.action_dim == 16 or "right_wrist" in obs:
            observation["images"]["right_wrist"] = self._wrist_image_for_policy(obs, "right_wrist", timings)
        if timings is not None:
            timings["build_observation_ms"] = _ms_since(t_total)
        return observation

    def _predict_action_chunk(self) -> np.ndarray:
        timings: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "predict_chunk",
            "step": self.episode_step,
            "mode": self.mode,
            "control_arm": self.control_arm,
        }
        observation = self._build_policy_observation(timings)
        t_ws = time.perf_counter()
        response = self.policy_client.infer(observation)
        timings["ws_roundtrip_ms"] = _ms_since(t_ws)
        t_parse = time.perf_counter()
        if "actions" not in response:
            raise KeyError(f"DexGraspVLA response missing 'actions': keys={list(response.keys())}")
        action = np.asarray(response["actions"], dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] < self.action_dim:
            raise ValueError(f"DexGraspVLA returned invalid action shape: {action.shape}")
        action = action[:, : self.action_dim]
        server_latency = response.get("server_latency_ms") or {}
        timings.update(
            {
                "response_parse_ms": _ms_since(t_parse),
                "server_infer_ms": response.get("server_infer_ms", ""),
                "server_parse_obs_ms": server_latency.get("parse_obs_ms", ""),
                "server_planner_total_ms": server_latency.get("planner_total_ms", ""),
                "server_planner_instruction_ms": server_latency.get("planner_instruction_ms", ""),
                "server_planner_bbox_ms": server_latency.get("planner_bbox_ms", ""),
                "server_sam_seed_ms": server_latency.get("sam_seed_ms", ""),
                "server_track_total_ms": server_latency.get("track_total_ms", ""),
                "server_controller_ms": server_latency.get("controller_ms", ""),
                "action_shape": "x".join(str(v) for v in action.shape),
            }
        )
        self.latency_logger.write(timings)
        logger.info(
            (
                "Step %s: obs=%.1fms ws=%.1fms server=%.1fms "
                "planner=%.1fms track=%.1fms controller=%.1fms action_shape=%s first=%s"
            ),
            self.episode_step,
            float(timings.get("build_observation_ms") or 0.0),
            float(timings.get("ws_roundtrip_ms") or 0.0),
            float(timings.get("server_infer_ms") or 0.0),
            float(timings.get("server_planner_total_ms") or 0.0),
            float(timings.get("server_track_total_ms") or 0.0),
            float(timings.get("server_controller_ms") or 0.0),
            action.shape,
            np.array2string(action[0], precision=4, suppress_small=False),
        )
        return action

    def _build_single_arm_action(
        self,
        arm: str,
        target: np.ndarray,
        gripper: int,
    ) -> dict[str, float | int]:
        joint_num = self.left_joint_num if arm == "left" else self.right_joint_num
        prefix = f"{arm}_"
        gripper_attr = f"_{arm}_gripper_cmd"
        prev_gripper = getattr(self, gripper_attr)
        if gripper != prev_gripper:
            setattr(self, gripper_attr, gripper)
            logger.info(
                "%s夹爪状态切换: %s",
                "左" if arm == "left" else "右",
                "闭合(0)" if gripper == 0 else "张开(1)",
            )
        action_dict = {f"joint_pos{i}": float(target[i]) for i in range(joint_num)}
        action_dict["gripper_pos"] = gripper
        return {f"{prefix}{key}": value for key, value in action_dict.items()}

    def execute_action(self, action: np.ndarray) -> dict[str, Any]:
        timings: dict[str, Any] = {}
        action = np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]
        if self.mode == "test":
            logger.info("[TEST] action=%s", np.array2string(action, precision=4))
            timings["mode"] = self.mode
            timings["control_arm"] = self.control_arm
            return timings

        action_dict: dict[str, float | int] = {}
        if self.action_dim == 8:
            target = action[:8]
            gripper = 1 if float(target[7]) >= self.GRIPPER_CLOSE_THRESHOLD else 0
            arm = "right" if self.control_arm == "right" else "left"
            action_dict = self._build_single_arm_action(arm, target, gripper)
            t_send = time.perf_counter()
            if arm == "left":
                self.robot.left_arm.send_action(
                    {key.removeprefix("left_"): value for key, value in action_dict.items()}
                )
            else:
                self.robot.right_arm.send_action(
                    {key.removeprefix("right_"): value for key, value in action_dict.items()}
                )
        else:
            left = action[:8]
            right = action[8:16]
            left_gripper = 1 if float(left[7]) >= self.GRIPPER_CLOSE_THRESHOLD else 0
            right_gripper = 1 if float(right[7]) >= self.GRIPPER_CLOSE_THRESHOLD else 0
            action_dict.update(self._build_single_arm_action("left", left, left_gripper))
            action_dict.update(self._build_single_arm_action("right", right, right_gripper))
            t_send = time.perf_counter()
            self.robot.send_action(action_dict)

        timings.update(
            {
                "send_action_ms": _ms_since(t_send),
                "mode": self.mode,
                "control_arm": self.control_arm,
                "left_gripper": self._left_gripper_cmd,
                "right_gripper": self._right_gripper_cmd,
            }
        )
        return timings

    def run_episode(self) -> None:
        logger.info(
            "Starting DexGraspVLA episode | mode=%s max_steps=%s control_freq=%s action_stride=%s prompt=%r",
            self.mode,
            self.max_steps,
            self.control_frequency,
            self.action_stride,
            self.task_prompt,
        )
        if self.reset_on_start:
            logger.info("Resetting robot position before inference ...")
            self.robot.reset_position()

        self.episode_step = 0
        try:
            while self.episode_step < self.max_steps:
                chunk = self._predict_action_chunk()
                steps = min(self.action_stride, int(chunk.shape[0]))
                for i in range(steps):
                    loop_start = time.perf_counter()
                    t_execute = time.perf_counter()
                    action_timings = self.execute_action(chunk[i])
                    action_timings["execute_action_ms"] = _ms_since(t_execute)
                    self.episode_step += 1
                    loop_elapsed_ms = _ms_since(loop_start)
                    elapsed = loop_elapsed_ms / 1000.0
                    sleep_s = self.dt - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    self.latency_logger.write(
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event": "execute_action",
                            "step": self.episode_step,
                            "chunk_index": i,
                            "loop_elapsed_ms": loop_elapsed_ms,
                            "sleep_ms": max(0.0, sleep_s * 1000.0),
                            **action_timings,
                        }
                    )
                    if self.episode_step >= self.max_steps:
                        break
        except KeyboardInterrupt:
            logger.info("User interrupted DexGraspVLA episode.")
        finally:
            logger.info("DexGraspVLA episode finished after %s steps.", self.episode_step)

    def cleanup(self) -> None:
        logger.info("Disconnecting Rokae robot ...")
        try:
            self.robot.disconnect()
        except Exception as exc:
            logger.warning("Robot disconnect failed, ignored: %s", exc)
