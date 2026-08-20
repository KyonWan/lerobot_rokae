#!/usr/bin/env python3
"""
Rokae 双臂机器人 <-> OpenPI 策略服务器桥接（三相机版）

负责处理：
1. 从双臂机械臂采集观测数据（左右关节位置、三路图像）
2. 通过 WebSocket 将观测发送至策略服务器
3. 接收动作预测
4. 在双臂机械臂上执行动作
"""

import concurrent.futures
import math
import threading
import time
import logging
import select
import sys
import termios
import tty
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_rokae.devices.bi_rokae_robot.config_bi_rokae_robot import (
    BiRokaeRobotConfig,
)
from lerobot_robot_rokae.devices.bi_rokae_robot.bi_rokae_robot import BiRokaeRobot
from lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import (
    ControlMode,
)
from openpi_client import websocket_client_policy

if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BiOpenPIPolicyBridge:
    """
    双臂 Rokae 机器人（左/右各 7 轴 + 夹爪）与 OpenPI 策略服务器之间的桥接。

    三相机配置：
      - external (顶部外部相机，cam_high_serial)
      - left_wrist (左腕相机，cam_left_wrist_serial)
      - right_wrist (右腕相机，cam_right_wrist_serial)

    状态向量（16 维）：
      [left_joint_pos0..6, left_gripper_pos, right_joint_pos0..6, right_gripper_pos]

    动作向量（16 维，与状态同阶）：
      [left_joint_pos0..6, left_gripper_pos, right_joint_pos0..6, right_gripper_pos]
    """

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        mode: str = "autonomous",
        max_steps: int = 1000,
        left_zmq_port: int = 5555,
        right_zmq_port: int = 5556,
        left_zmq_address: str | None = None,
        right_zmq_address: str | None = None,
        left_control_mode: ControlMode = ControlMode.JOINT_IMPEDNACE,
        right_control_mode: ControlMode = ControlMode.JOINT_IMPEDNACE,
        cam_high_serial: str | None = None,
        cam_left_wrist_serial: str | None = None,
        cam_right_wrist_serial: str | None = None,
        action_chunk_size: int = 50,
        rate_of_inference: int | None = 30,
        rtc_delay: int | None = None,
        dynamic_delay: bool = True,
        delay_window: int = 8,
        delay_safety_steps: int = 1,
        temporal_ensemble_coefficient: float | None = None,
        save_rgb_dir: str | None = None,
        debug_model_io: bool = False,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.mode = mode

        logger.info("正在连接策略服务器 %s:%s ...", policy_server_host, policy_server_port)
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        self.server_metadata = self.policy_client.get_server_metadata()
        logger.info("OpenPI server metadata: %s", self.server_metadata)

        metadata_rtc_enabled = bool(
            self.server_metadata.get("rtc_enable", self.server_metadata.get("rtc_enabled", False))
        )
        self.server_rtc_enabled = metadata_rtc_enabled
        self.supports_reset_rpc = bool(self.server_metadata.get("supports_reset_rpc", False))
        self.supports_dynamic_rtc_delay = bool(self.server_metadata.get("supports_dynamic_rtc_delay", False))
        metadata_delay = int(self.server_metadata.get("max_delay", self.server_metadata.get("rtc_delay", 0)) or 0)
        metadata_rate = int(
            self.server_metadata.get("rtc_rate_of_inference", self.server_metadata.get("rate_of_inference", 30)) or 30
        )

        self.max_rtc_delay = int(metadata_delay if rtc_delay is None else rtc_delay)
        self.rtc_delay = self.max_rtc_delay
        self.rate_of_inference: int = int(metadata_rate if rate_of_inference is None else rate_of_inference)
        self.rtc_enabled = (metadata_rtc_enabled or rtc_delay is not None) and self.max_rtc_delay > 0
        self.dynamic_delay = bool(dynamic_delay)
        self.delay_safety_steps = int(delay_safety_steps)
        self.inference_latencies_s = deque(maxlen=max(int(delay_window), 1))
        self._delay_clamp_warned = False

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future_action_chunk: concurrent.futures.Future | None = None
        self.future_request_step: int | None = None
        self.future_request_delay: int | None = None
        self.future_request_started_at: float | None = None
        self.future_request_observation: dict[str, Any] | None = None
        self.next_action_chunk: np.ndarray | None = None
        self.next_request_step: int | None = None
        self.next_request_delay: int | None = None
        self.chunk_ready_event = threading.Event()

        if self.rate_of_inference <= 0:
            raise ValueError(f"rate_of_inference 必须为正数，当前为 {self.rate_of_inference}")
        if action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size 必须为正数，当前为 {action_chunk_size}")
        if self.delay_safety_steps < 0:
            raise ValueError(f"delay_safety_steps 必须为非负数，当前为 {self.delay_safety_steps}")
        if self.rtc_enabled and self.rate_of_inference + self.max_rtc_delay >= action_chunk_size:
            logger.warning(
                "RTC 最大切换点 rate_of_inference + max_rtc_delay = %s 接近或超过 action_chunk_size = %s；"
                "若推理未及时返回，可能耗尽旧 chunk。",
                self.rate_of_inference + self.max_rtc_delay,
                action_chunk_size,
            )
        if self.max_rtc_delay > 0 and not metadata_rtc_enabled:
            logger.warning(
                "机器人端设置了 rtc_delay=%s，但策略服务器 metadata 未声明 rtc_enable=True。"
                "若服务器未启用 RTC，机器人端只能异步切换，无法强制服务器使用 action_prefix。",
                self.max_rtc_delay,
            )
        if self.server_rtc_enabled and not self.supports_reset_rpc:
            logger.warning(
                "OpenPI server metadata 未声明 supports_reset_rpc=True；"
                "每次开始推理前仍会尝试 reset，若失败请更新 OpenPI 代码或重启策略服务器。"
            )
        if self.rtc_enabled and self.dynamic_delay and not self.supports_dynamic_rtc_delay:
            logger.warning(
                "OpenPI server metadata 未声明 supports_dynamic_rtc_delay=True；"
                "仍会发送动态 delay 字段，若 server 代码未更新则会退化为固定 max_delay。"
            )

        # ---- 相机配置 ----
        # 顶部 external 常为 Orbbec（序列号含字母，如 CP2G…）；腕部一般为 RealSense（纯数字序列号）
        cameras: dict = {}
        if cam_high_serial:
            if cam_high_serial.isdigit():
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
            cameras["right_wrist"] = RealSenseCameraConfig(
                serial_number_or_name=cam_right_wrist_serial,
                width=640,
                height=480,
                fps=60,
                use_depth=False,
            )

        if not cameras:
            raise ValueError(
                "至少需要配置一个相机。请通过 --cam_high_serial、"
                "--cam_left_wrist_serial 或 --cam_right_wrist_serial 指定相机序列号。"
            )

        # ---- 双臂机器人配置 ----
        robot_config = BiRokaeRobotConfig(
            id="bi_rokae",
            left_zmq_address=left_zmq_address,
            right_zmq_address=right_zmq_address,
            left_zmq_port=left_zmq_port,
            right_zmq_port=right_zmq_port,
            left_control_mode=left_control_mode,
            right_control_mode=right_control_mode,
            control_loop_fps=control_frequency,
            cameras=cameras,
        )
        self.robot: BiRokaeRobot = make_robot_from_config(robot_config)
        self.robot.connect()
        self.robot.reset_position()

        self.left_joint_num: int = self.robot.left_arm.joint_num
        self.right_joint_num: int = self.robot.right_arm.joint_num
        # 动作维度：左臂 (joint_num + gripper) + 右臂 (joint_num + gripper)
        self.left_action_dim: int = self.left_joint_num + 1
        self.right_action_dim: int = self.right_joint_num + 1
        self.action_dim: int = self.left_action_dim + self.right_action_dim  # 16

        self.action_chunk_size: int = action_chunk_size
        self.temporal_ensemble_coefficient: float | None = temporal_ensemble_coefficient

        self.current_action_chunk: np.ndarray | None = None
        self.action_chunk_idx: int = 0
        self.episode_step: int = 0
        self.is_running: bool = False

        self.action_buffer: defaultdict[int, list] = defaultdict(list)
        self.action_buffer_size: int = (
            self.max_steps + self.action_chunk_size + self.rate_of_inference + self.max_rtc_delay
        )

        # 夹爪状态缓存（避免重复发送相同指令）
        self._left_gripper_cmd: int | None = None
        self._right_gripper_cmd: int | None = None

        self._save_rgb_dir: Path | None = Path(save_rgb_dir).expanduser().resolve() if save_rgb_dir else None
        if self._save_rgb_dir is not None:
            self._save_rgb_dir.mkdir(parents=True, exist_ok=True)
            logger.info("发送给 OpenPI 的 RGB 图像将保存到: %s", self._save_rgb_dir)

        self.debug_model_io: bool = debug_model_io
        self._keyboard_fd: int | None = None
        self._keyboard_old_attrs: list[Any] | None = None
        self._escape_buffer: str = ""

    # ------------------------------------------------------------------
    # 观测处理
    # ------------------------------------------------------------------

    def _obs_to_state(self, obs: dict) -> np.ndarray:
        """将观测字典转为策略状态向量（16 维）。"""
        left_joints = np.array(
            [obs[f"left_joint_pos{i}"] for i in range(self.left_joint_num)], dtype=np.float32
        )
        left_gripper = np.array([obs["left_gripper_pos"]], dtype=np.float32)
        right_joints = np.array(
            [obs[f"right_joint_pos{i}"] for i in range(self.right_joint_num)], dtype=np.float32
        )
        right_gripper = np.array([obs["right_gripper_pos"]], dtype=np.float32)
        return np.concatenate([left_joints, left_gripper, right_joints, right_gripper])

    def _log_policy_exchange(self, observation: dict[str, Any], actions: Any) -> None:
        """打印本次 infer 送入策略的观测与返回的 action chunk，便于对照训练/服务端约定。"""
        step = self.episode_step
        state = np.asarray(observation["state"], dtype=np.float64)
        logger.info(
            "[→ OpenPI] step=%s | state shape=%s dtype=%s | min=%.6g max=%.6g mean=%.6g | any_nan=%s",
            step,
            state.shape,
            state.dtype,
            float(np.nanmin(state)),
            float(np.nanmax(state)),
            float(np.nanmean(state)),
            bool(np.any(np.isnan(state))),
        )
        if state.size >= 8:
            logger.info(
                "[→ OpenPI] step=%s | state[0:8] 左臂7关节+左夹爪: %s",
                step,
                np.array2string(state[:8], precision=5, suppress_small=False),
            )
        if state.size >= 16:
            logger.info(
                "[→ OpenPI] step=%s | state[8:16] 右臂7关节+右夹爪: %s",
                step,
                np.array2string(state[8:16], precision=5, suppress_small=False),
            )

        images = observation.get("images", {})
        logger.info("[→ OpenPI] step=%s | images 键顺序（需与训练一致）: %s", step, list(images.keys()))
        for name, arr in images.items():
            x = np.asarray(arr)
            logger.info(
                "[→ OpenPI] step=%s | images[%r] shape=%s dtype=%s min=%s max=%s",
                step,
                name,
                x.shape,
                x.dtype,
                x.min(),
                x.max(),
            )

        pr = observation.get("prompt", "")
        prev = pr if len(pr) <= 240 else pr[:240] + "…"
        logger.info("[→ OpenPI] step=%s | prompt 长度=%d: %s", step, len(pr), prev)
        runtime = observation.get("__openpi_runtime__")
        if runtime:
            logger.info("[→ OpenPI] step=%s | RTC runtime: %s", step, runtime)

        act = np.asarray(actions, dtype=np.float64)
        logger.info(
            "[← OpenPI] step=%s | actions shape=%s dtype=%s | 期望末维=%d (左7+左爪+右7+右爪)",
            step,
            act.shape,
            act.dtype,
            self.action_dim,
        )
        if act.size == 0:
            return
        row0 = np.asarray(act[0]).reshape(-1)
        logger.info(
            "[← OpenPI] step=%s | chunk[0] 左7+Lg 右7+Rg: %s",
            step,
            np.array2string(row0, precision=5, suppress_small=False),
        )
        if act.ndim >= 2 and act.shape[0] >= 2:
            a2 = act.reshape(act.shape[0], -1)
            diffs = np.abs(a2[1:] - a2[:-1])
            t_std = float(np.std(a2, axis=0).mean())
            logger.info(
                "[← OpenPI] step=%s | chunk 沿时间维: 相邻步 max|Δ|=%.6g mean|Δ|=%.6g 各维在时间维上 std 的均值=%.6g",
                step,
                float(diffs.max()),
                float(diffs.mean()),
                t_std,
            )
            if bool(np.allclose(a2, a2[0:1], rtol=1e-5, atol=1e-6)):
                logger.warning(
                    "[← OpenPI] step=%s | chunk 内各行几乎相同 — 模型在整段 horizon 上输出了常数轨迹；"
                    "若观测在变化仍如此，需查服务端/训练；若观测不变，见确定性策略说明。",
                    step,
                )
        if self.debug_model_io and act.ndim >= 2 and act.shape[0] > 1:
            for i in range(1, min(3, int(act.shape[0]))):
                ri = np.asarray(act[i]).reshape(-1)
                logger.info(
                    "[← OpenPI] step=%s | chunk[%d]: %s",
                    step,
                    i,
                    np.array2string(ri, precision=5, suppress_small=False),
                )

    def _preprocess_images(self, obs: dict) -> dict[str, np.ndarray]:
        """返回与训练数据一致的原始 RGB HWC 图像，由 OpenPI 服务端负责 resize_with_pad。"""
        cam_images: dict[str, np.ndarray] = {}
        for cam_key in self.robot.cameras:
            image = np.asarray(obs[cam_key])
            if image.ndim != 3:
                raise ValueError(f"相机 {cam_key} 图像维度错误: got shape={image.shape}, expected HWC or CHW")

            if image.shape[-1] in (3, 4):
                image_rgb = image[..., :3]
            elif image.shape[0] in (3, 4):
                image_rgb = np.transpose(image[:3], (1, 2, 0))
            else:
                raise ValueError(f"相机 {cam_key} 图像通道数错误: got shape={image.shape}")

            if image_rgb.dtype != np.uint8:
                if np.issubdtype(image_rgb.dtype, np.floating):
                    image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
                else:
                    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

            if self._save_rgb_dir is not None:
                out_path = self._save_rgb_dir / f"{cam_key}_step{self.episode_step:06d}.png"
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(out_path), bgr):
                    logger.warning("保存 RGB 图像失败: %s", out_path)

            cam_images[cam_key] = image_rgb

        return cam_images

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()

    def _clamp_rtc_delay(self, delay: int) -> int:
        if not self.rtc_enabled:
            return 0
        return max(0, min(int(delay), self.max_rtc_delay))

    def _estimate_rtc_delay(self) -> int:
        if not self.rtc_enabled:
            return 0
        if not self.dynamic_delay or len(self.inference_latencies_s) == 0:
            return self._clamp_rtc_delay(self.rtc_delay)

        estimated_steps = math.ceil(max(self.inference_latencies_s) / self.dt) + self.delay_safety_steps
        return self._clamp_rtc_delay(estimated_steps)

    def _record_inference_latency(self, latency_s: float) -> int:
        self.inference_latencies_s.append(float(latency_s))
        measured_steps = max(0, math.ceil(latency_s / self.dt))
        if self.rtc_enabled and measured_steps > self.max_rtc_delay and not self._delay_clamp_warned:
            logger.warning(
                "实测推理延迟 %s 步超过 max_rtc_delay=%s；OpenPI RTC 约束会按上限发送，"
                "机器人端若晚到会跳过新 chunk 更靠后的动作以保持时间轴对齐。",
                measured_steps,
                self.max_rtc_delay,
            )
            self._delay_clamp_warned = True
        return measured_steps

    def _make_observation(
        self,
        task_prompt: str,
        rtc_delay: int | None = None,
        include_current_action_prefix: bool = False,
    ) -> dict:
        obs = self.robot.get_observation()
        state = self._obs_to_state(obs)
        images = self._preprocess_images(obs)
        observation = {"state": state, "images": images, "prompt": task_prompt}
        if include_current_action_prefix:
            observation["actions"] = np.repeat(state[None, :], self.action_chunk_size, axis=0).astype(np.float32)
        if self.rtc_enabled:
            observation["__openpi_runtime__"] = {
                "rtc_delay": self._clamp_rtc_delay(self.rtc_delay if rtc_delay is None else rtc_delay),
                "rtc_rate_of_inference": self.rate_of_inference,
                "dynamic_delay": self.dynamic_delay,
            }
        return observation

    def _normalize_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError(f"策略返回动作 chunk 维度错误: got shape={chunk.shape}, expected=(H, D)")
        if chunk.shape[1] < self.action_dim:
            raise ValueError(f"策略动作维度不足: got={chunk.shape[1]}, expected>={self.action_dim}")
        if chunk.shape[0] != self.action_chunk_size:
            logger.warning(
                "策略返回 chunk 长度 %s 与配置 action_chunk_size=%s 不一致，后续按返回长度执行。",
                chunk.shape[0],
                self.action_chunk_size,
            )
        return chunk

    def _store_action_candidates(self, chunk: np.ndarray, chunk_origin_step: int, start_idx: int = 0) -> None:
        for k in range(start_idx, len(chunk)):
            future_t = chunk_origin_step + k
            if future_t < self.action_buffer_size:
                self.action_buffer[future_t].append(chunk[k])

    def _request_chunk_sync(self, task_prompt: str) -> np.ndarray:
        request_delay = self._estimate_rtc_delay()
        observation = self._make_observation(
            task_prompt,
            rtc_delay=request_delay,
            include_current_action_prefix=self.rtc_enabled,
        )
        logger.info(
            "Step %s: 同步请求初始 OpenPI chunk，request_delay=%s，dynamic_delay=%s，current_action_prefix=%s。",
            self.episode_step,
            request_delay,
            self.dynamic_delay,
            self.rtc_enabled,
        )
        response = self.policy_client.infer(observation)
        chunk = self._normalize_action_chunk(response["actions"])
        self._log_policy_exchange(observation, chunk)
        prefix_steps = self._clamp_rtc_delay(request_delay)
        if self.rtc_enabled and prefix_steps > 0:
            chunk = chunk.copy()
            chunk[:prefix_steps, : self.action_dim] = observation["state"][: self.action_dim]
            logger.info("初始 chunk 前 %s 步使用当前动作作为 RTC prefix。", prefix_steps)
        return chunk

    def _reset_policy_state(self) -> None:
        if not self.server_rtc_enabled:
            return

        try:
            self.policy_client.reset()
        except Exception as e:
            raise RuntimeError(
                "无法重置 OpenPI server 端 RTC action_prefix。"
                "为避免跨 episode 或暂停后使用旧 chunk，请更新 OpenPI reset RPC 或重启策略服务器。"
            ) from e

        logger.info("OpenPI RTC 策略状态已重置，下一次 infer 将不使用旧 action_prefix。")

    def _clear_policy_runtime_state(self) -> None:
        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.next_action_chunk = None
        self.next_request_step = None
        self.next_request_delay = None
        self.future_action_chunk = None
        self.future_request_step = None
        self.future_request_delay = None
        self.future_request_started_at = None
        self.future_request_observation = None
        self.chunk_ready_event.clear()
        self.action_buffer.clear()

    def _finish_pending_infer(self, reason: str) -> None:
        if self.future_action_chunk is None:
            return

        logger.info("等待正在进行的 OpenPI 异步推理结束后再%s ...", reason)
        future = self.future_action_chunk
        try:
            future.result()
        except Exception as e:
            logger.warning("等待异步推理结束时收到错误（已忽略，将重置策略状态）: %s", e)
        finally:
            self.future_action_chunk = None
            self.future_request_step = None
            self.future_request_delay = None
            self.future_request_started_at = None
            self.future_request_observation = None
            self.chunk_ready_event.clear()

    def _async_infer(self, observation: dict) -> dict | None:
        try:
            return self.policy_client.infer(observation)
        except Exception as e:
            logger.error("异步推理失败: %s", e)
            return None

    def _submit_async_infer(self, task_prompt: str) -> None:
        if self.future_action_chunk is not None or self.next_action_chunk is not None:
            return

        request_step = self.episode_step
        request_delay = self._estimate_rtc_delay()
        observation = self._make_observation(task_prompt, rtc_delay=request_delay)
        self.future_request_step = request_step
        self.future_request_delay = request_delay
        self.future_request_started_at = time.perf_counter()
        self.future_request_observation = observation
        self.future_action_chunk = self.executor.submit(self._async_infer, observation)
        self.chunk_ready_event.clear()
        logger.info(
            "Step %s: 发起 OpenPI RTC 异步推理，当前 chunk idx=%s，rate_of_inference=%s，"
            "request_delay=%s，dynamic_delay=%s。",
            self.episode_step,
            self.action_chunk_idx,
            self.rate_of_inference,
            request_delay,
            self.dynamic_delay,
        )

    def _collect_async_result(self) -> None:
        if self.future_action_chunk is None or not self.future_action_chunk.done():
            return

        request_step = self.future_request_step
        request_delay = self.future_request_delay
        request_started_at = self.future_request_started_at
        observation = self.future_request_observation
        future = self.future_action_chunk
        self.future_action_chunk = None
        self.future_request_step = None
        self.future_request_delay = None
        self.future_request_started_at = None
        self.future_request_observation = None

        try:
            response = future.result()
        except Exception as e:
            logger.error("获取异步推理结果失败: %s", e)
            return

        if response is None:
            logger.error("异步推理返回空结果，将继续使用当前 chunk 并尝试重新请求。")
            return

        if request_step is None:
            request_step = self.episode_step
        if request_delay is None:
            request_delay = self._estimate_rtc_delay()
        measured_delay = None
        if request_started_at is not None:
            measured_delay = self._record_inference_latency(time.perf_counter() - request_started_at)

        self.next_action_chunk = self._normalize_action_chunk(response["actions"])
        self.next_request_step = request_step
        self.next_request_delay = self._clamp_rtc_delay(request_delay)
        self.chunk_ready_event.set()

        start_idx = self.next_request_delay if self.rtc_enabled else 0
        self._store_action_candidates(self.next_action_chunk, request_step, start_idx=start_idx)
        if observation is not None:
            self._log_policy_exchange(observation, self.next_action_chunk)
        logger.info(
            "Step %s: 异步推理完成，收到新动作块 shape=%s，对齐原点 step=%s，"
            "request_delay=%s，measured_delay=%s，执行起点 idx=%s。",
            self.episode_step,
            self.next_action_chunk.shape,
            request_step,
            self.next_request_delay,
            measured_delay,
            start_idx,
        )

    def _switch_point(self, request_delay: int | None = None) -> int:
        if self.rtc_enabled:
            delay = self.rtc_delay if request_delay is None else request_delay
            return self.rate_of_inference + self._clamp_rtc_delay(delay)
        return self.rate_of_inference

    def _next_start_idx(self, current_idx: int, request_delay: int | None = None) -> int:
        switch_point = self._switch_point(request_delay)
        late_steps = max(current_idx - switch_point, 0)
        if self.rtc_enabled:
            delay = self.rtc_delay if request_delay is None else request_delay
            return self._clamp_rtc_delay(delay) + late_steps
        return late_steps

    def _switch_to_next_chunk(self) -> bool:
        if self.current_action_chunk is None or self.next_action_chunk is None:
            return False

        request_delay = self.next_request_delay
        if self.action_chunk_idx < self._switch_point(request_delay):
            return False

        start_idx = self._next_start_idx(self.action_chunk_idx, request_delay)
        if start_idx >= len(self.next_action_chunk):
            logger.error(
                "新 chunk 已经过期: start_idx=%s, len=%s。保留当前 chunk，等待下一次请求。",
                start_idx,
                len(self.next_action_chunk),
            )
            self.next_action_chunk = None
            self.next_request_step = None
            self.next_request_delay = None
            self.chunk_ready_event.clear()
            return False

        self.current_action_chunk = self.next_action_chunk
        self.next_action_chunk = None
        self.next_request_step = None
        self.next_request_delay = None
        self.chunk_ready_event.clear()
        self.action_chunk_idx = start_idx
        logger.info(
            "Step %s: 切换到新动作块，从 idx=%s 开始执行（switch_point=%s）。",
            self.episode_step,
            start_idx,
            self._switch_point(request_delay),
        )
        return True

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------

    def execute_action(self, action: np.ndarray) -> None:
        """
        执行一步动作。

        action 布局：
          [left_joint_pos0..6 (7), left_gripper (1),
           right_joint_pos0..6 (7), right_gripper (1)]
        """
        if self.mode == "test":
            logger.info("[TEST] 模拟执行动作: %s", action)
            return

        left_action = action[: self.left_action_dim]   # 前 8 维
        right_action = action[self.left_action_dim :]  # 后 8 维

        # ---- 左夹爪 ----
        raw_left_gripper = float(left_action[self.left_joint_num])
        left_gripper_cmd: int | None = 1 if raw_left_gripper >= 0.7 else 0
        if left_gripper_cmd == self._left_gripper_cmd:
            left_gripper_cmd = None
        else:
            self._left_gripper_cmd = left_gripper_cmd
            logger.info("左夹爪状态切换: %s", "闭合(0)" if left_gripper_cmd == 0 else "张开(1)")

        # ---- 右夹爪 ----
        raw_right_gripper = float(right_action[self.right_joint_num])
        right_gripper_cmd: int | None = 1 if raw_right_gripper >= 0.7 else 0
        if right_gripper_cmd == self._right_gripper_cmd:
            right_gripper_cmd = None
        else:
            self._right_gripper_cmd = right_gripper_cmd
            logger.info("右夹爪状态切换: %s", "闭合(0)" if right_gripper_cmd == 0 else "张开(1)")

        # ---- 构造动作字典 ----
        action_dict: dict = {}

        # 左臂关节
        for i in range(self.left_joint_num):
            action_dict[f"left_joint_pos{i}"] = float(left_action[i])
        action_dict["left_gripper_pos"] = left_gripper_cmd if left_gripper_cmd is not None else self._left_gripper_cmd

        # 右臂关节
        for i in range(self.right_joint_num):
            action_dict[f"right_joint_pos{i}"] = float(right_action[i])
        action_dict["right_gripper_pos"] = right_gripper_cmd if right_gripper_cmd is not None else self._right_gripper_cmd

        self.robot.send_action(action_dict)

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
        """从当前位置平滑插值运动到目标位置（PCHIP）。"""
        obs = self.robot.get_observation()
        current_pose = self._obs_to_state(obs)

        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0.0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + duration

        logger.info("平滑移动至初始位置，耗时 %.1fs ...", duration)
        while time.time() < end_time:
            loop_start = time.time()
            t = loop_start - start_time
            position = interpolator(t)
            self.execute_action(position)
            elapsed = time.time() - loop_start
            remaining = self.dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    # 主控循环
    # ------------------------------------------------------------------

    def _setup_keyboard_listener(self) -> None:
        """将终端切到 cbreak，便于非阻塞读取方向键。"""
        if not sys.stdin.isatty():
            logger.warning("stdin 不是 TTY，无法监听方向键；将自动开始推理。")
            return
        self._keyboard_fd = sys.stdin.fileno()
        self._keyboard_old_attrs = termios.tcgetattr(self._keyboard_fd)
        tty.setcbreak(self._keyboard_fd)
        self._escape_buffer = ""

    def _restore_keyboard_listener(self) -> None:
        if self._keyboard_fd is None or self._keyboard_old_attrs is None:
            return
        termios.tcsetattr(self._keyboard_fd, termios.TCSADRAIN, self._keyboard_old_attrs)
        self._keyboard_fd = None
        self._keyboard_old_attrs = None
        self._escape_buffer = ""

    def _poll_arrow_key(self) -> str | None:
        """轮询读取终端输入，识别方向键（left/right）。"""
        if self._keyboard_fd is None:
            return None
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None

        ch = sys.stdin.read(1)
        if not ch:
            return None
        self._escape_buffer += ch
        if len(self._escape_buffer) > 3:
            self._escape_buffer = self._escape_buffer[-3:]

        if self._escape_buffer.endswith("\x1b[C"):
            self._escape_buffer = ""
            return "right"
        if self._escape_buffer.endswith("\x1b[D"):
            self._escape_buffer = ""
            return "left"
        return None

    def run_episode(self, task_prompt: str = "Use the left arm to move the black box to the center of the table, place the two small joint modules inside it, then put the black box back where it was. Next, use the left arm to move the blue box to the center of the table, place the two large joint modules inside it, then put the blue box back where it was.") -> None:
        logger.info("开始 Episode，任务指令: '%s'", task_prompt)
        self._reset_policy_state()
        logger.info("正在复位到初始位姿（reset_position）...")
        self.robot.reset_position()
        logger.info("复位完成。")

        self.episode_step = 0
        self._clear_policy_runtime_state()
        self._left_gripper_cmd = None
        self._right_gripper_cmd = None
        self.is_running = True
        is_first_step = True
        # waiting: 等右键开始推理；running: 正在推理，左键结束推理并回到 waiting
        wait_stage = "waiting"

        try:
            self._setup_keyboard_listener()
            if self._keyboard_fd is None:
                wait_stage = "running"
            else:
                logger.info("按右键开始推理；推理中按左键结束推理。(双击)")

            while self.is_running and self.episode_step < self.max_steps:
                loop_start = time.perf_counter()
                arrow_key = self._poll_arrow_key()

                if wait_stage == "waiting":
                    if arrow_key == "right":
                        wait_stage = "running"
                        self._finish_pending_infer("重新开始推理")
                        self._reset_policy_state()
                        self._clear_policy_runtime_state()
                        # 非首次开始推理时（曾跑过步数或曾暂停），先回到机械初始位姿，再按首步逻辑移到策略起点
                        if self.episode_step > 0:
                            logger.info("再次开始推理前，复位到初始位姿（reset_position）...")
                            self.robot.reset_position()
                            self._left_gripper_cmd = None
                            self._right_gripper_cmd = None
                        is_first_step = True
                        logger.info("检测到右键，开始推理。")
                    time.sleep(0.01)
                    continue

                if arrow_key == "left":
                    wait_stage = "waiting"
                    self._finish_pending_infer("暂停推理")
                    self._reset_policy_state()
                    self._clear_policy_runtime_state()
                    logger.info(
                        "检测到左键，推理结束。按右键开始推理，左键结束推理。(双击)"
                    )
                    time.sleep(0.01)
                    continue

                if self.current_action_chunk is None:
                    self.current_action_chunk = self._request_chunk_sync(task_prompt)
                    self._store_action_candidates(self.current_action_chunk, self.episode_step, start_idx=0)
                    self.action_chunk_idx = 0
                    logger.info("收到初始动作 chunk，shape=%s", self.current_action_chunk.shape)

                if (
                    self.current_action_chunk is not None
                    and self.action_chunk_idx >= self.rate_of_inference
                    and self.future_action_chunk is None
                    and self.next_action_chunk is None
                ):
                    self._submit_async_infer(task_prompt)

                self._collect_async_result()
                self._switch_to_next_chunk()

                if self.current_action_chunk is not None and self.action_chunk_idx >= len(self.current_action_chunk):
                    logger.warning("Step %s: 当前 chunk 已耗尽，等待异步推理结果。", self.episode_step)
                    while self.next_action_chunk is None and self.is_running:
                        arrow_key = self._poll_arrow_key()
                        if arrow_key == "left":
                            wait_stage = "waiting"
                            self._finish_pending_infer("暂停推理")
                            self._reset_policy_state()
                            self._clear_policy_runtime_state()
                            logger.info(
                                "检测到左键，推理结束。按右键开始推理，左键结束推理。(双击)"
                            )
                            break
                        if self.future_action_chunk is None:
                            self._submit_async_infer(task_prompt)
                        self._collect_async_result()
                        time.sleep(0.001)
                    if wait_stage == "waiting":
                        time.sleep(0.01)
                        continue
                    self._switch_to_next_chunk()

                if self.current_action_chunk is None:
                    raise RuntimeError("当前动作 chunk 为空，无法执行。")
                if self.action_chunk_idx >= len(self.current_action_chunk):
                    raise RuntimeError(
                        f"动作 chunk 仍不可用: idx={self.action_chunk_idx}, len={len(self.current_action_chunk)}"
                    )

                if self.temporal_ensemble_coefficient is not None:
                    candidates = self.action_buffer.get(self.episode_step, [])
                    if len(candidates) == 0:
                        a_t = self.current_action_chunk[self.action_chunk_idx]
                    else:
                        candidates_arr = np.array(candidates)
                        weights = self._get_weights(len(candidates_arr))
                        a_t = np.average(candidates_arr, axis=0, weights=weights)
                else:
                    a_t = self.current_action_chunk[self.action_chunk_idx]

                a_t = np.asarray(a_t, dtype=np.float32).reshape(-1)
                if a_t.shape[0] < self.action_dim:
                    raise ValueError(
                        f"策略动作维度不足: got={a_t.shape[0]}, expected>={self.action_dim}"
                    )
                a_t = a_t[: self.action_dim]

                if is_first_step:
                    logger.info("首步平滑移动至策略预测的初始位置 ...")
                    self.move_to_start_position(a_t, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(a_t)

                self.action_chunk_idx += 1
                self.episode_step += 1

                dt_s = time.perf_counter() - loop_start
                sleep_time = self.dt - dt_s
                if sleep_time > 0:
                    time.sleep(sleep_time)

                loop_s = time.perf_counter() - loop_start
                logger.info(
                    "Step %s: %.1fms (%.0f Hz)",
                    self.episode_step,
                    loop_s * 1e3,
                    1 / loop_s,
                )

        except KeyboardInterrupt:
            logger.info("用户中断，停止 Episode。")
        finally:
            self._restore_keyboard_listener()
            self.is_running = False
            logger.info("Episode 结束，共执行 %s 步。", self.episode_step)

    def cleanup(self) -> None:
        try:
            self._finish_pending_infer("断开连接")
        except Exception as e:
            logger.warning("清理异步推理时发生错误（已忽略）: %s", e)
        self.executor.shutdown(wait=False, cancel_futures=True)

        logger.info("正在断开机器人连接 ...")
        try:
            self.robot.disconnect()
            logger.info("已断开。")
        except Exception as e:
            logger.warning("断开连接时发生错误（已忽略）: %s", e)
