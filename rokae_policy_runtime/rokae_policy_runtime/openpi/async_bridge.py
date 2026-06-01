#!/usr/bin/env python3
"""
Rokae 单臂机器人 <-> OpenPI 策略服务器桥接

负责处理：
1. 从机械臂采集观测数据（关节位置、图像）
2. 通过 WebSocket 将观测发送至策略服务器
3. 接收动作预测
4. 在机械臂上执行动作
"""

import concurrent.futures
import logging
import math
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import (
    ControlMode,
    RokaeRobotConfig,
)
from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.rokae_robot import RokaeRobot
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OpenPIPolicyBridge:
    """单臂 Rokae 机器人与 OpenPI 策略服务器之间的桥接。"""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        mode: str = "autonomous",
        max_steps: int = 1000,
        zmq_port: int = 5555,
        zmq_address: str | None = None,
        control_mode: ControlMode = ControlMode.JOINT_IMPEDNACE,
        cam_high_serial: str | None = None,
        cam_wrist_serial: str | None = None,
        action_chunk_size: int = 50,
        rate_of_inference: int | None = None,
        rtc_delay: int | None = None,
        dynamic_delay: bool = True,
        delay_window: int = 8,
        delay_safety_steps: int = 1,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.mode = mode

        # 异步推理相关状态。OpenPI RTC 的上一 chunk 前缀缓存在策略服务器内，
        # 机器人端负责估计本次请求 delay，并按该请求的 delay 切换 chunk。
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # 异步任务执行器
        self.future_action_chunk = None  # 异步请求的Future对象，用于跟踪推理任务状态
        self.future_request_step: int | None = None  # 发起当前异步请求时的 episode step
        self.future_request_delay: int | None = None  # 发起当前异步请求时估计的 delay
        self.future_request_started_at: float | None = None  # 发起当前异步请求的单调时钟时间
        self.next_action_chunk: np.ndarray | None = None  # 已到达的新动作块（从异步推理获取）
        self.next_request_step: int | None = None  # next_action_chunk 的时间原点
        self.next_request_delay: int | None = None  # next_action_chunk 生成时使用的 delay
        self.chunk_ready_event = threading.Event()  # 新动作块到达事件，用于同步通知

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
        metadata_delay = int(self.server_metadata.get("max_delay", self.server_metadata.get("rtc_delay", 0)))
        metadata_rate = int(
            self.server_metadata.get("rtc_rate_of_inference", self.server_metadata.get("rate_of_inference", 30))
        )
        self.max_rtc_delay = int(metadata_delay if rtc_delay is None else rtc_delay)
        self.rtc_delay = self.max_rtc_delay
        self.rate_of_inference = int(metadata_rate if rate_of_inference is None else rate_of_inference)
        self.rtc_enabled = (metadata_rtc_enabled or rtc_delay is not None) and self.max_rtc_delay > 0
        self.dynamic_delay = bool(dynamic_delay)
        self.delay_safety_steps = int(delay_safety_steps)
        self.inference_latencies_s = deque(maxlen=max(int(delay_window), 1))
        self._delay_clamp_warned = False
        self.action_chunk_size: int = action_chunk_size

        if self.rate_of_inference <= 0:
            raise ValueError(f"rate_of_inference 必须为正数，当前为 {self.rate_of_inference}")
        if self.action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size 必须为正数，当前为 {self.action_chunk_size}")
        if self.delay_safety_steps < 0:
            raise ValueError(f"delay_safety_steps 必须为非负数，当前为 {self.delay_safety_steps}")
        if self.rtc_enabled and self.rate_of_inference + self.max_rtc_delay >= self.action_chunk_size:
            logger.warning(
                "RTC 最大切换点 rate_of_inference + max_rtc_delay = %s 接近或超过 action_chunk_size = %s；"
                "若推理未及时返回，可能耗尽旧 chunk。",
                self.rate_of_inference + self.max_rtc_delay,
                self.action_chunk_size,
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
                "每个 episode 开始前仍会尝试 reset，若失败请更新 OpenPI 代码或重启策略服务器。"
            )
        if self.rtc_enabled and self.dynamic_delay and not self.supports_dynamic_rtc_delay:
            logger.warning(
                "OpenPI server metadata 未声明 supports_dynamic_rtc_delay=True；"
                "仍会发送动态 delay 字段，若 server 代码未更新则会退化为固定 max_delay。"
            )

        cameras: dict = {}
        if cam_high_serial:
            cameras["external"] = RealSenseCameraConfig(
                serial_number_or_name=cam_high_serial,
                width=640,
                height=480,
                fps=60,
                use_depth=False,
            )
        if cam_wrist_serial:
            cameras["wrist"] = RealSenseCameraConfig(
                serial_number_or_name=cam_wrist_serial,
                width=640,
                height=480,
                fps=60,
                use_depth=False,
            )

        if not cameras:
            raise ValueError(
                "至少需要配置一个相机。请通过 --cam_high_serial 或 --cam_wrist_serial 指定相机序列号。"
            )

        robot_config = RokaeRobotConfig(
            id="rokae_single_arm",
            zmq_address=zmq_address,
            zmq_port=zmq_port,
            control_mode=control_mode,
            control_loop_fps=control_frequency,
            cameras=cameras,
        )
        self.robot: RokaeRobot = make_robot_from_config(robot_config)
        self.robot.connect()
        self.robot.reset_position()

        self.joint_num: int = self.robot.joint_num
        self.action_dim: int = self.joint_num + 1

        self.current_action_chunk: np.ndarray | None = None
        self.action_chunk_idx: int = 0
        self.episode_step: int = 0
        self.is_running: bool = False

        self.temporal_ensemble_coefficient: float | None = None

        self.action_buffer: defaultdict[int, list] = defaultdict(list)
        self.action_buffer_size: int = (
            self.max_steps + self.action_chunk_size + self.rate_of_inference + self.max_rtc_delay
        )

        self._gripper_cmd: int | None = None

    def _obs_to_state(self, obs: dict) -> np.ndarray:
        joint_positions = np.array(
            [obs[f"joint_pos{i}"] for i in range(self.joint_num)], dtype=np.float32
        )
        gripper = np.array([obs["gripper_pos"]], dtype=np.float32)
        return np.concatenate([joint_positions, gripper])

    def _preprocess_images(self, obs: dict) -> dict[str, np.ndarray]:
        cam_images: dict[str, np.ndarray] = {}
        for cam_key in self.robot.cameras:
            image_hwc = obs[cam_key]
            image_resized = cv2.resize(image_hwc, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_chw = np.transpose(image_rgb, (2, 0, 1))
            cam_images[cam_key] = image_chw
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

        # Use a conservative max-over-window estimate. If one recent request was slow,
        # keep enough committed old actions available for the next few requests.
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

    def _make_observation(self, task_prompt: str, rtc_delay: int | None = None) -> dict:
        obs = self.robot.get_observation()
        state = self._obs_to_state(obs)
        images = self._preprocess_images(obs)
        observation = {"state": state, "images": images, "prompt": task_prompt}
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
            self.action_chunk_size = int(chunk.shape[0])
            self.action_buffer_size = (
                self.max_steps + self.action_chunk_size + self.rate_of_inference + self.max_rtc_delay
            )
        return chunk

    def _store_action_candidates(self, chunk: np.ndarray, chunk_origin_step: int, start_idx: int = 0) -> None:
        """Store chunk actions on the absolute episode timeline for optional temporal ensembling."""
        for k in range(start_idx, len(chunk)):
            future_t = chunk_origin_step + k
            if future_t < self.action_buffer_size:
                self.action_buffer[future_t].append(chunk[k])

    def _request_chunk_sync(self, task_prompt: str) -> np.ndarray:
        response = self.policy_client.infer(
            self._make_observation(task_prompt, rtc_delay=self._estimate_rtc_delay())
        )
        return self._normalize_action_chunk(response["actions"])

    def _reset_policy_state(self) -> None:
        if not self.server_rtc_enabled:
            return

        try:
            self.policy_client.reset()
        except Exception as e:
            raise RuntimeError(
                "无法重置 OpenPI server 端 RTC action_prefix。"
                "为避免跨 episode 使用旧 chunk，请更新 OpenPI reset RPC 或重启策略服务器。"
            ) from e

        logger.info("OpenPI RTC 策略状态已重置，下一次 infer 将不使用旧 action_prefix。")

    def _submit_async_infer(self, task_prompt: str) -> None:
        if self.future_action_chunk is not None or self.next_action_chunk is not None:
            return
        request_step = self.episode_step
        request_delay = self._estimate_rtc_delay()
        observation = self._make_observation(task_prompt, rtc_delay=request_delay)
        self.future_request_step = request_step
        self.future_request_delay = request_delay
        self.future_request_started_at = time.perf_counter()
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
        future = self.future_action_chunk
        self.future_action_chunk = None
        self.future_request_step = None
        self.future_request_delay = None
        self.future_request_started_at = None

        try:
            result = future.result()
        except Exception as e:
            logger.error("获取异步推理结果失败: %s", e)
            return

        if result is None:
            logger.error("异步推理返回空动作块，将继续使用当前 chunk 并尝试重新请求。")
            return

        if request_step is None:
            request_step = self.episode_step
        if request_delay is None:
            request_delay = self._estimate_rtc_delay()
        measured_delay = None
        if request_started_at is not None:
            measured_delay = self._record_inference_latency(time.perf_counter() - request_started_at)

        self.next_action_chunk = self._normalize_action_chunk(result)
        self.next_request_step = request_step
        self.next_request_delay = self._clamp_rtc_delay(request_delay)
        self.chunk_ready_event.set()

        start_idx = self.next_request_delay if self.rtc_enabled else 0
        self._store_action_candidates(self.next_action_chunk, request_step, start_idx=start_idx)
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

    def execute_action(self, action: np.ndarray) -> None:
        if self.mode == "test":
            logger.info("[TEST] 模拟执行动作: %s", action)
            return

        raw_gripper = float(action[self.joint_num])
        gripper_cmd: int | None = 1 if raw_gripper >= 0.5 else 0

        if gripper_cmd == self._gripper_cmd:
            gripper_cmd = None
        else:
            self._gripper_cmd = gripper_cmd
            logger.info("夹爪状态切换: %s", "闭合(0)" if gripper_cmd == 0 else "张开(1)")

        action_dict = {f"joint_pos{i}": float(action[i]) for i in range(self.joint_num)}
        if gripper_cmd is not None:
            action_dict["gripper_pos"] = gripper_cmd
        else:
            action_dict["gripper_pos"] = self._gripper_cmd
        self.robot.send_action(action_dict)

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
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

    def _async_infer(self, observation):
        """异步执行推理请求

        该方法在线程池中执行，用于向策略服务器发送推理请求并获取动作块。
        由于网络通信和模型推理可能需要较长时间，通过异步执行可以避免阻塞主控制循环。

        参数:
            observation: 包含状态、图像和任务提示的观测字典

        返回:
            np.ndarray | None: 50步的动作块，如果推理失败则返回None
        """
        try:
            response = self.policy_client.infer(observation)
            return response["actions"]
        except Exception as e:
            logger.error("异步推理失败: %s", e)
            return None

    def run_episode(self, task_prompt: str = "move the arm") -> None:
        logger.info("开始 Episode，任务指令: '%s'", task_prompt)
        self._reset_policy_state()
        logger.info("正在复位到初始位姿（reset_position）...")
        self.robot.reset_position()
        logger.info("复位完成。")

        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.next_action_chunk = None
        self.next_request_step = None
        self.next_request_delay = None
        self.future_action_chunk = None
        self.future_request_step = None
        self.future_request_delay = None
        self.future_request_started_at = None
        self.chunk_ready_event.clear()
        self.action_buffer.clear()
        self._gripper_cmd = None
        self.is_running = True
        is_first_step = True

        try:
            while self.is_running and self.episode_step < self.max_steps:
                loop_start = time.perf_counter()

                if self.current_action_chunk is None:
                    logger.info("Step %s: 同步请求初始动作块...", self.episode_step)
                    self.current_action_chunk = self._request_chunk_sync(task_prompt)
                    self._store_action_candidates(self.current_action_chunk, self.episode_step, start_idx=0)
                    self.action_chunk_idx = 0
                    logger.info("收到初始动作 chunk，shape=%s", self.current_action_chunk.shape)

                # OpenPI RTC 约定：当旧 chunk 已经执行 rate_of_inference 步时发起下一次请求；
                # 请求级 request_delay 会随 observation 发送给 server，并决定后续 chunk 切换点。
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
                        if self.future_action_chunk is None:
                            self._submit_async_infer(task_prompt)
                        self._collect_async_result()
                        time.sleep(0.001)
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
                    "Step %s: %.1fms (%.0f Hz), chunk_idx=%s",
                    self.episode_step,
                    loop_s * 1e3,
                    1 / loop_s,
                    self.action_chunk_idx,
                )

        except KeyboardInterrupt:
            logger.info("用户中断，停止 Episode。")
        finally:
            self.is_running = False
            logger.info("Episode 结束，共执行 %s 步。", self.episode_step)

    def cleanup(self) -> None:
        """清理资源：断开机器人连接并关闭异步执行器

        在程序结束时调用此方法以释放所有资源，包括：
        1. 断开机器人硬件连接
        2. 关闭线程池执行器，等待异步任务完成
        3. 清理所有事件和缓存
        """
        logger.info("正在断开机器人连接 ...")
        try:
            self.robot.disconnect()
            logger.info("已断开。")
        except Exception as e:
            logger.warning("断开连接时发生错误（已忽略）: %s", e)
        finally:
            # 关闭线程池执行器，确保所有异步任务都完成
            if hasattr(self, 'executor'):
                logger.info("正在关闭异步执行器...")
                try:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                    logger.info("异步执行器已关闭。")
                except Exception as e:
                    logger.warning("关闭执行器时发生错误: %s", e)
