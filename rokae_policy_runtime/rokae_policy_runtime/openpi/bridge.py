#!/usr/bin/env python3
"""
Rokae 单臂机器人 <-> OpenPI 策略服务器桥接

负责处理：
1. 从机械臂采集观测数据（关节位置、图像）
2. 通过 WebSocket 将观测发送至策略服务器
3. 接收动作预测
4. 在机械臂上执行动作
"""

import time
import logging
from collections import defaultdict

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import (
    CallbackMode,
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
        callback_mode: CallbackMode = CallbackMode.JOINT_POS,
        cam_high_serial: str | None = None,
        cam_wrist_serial: str | None = None,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.mode = mode

        logger.info("正在连接策略服务器 %s:%s ...", policy_server_host, policy_server_port)
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
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
            callback_mode=callback_mode,
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
        self.action_chunk_size: int = 50
        self.rate_of_inference: int = 30
        self.episode_step: int = 0
        self.is_running: bool = False

        self.temporal_ensemble_coefficient: float | None = None

        self.action_buffer: defaultdict[int, list] = defaultdict(list)
        self.action_buffer_size: int = self.max_steps + self.action_chunk_size

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

    def run_episode(self, task_prompt: str = "move the arm") -> None:
        logger.info("开始 Episode，任务指令: '%s'", task_prompt)
        logger.info("正在复位到初始位姿（reset_position）...")
        self.robot.reset_position()
        logger.info("复位完成。")

        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.action_buffer.clear()
        self._gripper_cmd = None
        self.is_running = True
        is_first_step = True

        try:
            while self.is_running and self.episode_step < self.max_steps:
                loop_start = time.perf_counter()

                need_new_chunk = (
                    self.current_action_chunk is None
                    or self.action_chunk_idx >= self.rate_of_inference
                )
                if need_new_chunk:
                    obs = self.robot.get_observation()
                    state = self._obs_to_state(obs)
                    images = self._preprocess_images(obs)
                    observation = {"state": state, "images": images, "prompt": task_prompt}

                    logger.info("Step %s: 向策略服务器请求动作 chunk ...", self.episode_step)
                    response = self.policy_client.infer(observation)
                    self.current_action_chunk = response["actions"]

                    for k in range(self.action_chunk_size):
                        future_t = self.episode_step + k
                        if future_t < self.action_buffer_size:
                            self.action_buffer[future_t].append(self.current_action_chunk[k])

                    self.action_chunk_idx = 0
                    logger.info("收到动作 chunk，shape: %s", len(self.current_action_chunk))

                if self.temporal_ensemble_coefficient is not None:
                    candidates = self.action_buffer.get(self.episode_step, [])
                    if len(candidates) == 0:
                        a_t = np.zeros(self.action_dim, dtype=np.float32)
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
                logger.info("Step %s: %.1fms (%.0f Hz)", self.episode_step, loop_s * 1e3, 1 / loop_s)

        except KeyboardInterrupt:
            logger.info("用户中断，停止 Episode。")
        finally:
            self.is_running = False
            logger.info("Episode 结束，共执行 %s 步。", self.episode_step)

    def cleanup(self) -> None:
        logger.info("正在断开机器人连接 ...")
        try:
            self.robot.disconnect()
            logger.info("已断开。")
        except Exception as e:
            logger.warning("断开连接时发生错误（已忽略）: %s", e)
