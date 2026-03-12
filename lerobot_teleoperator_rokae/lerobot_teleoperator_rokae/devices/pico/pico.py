# ---------- 1. 依赖导入与全局常量：导入运行依赖，并定义双臂控制映射配置 ----------
import logging
from pathlib import Path
from typing import Any, Dict
import time
import threading
import atexit
import re
import importlib
from lerobot.teleoperators.teleoperator import Teleoperator
# from .config_pico import PicoConfig
try:
    from .config_pico import PicoConfig
except ImportError:
    from config_pico import PicoConfig
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import CONTROLLER_DEADZONE
import numpy as np
import meshcat.transformations as tf
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# 定义每条手臂对应的 XR 输入源和按键映射
DEFAULT_MANIPULATOR_CONFIG = {
    "left_arm": {
        "link_name": "left_tool0",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
    },
    "right_arm": {
        "link_name": "right_tool0",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}

ARM_MAP = {
    "left_arm": {
        "last": "_last_left_trigger_val",
        "pos": "left_gripper_pos",
    },
    "right_arm": {
        "last": "_last_right_trigger_val",
        "pos": "right_gripper_pos",
    },
}


class Pico(Teleoperator):
    """
    Pico VR Teleop class for controlling robot arms via Cartesian pose deltas.
    Outputs Cartesian pose deltas instead of joint positions, allowing the framework
    to handle inverse kinematics through processor pipelines.
    """

    config_class = PicoConfig
    name = "pico"

    # ---------- 2. Pico 类初始化：创建 XR 客户端、状态缓存、坐标系变换与调试记录容器 ----------
    def __init__(self, config: PicoConfig):
        super().__init__(config)
        # CLI/配置文件场景下无法直接注入对象，这里默认自动创建。
        self.xr_client = config.xr_client or XrClient()
        self.cfg = config
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        self._is_connected = False
        self._stop_event = threading.Event()
        self._last_left_trigger_val = 1.0
        self._last_right_trigger_val = 1.0
        self.manipulator_config = DEFAULT_MANIPULATOR_CONFIG
        self.R_headset_world = R.from_euler('ZYX', self.cfg.R_headset_world, degrees=True).as_matrix()

        # Store current delta values for each arm
        self.current_delta_xyz = {}
        self.current_delta_rot = {}
        self.left_gripper_pos = self.cfg.open_position
        self.right_gripper_pos = self.cfg.open_position

        # 增加用于离合机制（clutching）的变量：记录累积基准和上一帧的激活状态
        self.base_delta_xyz = {}
        self.base_delta_rot = {}
        self.was_active = {}

    # ---------- 3. 基础属性接口：暴露连接状态/特征接口，预留校准状态接口 ----------
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
        pass

    # ---------- 4. 连接与后台更新线程：初始化双臂状态并启动固定频率的姿态更新循环 ----------
    def connect(self) -> None:
        # Initialize delta values for each arm
        for arm_name in self.manipulator_config.keys():
            self.current_delta_xyz[arm_name] = np.zeros(3)
            self.current_delta_rot[arm_name] = np.zeros(3)
            self.init_controller_xyz[arm_name] = None
            self.init_controller_quat[arm_name] = None
            
            # 初始化基准偏移量和按键状态
            self.base_delta_xyz[arm_name] = np.zeros(3)
            self.base_delta_rot[arm_name] = np.zeros(3)
            self.was_active[arm_name] = False

        # Start update thread from xrclient
        threading.Thread(target=self._start_pose_update, daemon=True).start()

        # Check completed
        self._is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.\n")

    def reset_for_new_episode(self) -> None:
        """Clear accumulated delta targets to avoid replaying previous episode pose."""
        for arm_name in self.manipulator_config.keys():
            self.current_delta_xyz[arm_name] = np.zeros(3)
            self.current_delta_rot[arm_name] = np.zeros(3)
            self.base_delta_xyz[arm_name] = np.zeros(3)
            self.base_delta_rot[arm_name] = np.zeros(3)
            self.init_controller_xyz[arm_name] = None
            self.init_controller_quat[arm_name] = None
            self.was_active[arm_name] = False

        self._last_left_trigger_val = self.cfg.open_position
        self._last_right_trigger_val = self.cfg.open_position
        self.left_gripper_pos = self.cfg.open_position
        self.right_gripper_pos = self.cfg.open_position

    def _start_pose_update(self):
        """Update pose deltas from XR client in a separate thread."""
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()
                self._update_pose_deltas_from_xr()
                elapsed = time.perf_counter() - start
                time.sleep(max(0, 1/self.cfg.fps - elapsed))
            except Exception as e:
                logger.error(f"Error in pose update thread: {e}")

    # ---------- 5. XR 输入处理主流程：按手臂处理激活状态、夹爪切换与增量更新 ----------
    def _update_pose_deltas_from_xr(self):
        """从 XR 控制器输入更新 Cartesian 位姿增量。该位姿增量是相对于头显坐标系的位姿增量。"""
        for arm_name, config in self.manipulator_config.items():
            # Check if controller is active (grip button pressed)
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

            # Update gripper state
            trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
            if self.cfg.trigger_reverse:
                trigger_val = self.cfg.open_position - trigger_val
            if trigger_val < self.cfg.trigger_threshold:
                trigger_val = self.cfg.close_position
            else:
                trigger_val = self.cfg.open_position

            last_attr = ARM_MAP[arm_name]["last"]
            pos_attr = ARM_MAP[arm_name]["pos"]

            last_trigger = getattr(self, last_attr)
            gripper_pos = getattr(self, pos_attr)

            if last_trigger == 1 and trigger_val == 0:
                gripper_pos = (
                    self.cfg.close_position
                    if gripper_pos == self.cfg.open_position
                    else self.cfg.open_position
                )

            setattr(self, pos_attr, gripper_pos)
            setattr(self, last_attr, trigger_val)

            # Update pose deltas
            if active:
                # 如果前一帧未激活，说明是“刚按下”，需要清空初始手柄参考位姿，从当前的新位置重新开始计算相对移动
                if not self.was_active[arm_name]:
                    self.init_controller_xyz[arm_name] = None
                    self.init_controller_quat[arm_name] = None

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, arm_name)
                
                # 累加：当前发送的控制量 = 松开前累积的基准量 + 这一次按下后的相对偏移量               
                self.current_delta_xyz[arm_name] = self.base_delta_xyz[arm_name] + delta_xyz
                self.current_delta_rot[arm_name] = self.base_delta_rot[arm_name] + delta_rot_angle_axis
            else:
                # 当非active时，不再重置current_delta，使机械臂保持在原地
                # 如果前一帧是激活状态，说明是“刚松开”，将当前的位姿保存为新的基准量
                if self.was_active[arm_name]:
                    self.base_delta_xyz[arm_name] = self.current_delta_xyz[arm_name].copy()
                    self.base_delta_rot[arm_name] = self.current_delta_rot[arm_name].copy()
                    
                    # 提前清理参考位姿（保持逻辑完整性）
                    self.init_controller_xyz[arm_name] = None
                    self.init_controller_quat[arm_name] = None
            # 记录当前帧状态，供下一帧比较
            self.was_active[arm_name] = active

    # ---------- 6. 位姿增量计算：将 XR 位姿转换到世界系，并计算相对参考位姿的平移/旋转增量 ----------
    def _process_xr_pose(self, xr_pose, arm_name: str):
        """处理获得的头显位姿，将其转换到世界坐标系，并计算相对于参考位姿的平移/旋转增量。"""
        # xr_pose is typically [tx, ty, tz, qx, qy, qz, qw]
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array(
            [
                xr_pose[6],  # w
                xr_pose[3],  # x
                xr_pose[4],  # y
                xr_pose[5],  # z
            ]
        )
        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])  # Angle-axis
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.cfg.xyz_scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat) * self.cfg.rot_scale_factor
        return delta_xyz, delta_rot

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Returns Cartesian pose deltas for each arm.
        Format compatible with EEReferenceAndDelta processor:
        - target_x, target_y, target_z: position deltas
        - target_wx, target_wy, target_wz: rotation deltas (rotation vector)
        - gripper_position: gripper position
        """
        action = {}

        for arm_name in self.manipulator_config.keys():
            prefix = "left" if arm_name == "left_arm" else "right"

            # Get current delta values
            delta_xyz = self.current_delta_xyz[arm_name]
            delta_rot = self.current_delta_rot[arm_name]

            # Format: target_x, target_y, target_z, target_wx, target_wy, target_wz
            action[f"{prefix}_target_x"] = float(delta_xyz[0])
            action[f"{prefix}_target_y"] = float(delta_xyz[1])
            action[f"{prefix}_target_z"] = float(delta_xyz[2])
            action[f"{prefix}_target_wx"] = float(delta_rot[0])
            action[f"{prefix}_target_wy"] = float(delta_rot[1])
            action[f"{prefix}_target_wz"] = float(delta_rot[2])

            # Gripper: convert position to velocity-like value
            # For discrete gripper: 0 = close, 1 = open
            gripper_pos = self.left_gripper_pos if arm_name == "left_arm" else self.right_gripper_pos
            action[f"{prefix}_gripper_pos"] = gripper_pos

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # Stop update thread
        self._stop_event.set()

        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

# 单独运行 python pico.py 来测试 VR 设备是否连接成功并打印数据
if __name__ == "__main__":
    # 配置日志输出
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    # 实例化 XrClient
    xr_client = XrClient()

    # 1. 直接使用 config_pico.py 中定义的 PicoConfig 类
    # 这里的参数会覆盖 config_pico.py 中的默认值
    # 如果你想测试特定的参数（比如缩放因子），可以在这里修改
    teleop_config = PicoConfig(
        xr_client=xr_client,
        fps=60.0,             # 覆盖默认 fps
        control_mode="pico",  # 确保模式正确
        xyz_scale_factor=0.5,
        rot_scale_factor=0.5,     # 测试缩放
        # 其他参数如 trigger_threshold 等将使用 config_pico.py 中的默认值
    )

    print(f"[TEST] Initializing Pico with config: {teleop_config}")

    # 2. 实例化 Pico 遥操器
    teleop = Pico(teleop_config)
    
    # 3. 连接设备
    teleop.connect()

    i = 0

    try:
        print("[TEST] Pico Connected. Printing actions... (Press Ctrl+C to stop)")
        while True:
            # 获取并打印动作，用于检查数据是否正确（如坐标变化、夹爪状态）
            action = teleop.get_action()
            print(
                f"当前运行时长：{i*0.5}s\n"
                f"left_target_x: {action['left_target_x']:.4f}\n"
                f"left_target_y: {action['left_target_y']:.4f}\n"
                f"left_target_z: {action['left_target_z']:.4f}\n"
                # f"left_target_wx: {action['left_target_wx']:.4f}\n"
                # f"left_target_wy: {action['left_target_wy']:.4f}\n"
                # f"left_target_wz: {action['left_target_wz']:.4f}\n"
                f"left_gripper_position: {action['left_gripper_pos']:.4f}\n"
                # f"right_target_x: {action['right_target_x']:.4f}\n"
                # f"right_target_y: {action['right_target_y']:.4f}\n"
                # f"right_target_z: {action['right_target_z']:.4f}\n"
                # f"right_target_wx: {action['right_target_wx']:.4f}\n"
                # f"right_target_wy: {action['right_target_wy']:.4f}\n"
                # f"right_target_wz: {action['right_target_wz']:.4f}\n"
                # f"right_gripper_position: {action['right_gripper_pos']:.4f}\n"
            )
            
            # 控制打印频率，避免刷屏太快
            time.sleep(0.5) 
            i += 1
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user.")
    finally:
        # 4. 安全断开连接
        teleop.disconnect()