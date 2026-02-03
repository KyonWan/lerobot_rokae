from lerobot.processor.pipeline import RobotActionProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import RobotAction
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


@ProcessorStepRegistry.register("bi_spacemouse_vel_map")
@dataclass
class ExtractBiCartVelAndGripper(RobotActionProcessorStep):
    """
    Processor for bimanual spacemouse: extracts cartesian velocities and gripper states
    for both left and right arms.
    This processor runs in robot_action_processor pipeline (before sending to robot).
    """
    TRANS_MAX_VEL = 0.1
    ROT_MAX_VEL = 0.2

    def __post_init__(self):
        # 左臂：基相对于世界是沿X轴旋转-90°
        R_base_in_world_left = R.from_euler("X", -90, degrees=True)
        self.R_world_in_base_left = R_base_in_world_left.inv().as_matrix()

        # 右臂：基相对于世界是沿X轴旋转90°
        R_base_in_world_right = R.from_euler("X", 90, degrees=True)
        self.R_world_in_base_right = R_base_in_world_right.inv().as_matrix()

        # Gripper切换状态跟踪：记录上一次gripper_pos值和当前gripper状态
        self.left_gripper_pos_prev = 0  # 上一次左gripper_pos值
        self.right_gripper_pos_prev = 0  # 上一次右gripper_pos值
        self.left_gripper_state = 0  # 当前左gripper状态（0=close, 1=open）
        self.right_gripper_state = 0  # 当前右gripper状态（0=close, 1=open）

    def action(self, action: RobotAction) -> RobotAction:
        # Process left arm
        # 原始速度是世界相对于基的速度（在世界坐标系中表示）
        left_trans_vel_world = np.array([action[f"left_cart_vel{i}"] for i in range(3)])
        left_rot_vel_world = np.array([action[f"left_cart_vel{i+3}"] for i in range(3)])

        # 转换为基坐标系中的速度（末端相对于基）
        left_trans_vel = self.R_world_in_base_left @ left_trans_vel_world
        left_rot_vel = self.R_world_in_base_left @ left_rot_vel_world

        left_trans_vel *= self.TRANS_MAX_VEL
        left_rot_vel *= self.ROT_MAX_VEL

        # Process right arm
        # 原始速度是世界相对于基的速度（在世界坐标系中表示）
        right_trans_vel_world = np.array([action[f"right_cart_vel{i}"] for i in range(3)])
        right_rot_vel_world = np.array([action[f"right_cart_vel{i+3}"] for i in range(3)])

        # 转换为基坐标系中的速度（末端相对于基）
        right_trans_vel = self.R_world_in_base_right @ right_trans_vel_world
        right_rot_vel = self.R_world_in_base_right @ right_rot_vel_world

        right_trans_vel *= self.TRANS_MAX_VEL
        right_rot_vel *= self.ROT_MAX_VEL

        # Extract gripper states - 切换式开关逻辑
        # action中已经有left_gripper_pos和right_gripper_pos（单个值）
        # 检测gripper_pos从0变为1的上升沿，切换gripper状态
        left_gripper_pos_curr = action.get("left_gripper_pos", 0)
        right_gripper_pos_curr = action.get("right_gripper_pos", 0)

        # 标准化为0或1
        left_gripper_pos_curr = 1 if left_gripper_pos_curr else 0
        right_gripper_pos_curr = 1 if right_gripper_pos_curr else 0

        # 检测左gripper_pos上升沿（从0变为1）：切换gripper状态
        if left_gripper_pos_curr == 1 and self.left_gripper_pos_prev == 0:
            # gripper_pos从未按下变为按下，切换gripper状态
            self.left_gripper_state = 1 - self.left_gripper_state

        # 检测右gripper_pos上升沿（从0变为1）：切换gripper状态
        if right_gripper_pos_curr == 1 and self.right_gripper_pos_prev == 0:
            # gripper_pos从未按下变为按下，切换gripper状态
            self.right_gripper_state = 1 - self.right_gripper_state

        # 更新上一次gripper_pos状态
        self.left_gripper_pos_prev = left_gripper_pos_curr
        self.right_gripper_pos_prev = right_gripper_pos_curr

        # 使用切换后的gripper状态
        left_gripper = self.left_gripper_state
        right_gripper = self.right_gripper_state

        return {
            **{f"left_cart_vel{i}": float(left_trans_vel[i]) for i in range(3)},
            **{f"left_cart_vel{i+3}": float(left_rot_vel[i]) for i in range(3)},
            "left_gripper_pos": float(left_gripper),
            **{f"right_cart_vel{i}": float(right_trans_vel[i]) for i in range(3)},
            **{f"right_cart_vel{i+3}": float(right_rot_vel[i]) for i in range(3)},
            "right_gripper_pos": float(right_gripper),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Remove joint_pos features if they exist (they will be generated from cart_vel)
        for side in ["left", "right"]:
            for i in range(6):  # Assuming 6 DOF
                features[PipelineFeatureType.ACTION].pop(f"{side}_joint_pos{i}", None)
        return features
