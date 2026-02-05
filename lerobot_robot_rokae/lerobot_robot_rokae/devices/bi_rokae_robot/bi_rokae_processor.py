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

        # Extract gripper states - 直接使用teleop processor已经转换好的gripper状态
        # gripper_pos已经是夹爪开关状态（0=close, 1=open），不需要再次转换
        left_gripper = action.get("left_gripper_pos")
        right_gripper = action.get("right_gripper_pos")

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
        # Remove all joint_pos features (they will be replaced by cart_vel)
        # Iterate over a copy of keys to avoid modification during iteration
        action_features = features[PipelineFeatureType.ACTION]
        keys_to_remove = [
            key for key in action_features.keys()
            if key.startswith("left_joint_pos") or key.startswith("right_joint_pos")
        ]
        for key in keys_to_remove:
            action_features.pop(key, None)
        return features
