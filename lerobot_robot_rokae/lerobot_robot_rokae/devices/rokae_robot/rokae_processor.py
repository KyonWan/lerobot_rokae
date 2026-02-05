from lerobot.processor.pipeline import RobotActionProcessorStep, ObservationProcessorStep, PolicyActionProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_STATE, ACTION
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R

@ProcessorStepRegistry.register("space_mouse_vel_map")
@dataclass
class ExtractCartVelAndGripper(RobotActionProcessorStep):
    """
    Processor for single arm spacemouse: extracts cartesian velocities and gripper states.
    This processor runs in robot_action_processor pipeline (before sending to robot).
    """
    TRANS_MAX_VEL = 0.1
    ROT_MAX_VEL = 0.2

    def __post_init__(self):
        # 单臂：基相对于世界是沿X轴旋转90°（默认右臂）
        # 如果需要左臂，可以修改为-90°
        R_base_in_world = R.from_euler("X", 0, degrees=True)
        self.R_world_in_base = R_base_in_world.inv().as_matrix()

    def action(self, action: RobotAction) -> RobotAction:
        # 原始速度是世界相对于基的速度（在世界坐标系中表示）
        trans_vel_world = np.array([action[f"cart_vel{i}"] for i in range(3)])
        rot_vel_world = np.array([action[f"cart_vel{i+3}"] for i in range(3)])

        # 转换为基坐标系中的速度（末端相对于基）
        trans_vel = self.R_world_in_base @ trans_vel_world
        rot_vel = self.R_world_in_base @ rot_vel_world

        # 限制速度
        trans_vel *= self.TRANS_MAX_VEL
        rot_vel *= self.ROT_MAX_VEL

        # Extract gripper state - 直接使用teleop processor已经转换好的gripper状态
        # gripper_pos已经是夹爪开关状态（0=close, 1=open），不需要再次转换
        gripper = action.get("gripper_pos")

        return {
            **{f"cart_vel{i}": float(trans_vel[i]) for i in range(3)},
            **{f"cart_vel{i+3}": float(rot_vel[i]) for i in range(3)},
            "gripper_pos": float(gripper),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Remove all joint_pos features (they will be replaced by cart_vel)
        # Iterate over a copy of keys to avoid modification during iteration
        action_features = features[PipelineFeatureType.ACTION]
        keys_to_remove = [key for key in action_features.keys() if key.startswith("joint_pos")]
        for key in keys_to_remove:
            action_features.pop(key, None)
        return features
