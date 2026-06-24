"""各 teleop 设备共用的 pipeline 配置、基类与 processor。"""

from .config import (
    ArmConfig,
    CoreArmRuntime,
    arm_config,
    ensure_cart_pos_flan_in_base,
    pop_pipeline_scratch,
    read_pipeline_vec6,
    register_arm_action_features,
    register_arm_kinematics_features,
    register_gripper_action_feature,
    write_arm_joints,
    write_pipeline_vec6,
)
from .pipeline_base import ArmPipeline, init_arm_pipeline, init_kinematics_backend
from .processors import PosFlanInBaseToEndInRefProcessor, SolveArmProcessor

__all__ = [
    "ArmConfig",
    "CoreArmRuntime",
    "arm_config",
    "ensure_cart_pos_flan_in_base",
    "pop_pipeline_scratch",
    "read_pipeline_vec6",
    "register_arm_action_features",
    "register_arm_kinematics_features",
    "register_gripper_action_feature",
    "write_arm_joints",
    "write_pipeline_vec6",
    "ArmPipeline",
    "init_arm_pipeline",
    "init_kinematics_backend",
    "PosFlanInBaseToEndInRefProcessor",
    "SolveArmProcessor",
]
