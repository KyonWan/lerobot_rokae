from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_spacemouse")
@dataclass
class BiSpacemouseConfig(TeleoperatorConfig):
    # Device indices for left and right spacemouse devices
    left_device_index: Optional[int] = 0
    right_device_index: Optional[int] = 1
    # 上层参考笛卡尔速度上限（m/s 和 rad/s），会传入 BiInverseKinematicsProcessor
    trans_max_vel: float = 0.15
    rot_max_vel: float = 0.15
    # fixed_ar_dual | wheeled_ar_dual；见 pink_ik_helpers.list_kinematics_presets()
    kinematics_preset: str = "fixed_ar_dual"
    # 非 None 时覆盖预设解析出的路径 / 末端 link（仍可与 ROKAE_IK_URDF_PATH_* 环境变量组合使用）
    left_urdf_path: Optional[str] = None
    right_urdf_path: Optional[str] = None
    left_end_effector_frame: Optional[str] = None
    right_end_effector_frame: Optional[str] = None
