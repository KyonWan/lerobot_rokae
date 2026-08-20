"""各 teleop 设备共用的 arm 配置、runtime 与 action scratch 工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rokae_python_wrapper.rokae_kinematics.action_fields import obs_cart_pos_end_in_base
from rokae_python_wrapper.rokae_kinematics.cartesian_frames import (
    compute_tool_chain,
    flan_pose_from_end_observation,
)
from rokae_python_wrapper.rokae_kinematics.cartesian_ik_solver import CartesianIkSolver

PIPELINE_SCRATCH_SUFFIXES = (
    "delta_pos_end_in_ref",
    "delta_pos_flan_in_base",
)


@dataclass
class ArmConfig:
    key_prefix: str = ""
    joint_num: int = 7
    robot_type: str = ""
    robot_ip: str = ""
    initial_gripper_state: int = 1
    tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    base_frame_in_world: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
    joint_position_lower_limits: np.ndarray | None = None
    joint_position_upper_limits: np.ndarray | None = None
    joint_coupling_limit: dict | None = None


@dataclass
class CoreArmRuntime:
    """跨帧持久状态（法兰积分起点、逆解器、夹爪）。"""

    cart_pos_flan_in_base: np.ndarray | None = None
    ik_solver: CartesianIkSolver | None = None
    gripper_state: int = 1


def pipeline_scratch_key(key_prefix: str, suffix: str) -> str:
    return f"{key_prefix}__pipeline_{suffix}"


def write_pipeline_vec6(action: dict, key_prefix: str, suffix: str, values: np.ndarray) -> None:
    action[pipeline_scratch_key(key_prefix, suffix)] = np.asarray(
        values, dtype=np.float64
    ).reshape(6).tolist()


def read_pipeline_vec6(
    action: dict, key_prefix: str, suffix: str, *, default_zero: bool = True
) -> np.ndarray | None:
    raw = action.get(pipeline_scratch_key(key_prefix, suffix))
    if raw is None:
        return np.zeros(6, dtype=np.float64) if default_zero else None
    return np.asarray(raw, dtype=np.float64).reshape(6)


def write_arm_joints(
    action: dict,
    key_prefix: str,
    q: list[float] | np.ndarray,
    joint_num: int,
) -> None:
    p = key_prefix
    vals = [float(x) for x in q]
    for i in range(joint_num):
        action[f"{p}joint_pos{i}"] = vals[i]


def ensure_cart_pos_flan_in_base(
    cfg: ArmConfig,
    rt: CoreArmRuntime,
    obs: dict,
) -> None:
    if rt.cart_pos_flan_in_base is not None:
        return
    _, T_flan_in_end, _, _ = compute_tool_chain(
        cfg.tool_end_pos, cfg.tool_ref_pos, cfg.base_frame_in_world
    )
    cart_pos_end_in_base = obs_cart_pos_end_in_base(obs, cfg.key_prefix)
    rt.cart_pos_flan_in_base = flan_pose_from_end_observation(
        cart_pos_end_in_base, T_flan_in_end
    )


def pop_pipeline_scratch(action: dict, key_prefix: str) -> None:
    for suffix in PIPELINE_SCRATCH_SUFFIXES:
        action.pop(pipeline_scratch_key(key_prefix, suffix), None)


def arm_config(
    key_prefix: str,
    joint_num: int,
    robot_type: str,
    tool_end_pos: Any,
    tool_ref_pos: Any,
    base_frame_in_world: Any,
    trans_max_vel: float,
    rot_max_vel: float,
    *,
    robot_ip: str = "",
    initial_gripper_state: int = 1,
    joint_position_lower_limits: Any = None,
    joint_position_upper_limits: Any = None,
    joint_coupling_limit: dict | None = None,
) -> ArmConfig:
    return ArmConfig(
        key_prefix=key_prefix,
        joint_num=joint_num,
        robot_type=str(robot_type or ""),
        robot_ip=str(robot_ip or ""),
        initial_gripper_state=initial_gripper_state,
        tool_end_pos=_as_pose6(tool_end_pos),
        tool_ref_pos=_as_pose6(tool_ref_pos),
        base_frame_in_world=_as_pose6(base_frame_in_world),
        trans_max_vel=float(trans_max_vel),
        rot_max_vel=float(rot_max_vel),
        joint_position_lower_limits=(
            None
            if joint_position_lower_limits is None
            else np.asarray(joint_position_lower_limits, dtype=np.float64).reshape(joint_num)
        ),
        joint_position_upper_limits=(
            None
            if joint_position_upper_limits is None
            else np.asarray(joint_position_upper_limits, dtype=np.float64).reshape(joint_num)
        ),
        joint_coupling_limit=(
            None if joint_coupling_limit is None else dict(joint_coupling_limit)
        ),
    )


def _as_pose6(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(6, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(6)


def register_arm_kinematics_features(
    action_features: dict,
    key_prefix: str,
    joint_num: int,
) -> None:
    p = key_prefix
    for i in range(joint_num):
        action_features.setdefault(f"{p}joint_pos{i}", float)
    for i in range(6):
        action_features.setdefault(f"{p}cart_pos{i}", float)
    action_features.setdefault(f"{p}psi", float)


def register_gripper_action_feature(action_features: dict, key_prefix: str) -> None:
    action_features.setdefault(f"{key_prefix}gripper_pos", float)


def register_arm_action_features(
    action_features: dict,
    key_prefix: str,
    joint_num: int,
) -> None:
    register_arm_kinematics_features(action_features, key_prefix, joint_num)
    register_gripper_action_feature(action_features, key_prefix)
