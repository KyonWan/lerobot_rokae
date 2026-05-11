"""
Pink 7 轴 IK 与笛卡尔积分后逆解共用：URDF 路径、限位配置、`integrate_twist_then_solve_ik`。

供单臂 `spacemouse_processor` 与双臂 `bi_spacemouse_processor` 使用。

运动学预设（固定式双臂 AR5-5 07L/07R `*_tcp` vs 轮式双臂常用 `*_flan_link`）见
`KINEMATICS_PRESET_*` 与 `resolve_dual_arm_kinematics` / `resolve_single_arm_kinematics`；
URDF 仍优先由环境变量 `ROKAE_IK_URDF_PATH` / `ROKAE_IK_URDF_PATH_LEFT` / `ROKAE_IK_URDF_PATH_RIGHT` 覆盖。

双臂 6 轴因 `rokae_algo` 全局单例，使用 `cr6_kinematics_session` 每帧 `cr_init`→求解→`de_init`；
单臂 6 轴仍在处理器构造时一次 `cr_init`，不使用此会话。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from rokae_python_wrapper.rokae_kinematics import rokae_algo
from rokae_python_wrapper.rokae_kinematics import rokae_ik_interface as rk_ik

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import transform_to_pose

from .cartesian_ik_helpers import (
    compute_tool_chain,
    flan_pose_from_end_observation,
    integrate_end_in_ref,
    transform_to_quat_wfirst,
)

# 7 轴 pink IK 关节速度/加速度上限（rad/s、rad/s²），与笛卡尔 xd_limit_abs 分开约束
SNS_JOINT_VEL_MAX = 1.0
SNS_JOINT_ACC_MAX = 10.0

POSTURE_COSTS_7 = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 10e-3, 10e-3])

_ROKA_URDF_ROOT = Path(rk_ik.__file__).resolve().parent / "rokae_urdf"
_DEFAULT_ROKAE_URDF_LEFT = (
    _ROKA_URDF_ROOT / "AR5-5_07L-W4C4A2_description" / "urdf" / "AR5-5_07L-W4C4A2.urdf"
)
_DEFAULT_ROKAE_URDF_RIGHT = (
    _ROKA_URDF_ROOT / "AR5-5_07R-W4C4A2_description" / "urdf" / "AR5-5_07R-W4C4A2.urdf"
)


def _env_or_path(key: str, default: Path) -> str:
    return os.environ.get(key) or str(default)


def default_rokae_urdf_path() -> str:
    """与单臂历史一致：`ROKAE_IK_URDF_PATH` 可覆盖，否则为左臂 L 包 URDF。"""
    return _env_or_path("ROKAE_IK_URDF_PATH", _DEFAULT_ROKAE_URDF_LEFT)


def default_rokae_urdf_path_left() -> str:
    return _env_or_path("ROKAE_IK_URDF_PATH_LEFT", _DEFAULT_ROKAE_URDF_LEFT)


def default_rokae_urdf_path_right() -> str:
    return _env_or_path("ROKAE_IK_URDF_PATH_RIGHT", _DEFAULT_ROKAE_URDF_RIGHT)


# --- 运动学预设：切换机器人时改 CLI / 配置中的 kinematics_preset，无需改源码 ---
KINEMATICS_PRESET_FIXED_AR_DUAL = "fixed_ar_dual"
KINEMATICS_PRESET_WHEELED_AR_DUAL = "wheeled_ar_dual"

_EE_FIXED_TCP_L = "AR5-5_07L-W4C4A2_tcp"
_EE_FIXED_TCP_R = "AR5-5_07R-W4C4A2_tcp"
# 轮式模型 link 名与固定式不同；若与现场 URDF 不一致，请在 teleop 配置里覆盖末端 frame。
_EE_WHEELED_FLAN_L = "AR5-5_08L-W4C1C6-ZY2_flan_link"
_EE_WHEELED_FLAN_R = "AR5-5_08R-W4C1C6-ZY2_flan_link"


def list_kinematics_presets() -> list[str]:
    """可供 `--teleop.kinematics_preset` 使用的预设 id。"""
    return [KINEMATICS_PRESET_FIXED_AR_DUAL, KINEMATICS_PRESET_WHEELED_AR_DUAL]


def _normalize_kinematics_preset(preset: str | None) -> str:
    if preset is None or not str(preset).strip():
        return KINEMATICS_PRESET_FIXED_AR_DUAL
    return str(preset).strip().lower()


def resolve_dual_arm_kinematics(preset: str | None) -> tuple[str, str, str, str]:
    """
    根据预设返回 (左 URDF, 右 URDF, 左末端 link, 右末端 link)。

    URDF 路径始终走 ``default_rokae_urdf_path_left/right``（即仍可由环境变量覆盖）；
    轮式场景通常需将 ``ROKAE_IK_URDF_PATH_LEFT`` / ``RIGHT`` 指向轮式描述包。
    """
    name = _normalize_kinematics_preset(preset)
    if name in ("fixed", "fixed_ar", "fixed_dual"):
        name = KINEMATICS_PRESET_FIXED_AR_DUAL
    if name in ("wheeled", "wheeled_ar", "mobile_ar_dual"):
        name = KINEMATICS_PRESET_WHEELED_AR_DUAL

    l_urdf = default_rokae_urdf_path_left()
    r_urdf = default_rokae_urdf_path_right()
    if name == KINEMATICS_PRESET_FIXED_AR_DUAL:
        return l_urdf, r_urdf, _EE_FIXED_TCP_L, _EE_FIXED_TCP_R
    if name == KINEMATICS_PRESET_WHEELED_AR_DUAL:
        return l_urdf, r_urdf, _EE_WHEELED_FLAN_L, _EE_WHEELED_FLAN_R

    valid = ", ".join(list_kinematics_presets())
    raise ValueError(
        f"未知 kinematics_preset={preset!r}；请使用 {valid}，或在 teleop 中显式填写 URDF/末端 frame。"
    )


def resolve_single_arm_kinematics(preset: str | None) -> tuple[str, str]:
    """
    单臂：返回 (URDF, 末端 link)。URDF 为 ``default_rokae_urdf_path()``（可由 ``ROKAE_IK_URDF_PATH`` 覆盖）。
    """
    name = _normalize_kinematics_preset(preset)
    if name in ("fixed", "fixed_ar", "fixed_dual"):
        name = KINEMATICS_PRESET_FIXED_AR_DUAL
    if name in ("wheeled", "wheeled_ar", "mobile_ar_dual"):
        name = KINEMATICS_PRESET_WHEELED_AR_DUAL

    urdf = default_rokae_urdf_path()
    if name == KINEMATICS_PRESET_FIXED_AR_DUAL:
        return urdf, _EE_FIXED_TCP_L
    if name == KINEMATICS_PRESET_WHEELED_AR_DUAL:
        return urdf, _EE_WHEELED_FLAN_R

    valid = ", ".join(list_kinematics_presets())
    raise ValueError(
        f"未知 kinematics_preset={preset!r}；请使用 {valid}，或在 teleop 中显式填写 urdf_path / end_effector_frame。"
    )


@contextmanager
def cr6_kinematics_session(
    rbv: list[float],
    min_joint: list[float],
    max_joint: list[float],
):
    """双臂 6 轴单臂一步：在该臂 rbv/限位下 `cr_init`，退出时 `de_init`。"""
    rokae_algo.cr_init(rbv, min_joint, max_joint)
    try:
        yield
    finally:
        rokae_algo.de_init()


def configure_pink_7axis(
    ik_state: Any,
    min_joint: list[float],
    max_joint: list[float],
    joint_num: int,
    joint_vel_max: float,
    joint_acc_max: float,
) -> None:
    rk_ik.set_joint_position_limits(
        ik_state,
        np.array(min_joint, dtype=np.float64),
        np.array(max_joint, dtype=np.float64),
    )
    rk_ik.set_joint_velocity_limits(
        ik_state,
        np.array([joint_vel_max] * joint_num, dtype=np.float64),
    )
    rk_ik.set_joint_acceleration_limits(
        ik_state,
        np.array([joint_acc_max] * joint_num, dtype=np.float64),
    )
    rk_ik.set_posture_costs(ik_state, POSTURE_COSTS_7.copy())
    rk_ik.set_configuration_limit_gains(
        ik_state,
        np.array([1.0] * joint_num, dtype=np.float64),
    )


def _ik_6dof(T_flan_in_base: np.ndarray, q_init: list[float]) -> tuple[list[float], bool]:
    end_effector_list = transform_to_quat_wfirst(T_flan_in_base)
    q_solution, ec = rokae_algo.cr6_cart2jnt(q_init, end_effector_list)
    return q_solution, ec == 0


def _pose_to_rpy_xyz(pose_flan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_xyz = pose_flan[:3]
    target_rpy = pose_flan[3:]
    quat_target = R.from_euler(
        "xyz", np.asarray(target_rpy, dtype=np.float64), degrees=False
    ).as_quat()
    quat_wxyz = np.array(
        [quat_target[3], quat_target[0], quat_target[1], quat_target[2]],
        dtype=np.float64,
    )
    return target_xyz, quat_wxyz


def pink_7axis_step(
    ik_state: Any,
    T_flan_in_base: np.ndarray,
    cart_vel: np.ndarray,
    control_period: float,
    trans_max_vel: float,
    joint_num: int,
) -> tuple[list[float], bool]:
    pose_flan = transform_to_pose(T_flan_in_base)
    target_xyz, quat_wxyz = _pose_to_rpy_xyz(pose_flan)
    rk_ik.set_target_pose_wxyz(
        ik_state,
        np.asarray(target_xyz, dtype=np.float64),
        quat_wxyz,
    )
    # joint_cost = np.linalg.norm(cart_vel) / trans_max_vel * POSTURE_COSTS_7.copy()
    scale = np.linalg.norm(cart_vel) / trans_max_vel
    joint_cost = max(scale, 0.01) * POSTURE_COSTS_7.copy()

    rk_ik.set_posture_costs(ik_state, joint_cost)
    ik_out = rk_ik.step_and_get(ik_state, float(control_period))
    q_solution = ik_out["q"].tolist()[:joint_num]
    ok = np.all(np.isfinite(q_solution))
    return q_solution, ok


def pink_7axis_step_or_fallback(
    ik_state: Any,
    T_flan_in_base: np.ndarray,
    q_current: np.ndarray,
    cart_vel: np.ndarray,
    control_period: float,
    trans_max_vel: float,
    joint_num: int,
    *,
    log_prefix: str = "[InverseKinematicsProcessor]",
) -> tuple[list[float], bool]:
    try:
        return pink_7axis_step(
            ik_state,
            T_flan_in_base,
            cart_vel,
            control_period,
            trans_max_vel,
            joint_num,
        )
    except Exception as e:
        print(f"{log_prefix} 7轴 step 失败: {e}")
        return q_current.tolist(), False


def solve_ik(
    joint_num: int,
    T_flan_in_base_new: np.ndarray,
    q_current: np.ndarray,
    ik_state: Any | None,
    cart_vel: np.ndarray,
    control_period: float,
    trans_max_vel: float,
    *,
    log_prefix: str = "[InverseKinematicsProcessor]",
) -> tuple[list[float], bool]:
    q_init = q_current.tolist()
    if joint_num == 6:
        return _ik_6dof(T_flan_in_base_new, q_init)
    if joint_num == 7 and ik_state is not None:
        return pink_7axis_step_or_fallback(
            ik_state,
            T_flan_in_base_new,
            q_current,
            cart_vel,
            control_period,
            trans_max_vel,
            joint_num,
            log_prefix=log_prefix,
        )
    return q_init, False


def reset_pink_if_flagged(
    ik_state: Any, q_current: np.ndarray, reset_flag: bool
) -> bool:
    """reset_flag 为真时用当前关节角同步 Pink（仅 7 轴有 ik_state）；返回是否本帧为复位后首帧。"""
    if not reset_flag:
        return False
    if ik_state is not None:
        rk_ik.reset_configuration(ik_state, q_current)
    return True


def integrate_twist_then_solve_ik(
    joint_num: int,
    cart_vel: np.ndarray,
    q_current: np.ndarray,
    cart_pos_real_end_in_base: np.ndarray,
    psi: float,
    tool_end_pos: np.ndarray,
    tool_ref_pos: np.ndarray,
    base_frame_in_world: np.ndarray,
    cart_pose_flan_in_base: np.ndarray | None,
    control_period: float,
    trans_max_vel: float,
    ik_state: Any | None,
    *,
    log_prefix: str = "[InverseKinematicsProcessor]",
) -> tuple[list[float], np.ndarray, np.ndarray | None, float | None]:
    """
    ref 系 twist 积分后做逆解（6 轴 cr6，7 轴 Pink）。

    若逆解失败，关节保持当前值，法兰积分状态不提交（与单臂 `InverseKinematicsProcessor` 一致）。

    返回:
        q_solution, cart_pos_cmd_end_in_ref, 更新后的 flange 位姿（失败则为 None 表示不更新缓存）,
        仅在首次从观测建立法兰位姿且 joint_num==7 时返回 elbow（psi），否则为 None。
    """
    T_end_in_flan, T_flan_in_end, T_ref_in_base, T_base_in_ref = compute_tool_chain(
        tool_end_pos, tool_ref_pos, base_frame_in_world
    )

    elbow_out: float | None = None
    if cart_pose_flan_in_base is None:
        cart_pose_flan_in_base = flan_pose_from_end_observation(
            cart_pos_real_end_in_base, T_flan_in_end
        )
        if joint_num == 7:
            elbow_out = float(psi)

    assert cart_pose_flan_in_base is not None
    T_flan_in_base_new, cart_pos_cmd_end_in_ref = integrate_end_in_ref(
        T_base_in_ref,
        T_ref_in_base,
        cart_pose_flan_in_base,
        T_end_in_flan,
        T_flan_in_end,
        cart_vel,
        control_period,
    )

    q_solution, ik_ok = solve_ik(
        joint_num,
        T_flan_in_base_new,
        q_current,
        ik_state,
        cart_vel,
        control_period,
        trans_max_vel,
        log_prefix=log_prefix,
    )

    new_flange_pose: np.ndarray | None = None
    if ik_ok:
        new_flange_pose = transform_to_pose(T_flan_in_base_new)
    else:
        q_solution = q_current.tolist()

    return q_solution, cart_pos_cmd_end_in_ref, new_flange_pose, elbow_out
