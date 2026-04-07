"""
笛卡尔阻抗链、ref 系速度积分与观测键解析。

供单臂 `spacemouse_processor` 与双臂 `bi_spacemouse_processor` 共用，避免重复实现同一套
end/flange/ref/base 变换与 cart_vel 缩放逻辑（逆解后端见 `pink_ik_helpers` / `run_cross_wrist7_arm_step`）。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    compute_base_ref_transform,
    inv_homogeneous,
    pose_to_transform,
    transform_to_pose,
)


def transform_to_quat_wfirst(T: np.ndarray) -> list[float]:
    """齐次矩阵 → 逆解常用 [x,y,z,qw,qx,qy,qz]（scipy 四元数为 xyzw，此处输出 w-first）。"""
    pos = T[:3, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def compute_tool_chain(
    tool_end_pos: np.ndarray,
    tool_ref_pos: np.ndarray,
    base_frame_in_world: np.ndarray,
):
    """end/flange 与 ref/base 相对关系：返回 (T_end_in_flan, T_flan_in_end, T_ref_in_base, T_base_in_ref)。"""
    T_end_in_flan = pose_to_transform(tool_end_pos)
    T_flan_in_end = inv_homogeneous(T_end_in_flan)
    T_ref_in_base, _ = compute_base_ref_transform(tool_ref_pos, base_frame_in_world)
    T_base_in_ref = inv_homogeneous(T_ref_in_base)
    return T_end_in_flan, T_flan_in_end, T_ref_in_base, T_base_in_ref


def integrate_end_in_ref(
    T_base_in_ref: np.ndarray,
    T_ref_in_base: np.ndarray,
    cart_pose_flan_in_base: np.ndarray,
    T_end_in_flan: np.ndarray,
    T_flan_in_end: np.ndarray,
    cart_vel: np.ndarray,
    control_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在 ref 下按 twist 积分一步，得到新 T_flan_in_base 与 end_in_ref 的 6D 位姿。"""
    T_flan_in_base = pose_to_transform(cart_pose_flan_in_base)
    T_end_in_ref = T_base_in_ref @ T_flan_in_base @ T_end_in_flan
    T_end_in_ref_new = T_end_in_ref.copy()
    T_end_in_ref_new[:3, 3] += cart_vel[:3] * control_period
    R_rot = R.from_rotvec(cart_vel[3:] * control_period).as_matrix()
    T_end_in_ref_new[:3, :3] = R_rot @ T_end_in_ref_new[:3, :3]
    cart_pos_cmd_end_in_ref = transform_to_pose(T_end_in_ref_new)
    T_flan_in_base_new = T_ref_in_base @ T_end_in_ref_new @ T_flan_in_end
    return T_flan_in_base_new, cart_pos_cmd_end_in_ref


def flan_pose_from_end_observation(
    cart_pos_real_end_in_base: np.ndarray,
    T_flan_in_end: np.ndarray,
) -> np.ndarray:
    """由 end_in_base 与 flange/end 链得到法兰在基座标系下的 [x,y,z,rx,ry,rz]。"""
    T_end_in_base = pose_to_transform(cart_pos_real_end_in_base)
    return transform_to_pose(T_end_in_base @ T_flan_in_end)


def scale_action_cart_vel(
    action: dict,
    trans_max_vel: float,
    rot_max_vel: float,
    key_prefix: str = "",
) -> np.ndarray:
    """从 action 读取 cart_vel0..5，再按线速度/角速度上限缩放（twist 在 ref 系）。"""
    p = key_prefix
    cart_vel = np.array(
        [action.get(f"{p}cart_vel{i}", 0.0) for i in range(6)],
        dtype=np.float64,
    )
    cart_vel[:3] *= float(trans_max_vel)
    cart_vel[3:] *= float(rot_max_vel)
    return cart_vel


def obs_joint_vector(obs: dict, n: int, key_prefix: str = "") -> np.ndarray:
    """观测中的关节角向量；key_prefix 为 '' 时键为 joint_pos{i}，为 'left_' 时为 left_joint_pos{i}。"""
    p = key_prefix
    return np.array([obs[f"{p}joint_pos{i}"] for i in range(n)], dtype=np.float64)


def obs_cart_pos_end_in_base(obs: dict, key_prefix: str = "") -> np.ndarray:
    """end 相对 base 的笛卡尔位姿；缺省键时填 0.0。"""
    p = key_prefix
    return np.array(
        [obs.get(f"{p}cart_pos{i}", 0.0) for i in range(6)],
        dtype=np.float64,
    )


def write_arm_action_fields(
    action: dict,
    key_prefix: str,
    cart_pos_cmd_end_in_ref: np.ndarray,
    q_solution: np.ndarray | list[float],
    cart_vel: np.ndarray,
    psi: float,
    joint_num: int,
    gripper_state: int,
) -> None:
    """写入单臂 action：cart_pos / joint_pos / cart_vel / psi / gripper_pos（prefix 为空时键与单臂一致）。"""
    p = key_prefix
    for i in range(6):
        action[f"{p}cart_pos{i}"] = float(cart_pos_cmd_end_in_ref[i])
    for i in range(joint_num):
        action[f"{p}joint_pos{i}"] = float(q_solution[i])
    for i in range(6):
        action[f"{p}cart_vel{i}"] = float(cart_vel[i])
    action[f"{p}psi"] = float(psi)
    action[f"{p}gripper_pos"] = float(gripper_state)


def run_cross_wrist7_arm_step(
    rokae_algo,
    q_current: np.ndarray,
    cart_vel: np.ndarray,
    cart_pose_flan_in_base: np.ndarray | None,
    elbow: float | None,
    cart_pos_real_end_in_base: np.ndarray,
    psi: float,
    tool_end_pos: np.ndarray,
    tool_ref_pos: np.ndarray,
    base_frame_in_world: np.ndarray,
    joint_num: int,
    control_period: float,
) -> tuple[np.ndarray | None, np.ndarray, float | None, np.ndarray]:
    """
    单臂一步：twist 在 ref 下积分 → base 下法兰目标 → cross_wrist7_cart2jnt。

    调用前须已对同一 rbv 执行过 rokae_algo.cross_wrist7_init。

    返回:
        (新 cart_pose_flan_in_base, q_solution[joint_num], 新 elbow, cart_pos_cmd_end_in_ref)
    """
    T_end_in_flan, T_flan_in_end, T_ref_in_base, T_base_in_ref = compute_tool_chain(
        tool_end_pos, tool_ref_pos, base_frame_in_world
    )

    if cart_pose_flan_in_base is None:
        cart_pose_flan_in_base = flan_pose_from_end_observation(
            cart_pos_real_end_in_base, T_flan_in_end
        )
        if joint_num == 7:
            elbow = float(psi)

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

    end_effector_list = transform_to_quat_wfirst(T_flan_in_base_new)
    psi_cmd = float(elbow) if elbow is not None else 0.0
    q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(
        q_current.tolist(), end_effector_list, psi_cmd
    )
    if ec == 0:
        cart_pose_flan_in_base = transform_to_pose(T_flan_in_base_new)
    else:
        q_solution = q_current.tolist()

    q_arr = np.array(q_solution[:joint_num], dtype=np.float64)
    return cart_pose_flan_in_base, q_arr, elbow, cart_pos_cmd_end_in_ref
