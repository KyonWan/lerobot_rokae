"""SpaceMouse 笛卡尔速度 → 关节指令；6 轴 rokae_algo，7 轴 Pink IK（rokae_ik_interface）。"""

from lerobot.processor.pipeline import ProcessorStepRegistry, ProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from rokae_python_wrapper.rokae_kinematics import rokae_algo
from rokae_python_wrapper.rokae_kinematics import rokae_ik_interface as rk_ik

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    pose_to_transform,
    transform_to_pose,
)

from ..cartesian_ik_helpers import (
    compute_tool_chain,
    flan_pose_from_end_observation,
    integrate_end_in_ref,
    obs_cart_pos_end_in_base,
    obs_joint_vector,
    scale_action_cart_vel,
    transform_to_quat_wfirst as _transform_to_quat_wfirst,
    write_arm_action_fields,
)

# 7 轴 pink IK 关节速度/加速度上限（rad/s、rad/s²），与笛卡尔 xd_limit_abs 分开约束
_SNS_JOINT_VEL_MAX = 1.0
_SNS_JOINT_ACC_MAX = 10.0

_POSTURE_COSTS_7 = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 10e-3, 10e-3])

_ROKA_URDF_ROOT = Path(rk_ik.__file__).resolve().parent / "rokae_urdf"
_DEFAULT_ROKAE_URDF = (
    _ROKA_URDF_ROOT / "AR5-5_07L-W4C4A2_description" / "urdf" / "AR5-5_07L-W4C4A2.urdf"
)


def _env_or_path(key: str, default: Path) -> str:
    return os.environ.get(key) or str(default)


def _default_rokae_urdf_path() -> str:
    return _env_or_path("ROKAE_IK_URDF_PATH", _DEFAULT_ROKAE_URDF)


def _end_effector_to_cart_pose_6(end_effector: list[float]) -> np.ndarray:
    """将正解返回值转为内部 [x,y,z,rx,ry,rz]（欧拉 xyz）。
    支持 end_effector 为 7 维 [x,y,z,qw,qx,qy,qz]（四元数 w first）或 6 维 [x,y,z,rx,ry,rz]。"""
    arr = np.array(end_effector, dtype=np.float64)
    if len(arr) >= 7:
        out = np.zeros(6, dtype=np.float64)
        out[:3] = arr[:3]
        r = R.from_quat([arr[4], arr[5], arr[6], arr[3]])
        out[3:] = r.as_euler("xyz", degrees=False)
        return out
    if len(arr) >= 6:
        return np.array(arr[:6], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def _cart_pose_6_to_quat_wfirst(cart_pose: np.ndarray) -> list[float]:
    """将内部 [x,y,z,rx,ry,rz] 转为逆解接口需要的 [x,y,z,qw,qx,qy,qz]（四元数 w first）。"""
    r = R.from_euler("xyz", cart_pose[3:], degrees=False)
    q = r.as_quat()
    return list(cart_pose[:3]) + [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def update_gripper_state_from_buttons(
    buttons: list | tuple,
    button_prev: int,
    gripper_state: int,
) -> tuple[int, int]:
    button_curr = 0
    if isinstance(buttons, (list, tuple)) and len(buttons) > 0:
        button_curr = 1 if buttons[0] else 0
    if button_curr == 1 and button_prev == 0:
        gripper_state = 1 - gripper_state
    return button_curr, gripper_state


def _configure_pink_7axis(
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
    rk_ik.set_posture_costs(ik_state, _POSTURE_COSTS_7.copy())
    rk_ik.set_configuration_limit_gains(
        ik_state,
        np.array([1.0] * joint_num, dtype=np.float64),
    )


def _ik_6dof(T_flan_in_base: np.ndarray, q_init: list[float]) -> tuple[list[float], bool]:
    end_effector_list = _transform_to_quat_wfirst(T_flan_in_base)
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


def _pink_7axis_step(
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
    joint_cost = (
        np.linalg.norm(cart_vel) / trans_max_vel * _POSTURE_COSTS_7.copy()
    )
    rk_ik.set_posture_costs(ik_state, joint_cost)
    ik_out = rk_ik.step_and_get(ik_state, float(control_period))
    q_solution = ik_out["q"].tolist()[:joint_num]
    ok = np.all(np.isfinite(q_solution))
    return q_solution, ok


def _pink_7axis_step_or_fallback(
    ik_state: Any,
    T_flan_in_base: np.ndarray,
    q_current: np.ndarray,
    cart_vel: np.ndarray,
    control_period: float,
    trans_max_vel: float,
    joint_num: int,
) -> tuple[list[float], bool]:
    try:
        return _pink_7axis_step(
            ik_state,
            T_flan_in_base,
            cart_vel,
            control_period,
            trans_max_vel,
            joint_num,
        )
    except Exception as e:
        print(f"[InverseKinematicsProcessor] 7轴 step 失败: {e}")
        return q_current.tolist(), False


def _solve_ik(
    joint_num: int,
    T_flan_in_base_new: np.ndarray,
    q_current: np.ndarray,
    ik_state: Any | None,
    cart_vel: np.ndarray,
    control_period: float,
    trans_max_vel: float,
) -> tuple[list[float], bool]:
    q_init = q_current.tolist()
    if joint_num == 6:
        return _ik_6dof(T_flan_in_base_new, q_init)
    if joint_num == 7 and ik_state is not None:
        return _pink_7axis_step_or_fallback(
            ik_state,
            T_flan_in_base_new,
            q_current,
            cart_vel,
            control_period,
            trans_max_vel,
            joint_num,
        )
    return q_init, False


def _reset_pink_if_flagged(
    ik_state: Any, q_current: np.ndarray, reset_flag: bool
) -> bool:
    """reset_flag 为真时用当前关节角同步 Pink（仅 7 轴有 ik_state）；返回是否本帧为复位后首帧。"""
    if not reset_flag:
        return False
    if ik_state is not None:
        rk_ik.reset_configuration(ik_state, q_current)
    return True


def _fill_fallback_action(
    transition: Transition, joint_num: int, gripper_state: int
) -> Transition:
    transition = transition.copy()
    obs = transition.get(TransitionKey.OBSERVATION)
    action = transition.get(TransitionKey.ACTION)
    for i in range(joint_num):
        action.setdefault(f"joint_pos{i}", float(obs[f"joint_pos{i}"]))
    action.setdefault("gripper_pos", float(gripper_state))
    transition[TransitionKey.ACTION] = action
    return transition


@ProcessorStepRegistry.register("inverse_kinematics_processor")
@dataclass
class InverseKinematicsProcessor(ProcessorStep):
    """
    笛卡尔速度积分后做逆解：6 轴为 rokae_algo，7 轴为 Pink（rokae_ik_interface）。

    rbv、min_joint、max_joint 由 rokae 配置提供；7 轴需 URDF 与末端 frame 名。
    """
    trans_max_vel: float
    rot_max_vel: float
    joint_num: int = 6
    control_period: float = 1.0 / 30
    initial_gripper_state: int = 1
    rbv: list[float] = field(default_factory=list)
    min_joint: list[float] = field(default_factory=list)
    max_joint: list[float] = field(default_factory=list)

    tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    base_frame_in_world: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    urdf_path: str = field(default_factory=_default_rokae_urdf_path)
    end_effector_frame: str = "AR5-5_07L-W4C4A2_tcp"
    joint_vel_max: float = _SNS_JOINT_VEL_MAX
    joint_acc_max: float = _SNS_JOINT_ACC_MAX

    _cart_pos_flan_in_base: np.ndarray | None = field(default=None, init=False, repr=False)
    _elbow: float = field(default=0.0, init=False, repr=False)
    _inited: bool = field(default=False, init=False, repr=False)
    _ik_state: Any = field(default=None, init=False, repr=False)
    button_prev: int = field(default=0, init=False, repr=False)
    gripper_state: int = field(default=1, init=False, repr=False)
    reset_ik_state: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.gripper_state = self.initial_gripper_state
        if not self.rbv or not self.min_joint or not self.max_joint:
            self._inited = False
            return
        try:
            self._inited = self._init_kinematics_backend()
        except Exception as e:
            print(f"[InverseKinematicsProcessor] 7轴初始化失败: {e} ")
            self._ik_state = None
            self._inited = False

    def _init_kinematics_backend(self) -> bool:
        if self.joint_num == 6:
            return rokae_algo.cr_init(self.rbv, self.min_joint, self.max_joint)
        if self.joint_num != 7:
            return False
        self._ik_state = rk_ik.create_solver_state(
            urdf_path=self.urdf_path,
            end_effector_frame=self.end_effector_frame,
        )
        _configure_pink_7axis(
            self._ik_state,
            self.min_joint,
            self.max_joint,
            self.joint_num,
            self.joint_vel_max,
            self.joint_acc_max,
        )
        ok = self._ik_state is not None
        if ok:
            # 首帧与真实关节对齐（create_solver_state 内 q 为限位中点）
            self.reset_ik_state = True
        return ok

    def __del__(self) -> None:
        if not getattr(self, "_inited", False):
            return
        try:
            jn = getattr(self, "joint_num", 6)
            if jn == 7:
                self._ik_state = None
            elif jn == 6:
                rokae_algo.de_init()
        except Exception:
            pass

    def reset(self, gripper_state: int | None = None) -> None:
        self._cart_pos_flan_in_base = None
        self._elbow = 0.0
        self.button_prev = 0
        self.gripper_state = (
            gripper_state if gripper_state is not None else self.initial_gripper_state
        )
        self.reset_ik_state = True

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited:
            return _fill_fallback_action(transition, self.joint_num, self.gripper_state)

        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        buttons = action.pop("buttons", [0, 0])
        self.button_prev, self.gripper_state = update_gripper_state_from_buttons(
            buttons=buttons,
            button_prev=self.button_prev,
            gripper_state=self.gripper_state,
        )

        q_current = obs_joint_vector(obs, self.joint_num, "")
        reset_consumed = _reset_pink_if_flagged(
            self._ik_state, q_current, self.reset_ik_state
        )
        if reset_consumed:
            self.reset_ik_state = False

        cart_pos_real = obs_cart_pos_end_in_base(obs, "")
        psi = float(obs.get("psi", 0.0))
        cart_vel = scale_action_cart_vel(
            action, self.trans_max_vel, self.rot_max_vel, ""
        )
        # episode reset 后 _cart_pos_flan_in_base 已清空，首帧若仍用残留 cart_vel 积分，
        # 会与刚同步的关节/Pink 状态打架，出现关节抖一下；本帧只做位姿对齐不跟 twist。
        if reset_consumed:
            cart_vel = np.zeros(6, dtype=np.float64)

        q_sol, cart_cmd = self._process_coordinate_transform_and_ik(
            cart_vel, q_current, cart_pos_real, psi
        )
        write_arm_action_fields(
            action,
            "",
            cart_cmd,
            q_sol,
            cart_vel,
            psi,
            self.joint_num,
            self.gripper_state,
        )
        transition[TransitionKey.ACTION] = action
        return transition

    def _process_coordinate_transform_and_ik(
        self,
        cart_vel: np.ndarray,
        q_current: np.ndarray,
        cart_pos_real_end_in_base: np.ndarray,
        psi: float = 0.0,
    ) -> tuple[list[float], np.ndarray]:
        T_end_in_flan, T_flan_in_end, T_ref_in_base, T_base_in_ref = compute_tool_chain(
            self.tool_end_pos, self.tool_ref_pos, self.base_frame_in_world
        )

        if self._cart_pos_flan_in_base is None:
            self._cart_pos_flan_in_base = flan_pose_from_end_observation(
                cart_pos_real_end_in_base, T_flan_in_end
            )
            if self.joint_num == 7:
                self._elbow = float(psi)

        assert self._cart_pos_flan_in_base is not None
        T_flan_in_base_new, cart_pos_cmd_end_in_ref = integrate_end_in_ref(
            T_base_in_ref,
            T_ref_in_base,
            self._cart_pos_flan_in_base,
            T_end_in_flan,
            T_flan_in_end,
            cart_vel,
            self.control_period,
        )

        q_solution, ik_ok = _solve_ik(
            self.joint_num,
            T_flan_in_base_new,
            q_current,
            self._ik_state,
            cart_vel,
            self.control_period,
            self.trans_max_vel,
        )

        if ik_ok:
            self._cart_pos_flan_in_base = transform_to_pose(T_flan_in_base_new)
        else:
            q_solution = q_current.tolist()

        return q_solution, cart_pos_cmd_end_in_ref

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for i in range(self.joint_num):
            action_features.setdefault(f"joint_pos{i}", float)
        for i in range(6):
            action_features.setdefault(f"cart_pos{i}", float)
            action_features.setdefault(f"cart_vel{i}", float)
        action_features.setdefault("psi", float)
        action_features.setdefault("gripper_pos", float)
        return features
