"""
双臂 SpaceMouse：左右臂各自一套笛卡尔速度积分 + 逆解（6 轴 rokae_algo cr6，7 轴 Pink），键名为 left_* / right_*。

几何与观测解析与单臂共用 `cartesian_ik_helpers`。7 轴左右各一 `IKState`；6 轴因 `rokae_algo` 全局单例，每帧对该臂使用
`cr6_kinematics_session`（cr_init→一步→de_init），与单臂「积分 + cr6」一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition

from rokae_python_wrapper.rokae_kinematics import rokae_ik_interface as rk_ik

from ..cartesian_ik_helpers import (
    obs_cart_pos_end_in_base,
    obs_joint_vector,
    scale_action_cart_vel,
    write_arm_action_fields,
)
from ..pink_ik_helpers import (
    SNS_JOINT_ACC_MAX,
    SNS_JOINT_VEL_MAX,
    configure_pink_7axis,
    cr6_kinematics_session,
    default_rokae_urdf_path_left,
    default_rokae_urdf_path_right,
    integrate_twist_then_solve_ik,
    reset_pink_if_flagged,
)
from ..spacemouse.spacemouse_processor import update_gripper_state_from_buttons


def _register_arm_action_features(
    action_features: dict,
    key_prefix: str,
    joint_num: int,
) -> None:
    """声明单臂侧 action 特征类型（用于数据集 schema）。"""
    p = key_prefix
    for i in range(joint_num):
        action_features.setdefault(f"{p}joint_pos{i}", float)
    for i in range(6):
        action_features.setdefault(f"{p}cart_pos{i}", float)
        action_features.setdefault(f"{p}cart_vel{i}", float)
    action_features.setdefault(f"{p}psi", float)
    action_features.setdefault(f"{p}gripper_pos", float)


@ProcessorStepRegistry.register("bi_inverse_kinematics_processor")
@dataclass
class BiInverseKinematicsProcessor(ProcessorStep):
    """
    双臂逆解处理器：与单臂相同的 ref 系 twist 积分链。
    每侧 `left_joint_num` / `right_joint_num` 仅可为 6 或 7；6 轴为 cr6，7 轴为 Pink。
    """

    control_period: float
    trans_max_vel: float
    rot_max_vel: float
    left_joint_num: int = 7
    right_joint_num: int = 7
    initial_left_gripper_state: int = 1
    initial_right_gripper_state: int = 1

    left_rbv: list[float] = field(default_factory=list)
    right_rbv: list[float] = field(default_factory=list)
    left_min_joint: list[float] = field(default_factory=list)
    left_max_joint: list[float] = field(default_factory=list)
    right_min_joint: list[float] = field(default_factory=list)
    right_max_joint: list[float] = field(default_factory=list)

    left_urdf_path: str = field(default_factory=default_rokae_urdf_path_left)
    right_urdf_path: str = field(default_factory=default_rokae_urdf_path_right)
    left_end_effector_frame: str = "AR5-5_07L-W4C4A2_tcp"
    right_end_effector_frame: str = "AR5-5_07R-W4C4A2_tcp"
    joint_vel_max: float = SNS_JOINT_VEL_MAX
    joint_acc_max: float = SNS_JOINT_ACC_MAX

    left_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_base_frame_in_world: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )

    right_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_base_frame_in_world: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )

    _left_cart_pos_flan_in_base: np.ndarray | None = field(default=None, init=False, repr=False)
    _right_cart_pos_flan_in_base: np.ndarray | None = field(default=None, init=False, repr=False)
    _left_elbow: float | None = field(default=None, init=False, repr=False)
    _right_elbow: float | None = field(default=None, init=False, repr=False)
    _left_ik_state: Any = field(default=None, init=False, repr=False)
    _right_ik_state: Any = field(default=None, init=False, repr=False)
    _inited: bool = field(default=False, init=False, repr=False)
    left_button_prev: int = field(default=0, init=False, repr=False)
    right_button_prev: int = field(default=0, init=False, repr=False)
    left_gripper_state: int = field(default=1, init=False, repr=False)
    right_gripper_state: int = field(default=1, init=False, repr=False)
    reset_left_ik: bool = field(default=False, init=False, repr=False)
    reset_right_ik: bool = field(default=False, init=False, repr=False)
    # reset() 后首帧将 cart_vel 置零，避免法兰位姿重建后仍积分残留 twist 导致双臂抖一下
    _suppress_cart_vel_once: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.left_gripper_state = self.initial_left_gripper_state
        self.right_gripper_state = self.initial_right_gripper_state
        if self.left_joint_num not in (6, 7) or self.right_joint_num not in (6, 7):
            raise ValueError(
                "BiInverseKinematicsProcessor 每侧仅支持 6 或 7 轴；"
                f"got left={self.left_joint_num}, right={self.right_joint_num}"
            )
        if not self.left_rbv or not self.left_min_joint or not self.left_max_joint:
            self._inited = False
            return
        if not self.right_rbv or not self.right_min_joint or not self.right_max_joint:
            self._inited = False
            return
        try:
            self._inited = self._init_kinematics_backend()
        except Exception as e:
            print(f"[BiInverseKinematicsProcessor] 运动学初始化失败: {e}")
            self._left_ik_state = None
            self._right_ik_state = None
            self._inited = False

    def _init_kinematics_backend(self) -> bool:
        self._left_ik_state = None
        self._right_ik_state = None

        if self.left_joint_num == 7:
            self._left_ik_state = rk_ik.create_solver_state(
                urdf_path=self.left_urdf_path,
                end_effector_frame=self.left_end_effector_frame,
            )
            configure_pink_7axis(
                self._left_ik_state,
                self.left_min_joint,
                self.left_max_joint,
                self.left_joint_num,
                self.joint_vel_max,
                self.joint_acc_max,
            )
        if self.right_joint_num == 7:
            self._right_ik_state = rk_ik.create_solver_state(
                urdf_path=self.right_urdf_path,
                end_effector_frame=self.right_end_effector_frame,
            )
            configure_pink_7axis(
                self._right_ik_state,
                self.right_min_joint,
                self.right_max_joint,
                self.right_joint_num,
                self.joint_vel_max,
                self.joint_acc_max,
            )

        if self.left_joint_num == 7:
            self.reset_left_ik = True
        if self.right_joint_num == 7:
            self.reset_right_ik = True
        return True

    def __del__(self) -> None:
        try:
            self._left_ik_state = None
            self._right_ik_state = None
        except Exception:
            pass

    def reset(
        self,
        left_gripper_state: int | None = None,
        right_gripper_state: int | None = None,
    ) -> None:
        self._left_cart_pos_flan_in_base = None
        self._right_cart_pos_flan_in_base = None
        self._left_elbow = None
        self._right_elbow = None
        self.left_button_prev = 0
        self.right_button_prev = 0
        self.left_gripper_state = (
            left_gripper_state
            if left_gripper_state is not None
            else self.initial_left_gripper_state
        )
        self.right_gripper_state = (
            right_gripper_state
            if right_gripper_state is not None
            else self.initial_right_gripper_state
        )
        self.reset_left_ik = True
        self.reset_right_ik = True
        self._suppress_cart_vel_once = True

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited:
            return transition

        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        left_buttons = action.pop("left_buttons", [0, 0])
        right_buttons = action.pop("right_buttons", [0, 0])
        self.left_button_prev, self.left_gripper_state = update_gripper_state_from_buttons(
            left_buttons, self.left_button_prev, self.left_gripper_state
        )
        self.right_button_prev, self.right_gripper_state = update_gripper_state_from_buttons(
            right_buttons, self.right_button_prev, self.right_gripper_state
        )

        left_q = obs_joint_vector(obs, self.left_joint_num, "left_")
        right_q = obs_joint_vector(obs, self.right_joint_num, "right_")
        left_reset = reset_pink_if_flagged(
            self._left_ik_state, left_q, self.reset_left_ik
        )
        right_reset = reset_pink_if_flagged(
            self._right_ik_state, right_q, self.reset_right_ik
        )
        if left_reset:
            self.reset_left_ik = False
        if right_reset:
            self.reset_right_ik = False

        left_cart_real = obs_cart_pos_end_in_base(obs, "left_")
        right_cart_real = obs_cart_pos_end_in_base(obs, "right_")
        left_psi = float(obs.get("left_psi", 0.0))
        right_psi = float(obs.get("right_psi", 0.0))

        left_vel = scale_action_cart_vel(
            action, self.trans_max_vel, self.rot_max_vel, "left_"
        )
        right_vel = scale_action_cart_vel(
            action, self.trans_max_vel, self.rot_max_vel, "right_"
        )
        if self._suppress_cart_vel_once:
            left_vel = np.zeros(6, dtype=np.float64)
            right_vel = np.zeros(6, dtype=np.float64)
            self._suppress_cart_vel_once = False
        if left_reset:
            left_vel = np.zeros(6, dtype=np.float64)
        if right_reset:
            right_vel = np.zeros(6, dtype=np.float64)

        if self.left_joint_num == 6:
            with cr6_kinematics_session(
                self.left_rbv, self.left_min_joint, self.left_max_joint
            ):
                left_q_sol, left_cart_cmd, new_left_fl, elbow_out_l = (
                    integrate_twist_then_solve_ik(
                        6,
                        left_vel,
                        left_q,
                        left_cart_real,
                        left_psi,
                        self.left_tool_end_pos,
                        self.left_tool_ref_pos,
                        self.left_base_frame_in_world,
                        self._left_cart_pos_flan_in_base,
                        self.control_period,
                        self.trans_max_vel,
                        None,
                        log_prefix="[BiInverseKinematicsProcessor] left",
                    )
                )
        else:
            left_q_sol, left_cart_cmd, new_left_fl, elbow_out_l = (
                integrate_twist_then_solve_ik(
                    self.left_joint_num,
                    left_vel,
                    left_q,
                    left_cart_real,
                    left_psi,
                    self.left_tool_end_pos,
                    self.left_tool_ref_pos,
                    self.left_base_frame_in_world,
                    self._left_cart_pos_flan_in_base,
                    self.control_period,
                    self.trans_max_vel,
                    self._left_ik_state,
                    log_prefix="[BiInverseKinematicsProcessor] left",
                )
            )
        if elbow_out_l is not None:
            self._left_elbow = elbow_out_l
        if new_left_fl is not None:
            self._left_cart_pos_flan_in_base = new_left_fl

        if self.right_joint_num == 6:
            with cr6_kinematics_session(
                self.right_rbv, self.right_min_joint, self.right_max_joint
            ):
                right_q_sol, right_cart_cmd, new_right_fl, elbow_out_r = (
                    integrate_twist_then_solve_ik(
                        6,
                        right_vel,
                        right_q,
                        right_cart_real,
                        right_psi,
                        self.right_tool_end_pos,
                        self.right_tool_ref_pos,
                        self.right_base_frame_in_world,
                        self._right_cart_pos_flan_in_base,
                        self.control_period,
                        self.trans_max_vel,
                        None,
                        log_prefix="[BiInverseKinematicsProcessor] right",
                    )
                )
        else:
            right_q_sol, right_cart_cmd, new_right_fl, elbow_out_r = (
                integrate_twist_then_solve_ik(
                    self.right_joint_num,
                    right_vel,
                    right_q,
                    right_cart_real,
                    right_psi,
                    self.right_tool_end_pos,
                    self.right_tool_ref_pos,
                    self.right_base_frame_in_world,
                    self._right_cart_pos_flan_in_base,
                    self.control_period,
                    self.trans_max_vel,
                    self._right_ik_state,
                    log_prefix="[BiInverseKinematicsProcessor] right",
                )
            )
        if elbow_out_r is not None:
            self._right_elbow = elbow_out_r
        if new_right_fl is not None:
            self._right_cart_pos_flan_in_base = new_right_fl

        write_arm_action_fields(
            action,
            "left_",
            left_cart_cmd,
            left_q_sol,
            left_vel,
            left_psi,
            self.left_joint_num,
            self.left_gripper_state,
        )
        write_arm_action_fields(
            action,
            "right_",
            right_cart_cmd,
            right_q_sol,
            right_vel,
            right_psi,
            self.right_joint_num,
            self.right_gripper_state,
        )

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        _register_arm_action_features(action_features, "left_", self.left_joint_num)
        _register_arm_action_features(action_features, "right_", self.right_joint_num)
        return features
