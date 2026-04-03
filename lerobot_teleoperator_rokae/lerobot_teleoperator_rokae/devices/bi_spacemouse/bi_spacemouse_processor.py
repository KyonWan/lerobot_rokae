"""
双臂 SpaceMouse：左右臂各自一套笛卡尔速度积分 + `cross_wrist7` 逆解，键名为 left_* / right_*。

几何与观测解析与单臂共用 `cartesian_ik_helpers`；每条臂在逆解前分别 `cross_wrist7_init`，结束后 `de_init`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition

try:
    from rokae_python_wrapper.rokae_kinematics import rokae_algo
except ImportError:
    rokae_algo = None

from ..cartesian_ik_helpers import (
    obs_cart_pos_end_in_base,
    obs_joint_vector,
    run_cross_wrist7_arm_step,
    scale_action_cart_vel,
    write_arm_action_fields,
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
    双臂逆解处理器：与单臂相同的 ref 系 twist 积分链，逆解为 `rokae_algo.cross_wrist7_cart2jnt`。

    左右臂参数、工具链、内部法兰位姿缓存相互独立；每帧顺序为左臂 init → 左臂 IK → de_init → 右臂同理。
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
    _inited: bool = field(default=False, init=False, repr=False)
    left_button_prev: int = field(default=0, init=False, repr=False)
    right_button_prev: int = field(default=0, init=False, repr=False)
    left_gripper_state: int = field(default=1, init=False, repr=False)
    right_gripper_state: int = field(default=1, init=False, repr=False)
    # reset() 后首帧将 cart_vel 置零，避免法兰位姿重建后仍积分残留 twist 导致双臂抖一下
    _suppress_cart_vel_once: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.left_gripper_state = self.initial_left_gripper_state
        self.right_gripper_state = self.initial_right_gripper_state
        self._inited = (
            rokae_algo is not None
            and bool(self.left_rbv and self.left_min_joint and self.left_max_joint)
            and bool(self.right_rbv and self.right_min_joint and self.right_max_joint)
        )

    def __del__(self) -> None:
        if not getattr(self, "_inited", False):
            return
        try:
            if rokae_algo is not None:
                rokae_algo.de_init()
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
        self._suppress_cart_vel_once = True

    def _run_one_arm(
        self,
        *,
        rbv: list[float],
        min_j: list[float],
        max_j: list[float],
        q_current: np.ndarray,
        cart_vel: np.ndarray,
        cart_pose_flan: np.ndarray | None,
        elbow: float | None,
        cart_pos_real: np.ndarray,
        psi: float,
        tool_end: np.ndarray,
        tool_ref: np.ndarray,
        base_in_world: np.ndarray,
        joint_num: int,
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, np.ndarray]:
        assert rokae_algo is not None
        rokae_algo.cross_wrist7_init(rbv, min_j, max_j)
        try:
            return run_cross_wrist7_arm_step(
                rokae_algo,
                q_current,
                cart_vel,
                cart_pose_flan,
                elbow,
                cart_pos_real,
                psi,
                tool_end,
                tool_ref,
                base_in_world,
                joint_num,
                self.control_period,
            )
        finally:
            rokae_algo.de_init()

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited or rokae_algo is None:
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

        self._left_cart_pos_flan_in_base, left_q_sol, self._left_elbow, left_cart_cmd = (
            self._run_one_arm(
                rbv=self.left_rbv,
                min_j=self.left_min_joint,
                max_j=self.left_max_joint,
                q_current=left_q,
                cart_vel=left_vel,
                cart_pose_flan=self._left_cart_pos_flan_in_base,
                elbow=self._left_elbow,
                cart_pos_real=left_cart_real,
                psi=left_psi,
                tool_end=self.left_tool_end_pos,
                tool_ref=self.left_tool_ref_pos,
                base_in_world=self.left_base_frame_in_world,
                joint_num=self.left_joint_num,
            )
        )

        self._right_cart_pos_flan_in_base, right_q_sol, self._right_elbow, right_cart_cmd = (
            self._run_one_arm(
                rbv=self.right_rbv,
                min_j=self.right_min_joint,
                max_j=self.right_max_joint,
                q_current=right_q,
                cart_vel=right_vel,
                cart_pose_flan=self._right_cart_pos_flan_in_base,
                elbow=self._right_elbow,
                cart_pos_real=right_cart_real,
                psi=right_psi,
                tool_end=self.right_tool_end_pos,
                tool_ref=self.right_tool_ref_pos,
                base_in_world=self.right_base_frame_in_world,
                joint_num=self.right_joint_num,
            )
        )

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
