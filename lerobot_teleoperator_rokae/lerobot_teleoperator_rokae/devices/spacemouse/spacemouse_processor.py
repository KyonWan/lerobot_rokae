"""SpaceMouse 笛卡尔速度 → 关节指令；6 轴 rokae_algo，7 轴 Pink IK（rokae_ik_interface）。"""

from lerobot.processor.pipeline import ProcessorStepRegistry, ProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rokae_python_wrapper.rokae_kinematics import rokae_algo
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
    default_rokae_urdf_path,
    integrate_twist_then_solve_ik,
    reset_pink_if_flagged,
)


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
    urdf_path: str = field(default_factory=default_rokae_urdf_path)
    # 须与 URDF 中末端 link 名一致；默认对齐 pink_ik_helpers 内置 AR5-5_07L-W4C4A2 包
    end_effector_frame: str = "AR5-5_07L-W4C4A2_tcp"
    joint_vel_max: float = SNS_JOINT_VEL_MAX
    joint_acc_max: float = SNS_JOINT_ACC_MAX

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
        configure_pink_7axis(
            self._ik_state,
            self.min_joint,
            self.max_joint,
            self.joint_num,
            self.joint_vel_max,
            self.joint_acc_max,
        )
        ok = self._ik_state is not None
        if ok:
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
        reset_consumed = reset_pink_if_flagged(
            self._ik_state, q_current, self.reset_ik_state
        )
        if reset_consumed:
            self.reset_ik_state = False

        cart_pos_real = obs_cart_pos_end_in_base(obs, "")
        psi = float(obs.get("psi", 0.0))
        cart_vel = scale_action_cart_vel(
            action, self.trans_max_vel, self.rot_max_vel, ""
        )
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
        q_solution, cart_pos_cmd_end_in_ref, new_flange_pose, elbow_out = (
            integrate_twist_then_solve_ik(
                self.joint_num,
                cart_vel,
                q_current,
                cart_pos_real_end_in_base,
                psi,
                self.tool_end_pos,
                self.tool_ref_pos,
                self.base_frame_in_world,
                self._cart_pos_flan_in_base,
                self.control_period,
                self.trans_max_vel,
                self._ik_state,
                log_prefix="[InverseKinematicsProcessor]",
            )
        )
        if elbow_out is not None:
            self._elbow = elbow_out
        if new_flange_pose is not None:
            self._cart_pos_flan_in_base = new_flange_pose
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
