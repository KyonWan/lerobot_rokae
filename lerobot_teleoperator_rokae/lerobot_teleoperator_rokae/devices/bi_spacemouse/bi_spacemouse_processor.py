from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from rokae_python_wrapper.rokae_kinematics import rokae_algo
except ImportError:
    rokae_algo = None

from ..spacemouse.spacemouse_processor import (
    _end_effector_to_cart_pose_6,
    _cart_pose_6_to_quat_wfirst,
    update_gripper_state_from_buttons,
)


@ProcessorStepRegistry.register("generate_bi_joint_pos_cmd")
@dataclass
class GenerateBiJointPosCmd(ProcessorStep):
    """
    Processor for bimanual spacemouse: generates joint position commands from cartesian velocities
    by maintaining current joint positions (similar to single arm GenerateJointPosCmd).
    This copies joint positions from observation to action, and converts button presses to gripper toggle states.
    """
    left_joint_num: int = 7
    right_joint_num: int = 7
    initial_left_gripper_state: int = 1  # 初始左夹爪状态（0=close, 1=open），在episode开始前设置
    initial_right_gripper_state: int = 1  # 初始右夹爪状态（0=close, 1=open），在episode开始前设置
    
    def __post_init__(self):
        # Gripper切换状态跟踪：记录上一次按钮状态和当前gripper状态
        self.left_button_prev = 0  # 上一次左按钮状态（0=未按下，1=按下）
        self.right_button_prev = 0  # 上一次右按钮状态（0=未按下，1=按下）
        self.left_gripper_state = self.initial_left_gripper_state  # 当前左gripper状态（0=close, 1=open）
        self.right_gripper_state = self.initial_right_gripper_state  # 当前右gripper状态（0=close, 1=open）
    
    def reset(self, left_gripper_state: int | None = None, right_gripper_state: int | None = None) -> None:
        """
        Reset gripper states to initial values or specified values.
        Called at the start of each episode to synchronize gripper states.
        
        Args:
            left_gripper_state: Left gripper state to set (0=close, 1=open). If None, use initial_left_gripper_state.
            right_gripper_state: Right gripper state to set (0=close, 1=open). If None, use initial_right_gripper_state.
        """
        self.left_button_prev = 0
        self.right_button_prev = 0
        self.left_gripper_state = left_gripper_state if left_gripper_state is not None else self.initial_left_gripper_state
        self.right_gripper_state = right_gripper_state if right_gripper_state is not None else self.initial_right_gripper_state
    
    def __call__(self, transition: Transition) -> Transition:
        """
        Process transition: copy joint positions from observation to action,
        and convert button presses to gripper toggle states.
        """
        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        
        # Extract buttons from action
        left_buttons = action.pop("left_buttons")
        right_buttons = action.pop("right_buttons")
        
        # Copy joint positions from observation to action for left arm
        for i in range(self.left_joint_num):
            obs_key = f"left_joint_pos{i}"
            if obs_key in obs:
                action[f"left_joint_pos{i}"] = obs[obs_key]
        
        # Copy joint positions from observation to action for right arm
        for i in range(self.right_joint_num):
            obs_key = f"right_joint_pos{i}"
            if obs_key in obs:
                action[f"right_joint_pos{i}"] = obs[obs_key]
        
        # Convert button presses to gripper toggle states
        # 检测按钮上升沿（从0变为1）：切换gripper状态
        left_button_curr = 0
        if isinstance(left_buttons, (list, tuple)) and len(left_buttons) > 0:
            left_button_curr = 1 if left_buttons[0] else 0
        
        right_button_curr = 0
        if isinstance(right_buttons, (list, tuple)) and len(right_buttons) > 0:
            right_button_curr = 1 if right_buttons[0] else 0
        
        # 检测左按钮上升沿（从0变为1）：切换gripper状态
        if left_button_curr == 1 and self.left_button_prev == 0:
            # 按钮从未按下变为按下，切换gripper状态
            self.left_gripper_state = 1 - self.left_gripper_state
        
        # 检测右按钮上升沿（从0变为1）：切换gripper状态
        if right_button_curr == 1 and self.right_button_prev == 0:
            # 按钮从未按下变为按下，切换gripper状态
            self.right_gripper_state = 1 - self.right_gripper_state
        
        # 更新上一次按钮状态
        self.left_button_prev = left_button_curr
        self.right_button_prev = right_button_curr
        
        # 使用切换后的gripper状态（0=close, 1=open）
        action["left_gripper_pos"] = float(self.left_gripper_state)
        action["right_gripper_pos"] = float(self.right_gripper_state)
        
        transition[TransitionKey.ACTION] = action
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Ensure joint_pos features exist for both arms
        for side in ["left", "right"]:
            joint_num = self.left_joint_num if side == "left" else self.right_joint_num
            for i in range(joint_num):
                features[PipelineFeatureType.ACTION][f"{side}_joint_pos{i}"] = float
        # Ensure gripper_pos features exist
        features[PipelineFeatureType.ACTION]["left_gripper_pos"] = float
        features[PipelineFeatureType.ACTION]["right_gripper_pos"] = float
        return features


@ProcessorStepRegistry.register("bi_inverse_kinematics_processor")
@dataclass
class BiInverseKinematicsProcessor(ProcessorStep):
    """
    双臂版 InverseKinematicsProcessor：左右臂分别用 rokae_algo cross_wrist7 做笛卡尔速度积分 + 逆解，
    逻辑与单臂一致，key 使用 left_* / right_*。复用 spacemouse_processor 的 helper 与 update_gripper_state_from_buttons。
    """
    left_joint_num: int = 7
    right_joint_num: int = 7
    control_period: float = 1.0 / 30
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
    initial_left_gripper_state: int = 1
    initial_right_gripper_state: int = 1
    # rokae_algo 运动学初始化参数：只支持左右分别配置
    left_rbv: list[float] = field(default_factory=list)
    right_rbv: list[float] = field(default_factory=list)
    left_min_joint: list[float] = field(default_factory=list)
    left_max_joint: list[float] = field(default_factory=list)
    right_min_joint: list[float] = field(default_factory=list)
    right_max_joint: list[float] = field(default_factory=list)

    # 左右臂各自的几何配置（均来源于对应 RokaeRobot，通过 ZMQ server）
    left_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    left_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    right_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    _left_cart_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _right_cart_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _left_elbow: float | None = field(default=None, init=False, repr=False)
    _right_elbow: float | None = field(default=None, init=False, repr=False)
    _inited: bool = field(default=False, init=False, repr=False)
    left_button_prev: int = field(default=0, init=False, repr=False)
    right_button_prev: int = field(default=0, init=False, repr=False)
    left_gripper_state: int = field(default=1, init=False, repr=False)
    right_gripper_state: int = field(default=1, init=False, repr=False)

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
        self._left_cart_pose = None
        self._right_cart_pose = None
        self._left_elbow = None
        self._right_elbow = None
        self.left_button_prev = 0
        self.right_button_prev = 0
        self.left_gripper_state = left_gripper_state if left_gripper_state is not None else self.initial_left_gripper_state
        self.right_gripper_state = right_gripper_state if right_gripper_state is not None else self.initial_right_gripper_state

    def _effective_left_params(self) -> tuple[list[float], list[float], list[float]]:
        return (self.left_rbv, self.left_min_joint, self.left_max_joint)

    def _effective_right_params(self) -> tuple[list[float], list[float], list[float]]:
        return (self.right_rbv, self.right_min_joint, self.right_max_joint)

    def _run_arm(
        self,
        q_current: np.ndarray,
        cart_vel: np.ndarray,
        cart_pose: np.ndarray | None,
        elbow: float | None,
        tool_end_pos: np.ndarray,
        tool_ref_pos: np.ndarray,
        base_frame_in_world: np.ndarray,
        joint_num: int,
    ) -> tuple[np.ndarray | None, np.ndarray, float | None]:
        """
        单臂一步：SpaceMouse 的 twist 为 end@ref，在 ref 坐标系下；
        将其转换为 flan@base，在 base 坐标系下积分，然后用 rokae_algo 做逆解。
        返回 (新 cart_pose, q_solution, 新 elbow)。
        """
        # 初始化当前笛卡尔位姿（flan 相对于 base）
        if cart_pose is None:
            end_effector, psi, ec = rokae_algo.cross_wrist7_jnt2cart(q_current.tolist())
            if ec == 0 and len(end_effector) >= 6:
                cart_pose = _end_effector_to_cart_pose_6(end_effector)
                elbow = float(psi)
            else:
                cart_pose = np.zeros(6, dtype=np.float64)
                elbow = 0.0

        # 限幅（与单臂一致）
        cart_vel = np.array(cart_vel, dtype=np.float64)
        cart_vel[:3] *= float(self.trans_max_vel)
        cart_vel[3:] *= float(self.rot_max_vel)

        # 齐次变换辅助函数
        def make_homogeneous(R_mat: np.ndarray, trans_vec: np.ndarray) -> np.ndarray:
            H = np.eye(4, dtype=np.float64)
            H[:3, :3] = R_mat
            H[:3, 3] = trans_vec
            return H

        def inv_homogeneous(T: np.ndarray) -> np.ndarray:
            R_part = T[:3, :3]
            t_part = T[:3, 3]
            T_inv = np.eye(4, dtype=np.float64)
            R_inv = R_part.T
            t_inv = -R_inv @ t_part
            T_inv[:3, :3] = R_inv
            T_inv[:3, 3] = t_inv
            return T_inv

        # base / ref / flan / end 之间的变换
        R_world_base = R.from_euler("xyz", base_frame_in_world[3:], degrees=False).as_matrix()
        T_world_base = make_homogeneous(R_world_base, np.array(base_frame_in_world[:3], dtype=np.float64))

        R_world_ref = R.from_euler("xyz", tool_ref_pos[3:], degrees=False).as_matrix()
        T_world_ref = make_homogeneous(R_world_ref, np.array(tool_ref_pos[:3], dtype=np.float64))

        T_base_ref = inv_homogeneous(T_world_base) @ T_world_ref
        T_ref_base = inv_homogeneous(T_base_ref)

        R_flan_end = R.from_euler("xyz", tool_end_pos[3:], degrees=False).as_matrix()
        T_flan_end = make_homogeneous(R_flan_end, np.array(tool_end_pos[:3], dtype=np.float64))
        T_end_flan = inv_homogeneous(T_flan_end)

        R_base_flan = R.from_euler("xyz", cart_pose[3:], degrees=False).as_matrix()
        T_base_flan = make_homogeneous(R_base_flan, np.array(cart_pose[:3], dtype=np.float64))

        # ref 系下的 end 位姿
        T_ref_end = T_ref_base @ T_base_flan @ T_flan_end

        # 在 ref 系下按 SpaceMouse twist 做一次小步积分
        T_ref_end_new = T_ref_end.copy()
        T_ref_end_new[:3, 3] += cart_vel[:3] * float(self.control_period)
        R_rot = R.from_rotvec(cart_vel[3:] * float(self.control_period)).as_matrix()
        T_ref_end_new[:3, :3] = R_rot @ T_ref_end_new[:3, :3]

        # 回到 base 系下的 flange 位姿
        T_base_flan = T_base_ref @ T_ref_end_new @ T_end_flan

        cart_pose[:3] = T_base_flan[:3, 3]
        R_new_base_flan = R.from_matrix(T_base_flan[:3, :3])
        cart_pose[3:] = R_new_base_flan.as_euler("xyz", degrees=False)

        # 逆解
        end_effector_list = _cart_pose_6_to_quat_wfirst(cart_pose)
        psi = float(elbow) if elbow is not None else 0.0
        q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(q_current.tolist(), end_effector_list, psi)
        if ec != 0 or len(q_solution) < joint_num:
            q_solution = q_current.tolist()

        return cart_pose, np.array(q_solution[:joint_num], dtype=np.float64), elbow

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

        left_q = np.array([obs[f"left_joint_pos{i}"] for i in range(self.left_joint_num)], dtype=np.float64)
        right_q = np.array([obs[f"right_joint_pos{i}"] for i in range(self.right_joint_num)], dtype=np.float64)
        left_cart_vel = np.array([action.get(f"left_cart_vel{i}", 0.0) for i in range(6)], dtype=np.float64)
        right_cart_vel = np.array([action.get(f"right_cart_vel{i}", 0.0) for i in range(6)], dtype=np.float64)

        left_rbv, left_min, left_max = self._effective_left_params()
        rokae_algo.cross_wrist7_init(left_rbv, left_min, left_max)
        self._left_cart_pose, left_q_sol, self._left_elbow = self._run_arm(
            q_current=left_q,
            cart_vel=left_cart_vel,
            cart_pose=self._left_cart_pose,
            elbow=self._left_elbow,
            tool_end_pos=self.left_tool_end_pos,
            tool_ref_pos=self.left_tool_ref_pos,
            base_frame_in_world=self.left_base_frame_in_world,
            joint_num=self.left_joint_num,
        )
        rokae_algo.de_init()

        right_rbv, right_min, right_max = self._effective_right_params()
        rokae_algo.cross_wrist7_init(right_rbv, right_min, right_max)
        self._right_cart_pose, right_q_sol, self._right_elbow = self._run_arm(
            q_current=right_q,
            cart_vel=right_cart_vel,
            cart_pose=self._right_cart_pose,
            elbow=self._right_elbow,
            tool_end_pos=self.right_tool_end_pos,
            tool_ref_pos=self.right_tool_ref_pos,
            base_frame_in_world=self.right_base_frame_in_world,
            joint_num=self.right_joint_num,
        )
        rokae_algo.de_init()

        for i in range(self.left_joint_num):
            action[f"left_joint_pos{i}"] = float(left_q_sol[i])
        for i in range(self.right_joint_num):
            action[f"right_joint_pos{i}"] = float(right_q_sol[i])
        action["left_gripper_pos"] = float(self.left_gripper_state)
        action["right_gripper_pos"] = float(self.right_gripper_state)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for i in range(self.left_joint_num):
            features[PipelineFeatureType.ACTION][f"left_joint_pos{i}"] = float
        for i in range(self.right_joint_num):
            features[PipelineFeatureType.ACTION][f"right_joint_pos{i}"] = float
        features[PipelineFeatureType.ACTION]["left_gripper_pos"] = float
        features[PipelineFeatureType.ACTION]["right_gripper_pos"] = float
        return features
