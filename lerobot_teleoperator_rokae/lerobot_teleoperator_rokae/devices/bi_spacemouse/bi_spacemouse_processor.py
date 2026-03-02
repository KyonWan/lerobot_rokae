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

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    TransformCache,
    inv_homogeneous,
    pose_to_transform,
    transform_to_pose,
)


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
    _left_transform_cache: TransformCache = field(default_factory=TransformCache, init=False, repr=False)
    _right_transform_cache: TransformCache = field(default_factory=TransformCache, init=False, repr=False)
    _left_cached_tool_end_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _left_cached_T_flan_end: np.ndarray | None = field(default=None, init=False, repr=False)
    _right_cached_tool_end_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _right_cached_T_flan_end: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.left_gripper_state = self.initial_left_gripper_state
        self.right_gripper_state = self.initial_right_gripper_state
        # 初始化缓存对象
        self._left_transform_cache = TransformCache()
        self._right_transform_cache = TransformCache()
        self._left_cached_tool_end_pos = None
        self._left_cached_T_flan_end = None
        self._right_cached_tool_end_pos = None
        self._right_cached_T_flan_end = None
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
        # 清除缓存
        self._left_transform_cache.clear_cache()
        self._right_transform_cache.clear_cache()
        self._left_cached_tool_end_pos = None
        self._left_cached_T_flan_end = None
        self._right_cached_tool_end_pos = None
        self._right_cached_T_flan_end = None

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
        cart_pos_end_in_base: np.ndarray,
        psi: float,
        tool_end_pos: np.ndarray,
        tool_ref_pos: np.ndarray,
        base_frame_in_world: np.ndarray,
        joint_num: int,
        transform_cache: TransformCache,
        cached_tool_end_pos: np.ndarray | None,
        cached_T_flan_end: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        单臂一步：SpaceMouse 的 twist 为 end@ref，在 ref 坐标系下；
        将其转换为 flan@base，在 base 坐标系下积分，然后用 rokae_algo 做逆解。
        返回 (新 cart_pose, q_solution, 新 elbow, cart_pos_ref_end, 新 cached_tool_end_pos, 新 cached_T_flan_end)。
        """
        # 检查必需参数
        if cart_pos_end_in_base is None:
            raise ValueError("cart_pos_end_in_base must be provided")
        # psi 总是有值（非7轴时为 0.0），无需检查 None

        # 限幅（与单臂一致）
        cart_vel = np.array(cart_vel, dtype=np.float64)
        cart_vel[:3] *= float(self.trans_max_vel)
        cart_vel[3:] *= float(self.rot_max_vel)

        # 缓存 T_flan_end（tool_end_pos 通常不变）
        if (
            cached_T_flan_end is None
            or cached_tool_end_pos is None
            or not np.allclose(cached_tool_end_pos, tool_end_pos, rtol=1e-10, atol=1e-10)
        ):
            T_end_flan = pose_to_transform(tool_end_pos)  # end 相对于 flange
            cached_T_flan_end = inv_homogeneous(T_end_flan)  # flange 相对于 end
            cached_tool_end_pos = tool_end_pos.copy()
        T_flan_end = cached_T_flan_end
        
        # 初始化当前笛卡尔位姿
        if cart_pose is None:
            # cart_pos_end_in_base 是 end 相对于 base
            T_base_end = pose_to_transform(cart_pos_end_in_base)
            
            # 转换为 flange 相对于 base: T_base_flan = T_base_end @ T_end_flan
            T_end_flan = inv_homogeneous(T_flan_end)
            T_base_flan = T_base_end @ T_end_flan
            
            cart_pose = transform_to_pose(T_base_flan)
            
            # 对于7轴，使用从观测中获取的psi
            if joint_num == 7:
                elbow = float(psi)

        # 使用缓存获取 T_base_ref 和 T_ref_base
        T_base_ref, _ = transform_cache.get_base_ref_transform(
            tool_ref_pos, base_frame_in_world
        )
        T_ref_base = inv_homogeneous(T_base_ref)

        # T_base_flan 从当前 cart_pose 计算（会变化，不缓存）
        T_base_flan = pose_to_transform(cart_pose)

        # T_ref_end = T_ref_base @ T_base_flan @ T_flan_end
        T_ref_end = T_ref_base @ T_base_flan @ T_flan_end

        T_ref_end_new = T_ref_end.copy()
        T_ref_end_new[:3, 3] += cart_vel[:3] * self.control_period
        R_rot = R.from_rotvec(cart_vel[3:] * self.control_period).as_matrix()
        T_ref_end_new[:3, :3] = R_rot @ T_ref_end_new[:3, :3]

        # 提取 end 相对于 ref 的位置（用于 cart_pos mode）
        cart_pos_ref_end = transform_to_pose(T_ref_end_new)

        T_end_flan = inv_homogeneous(T_flan_end)
        T_base_flan = T_base_ref @ T_ref_end_new @ T_end_flan

        # 更新当前位置 [x, y, z, rx, ry, rz]（位置 + 姿态）
        cart_pose = transform_to_pose(T_base_flan)

        # 逆解
        end_effector_list = _cart_pose_6_to_quat_wfirst(cart_pose)
        psi = float(elbow) if elbow is not None else 0.0
        q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(q_current.tolist(), end_effector_list, psi)
        if ec != 0 or len(q_solution) < joint_num:
            q_solution = q_current.tolist()

        return cart_pose, np.array(q_solution[:joint_num], dtype=np.float64), elbow, cart_pos_ref_end, cached_tool_end_pos, cached_T_flan_end

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
        
        # 获取 cart_pos_end_in_base（end相对于base）
        left_cart_pos_end_in_base = np.array([obs.get(f"left_cart_pos{i}", 0.0) for i in range(6)], dtype=np.float64)
        right_cart_pos_end_in_base = np.array([obs.get(f"right_cart_pos{i}", 0.0) for i in range(6)], dtype=np.float64)
        
        # 统一处理 psi：非7轴时为 0.0
        left_psi = float(obs.get("left_psi", 0.0))
        right_psi = float(obs.get("right_psi", 0.0))
        
        left_cart_vel = np.array([action.get(f"left_cart_vel{i}", 0.0) for i in range(6)], dtype=np.float64)
        right_cart_vel = np.array([action.get(f"right_cart_vel{i}", 0.0) for i in range(6)], dtype=np.float64)

        left_rbv, left_min, left_max = self._effective_left_params()
        rokae_algo.cross_wrist7_init(left_rbv, left_min, left_max)
        self._left_cart_pose, left_q_sol, self._left_elbow, left_cart_pos_ref_end, self._left_cached_tool_end_pos, self._left_cached_T_flan_end = self._run_arm(
            q_current=left_q,
            cart_vel=left_cart_vel,
            cart_pose=self._left_cart_pose,
            elbow=self._left_elbow,
            cart_pos_end_in_base=left_cart_pos_end_in_base,
            psi=left_psi,
            tool_end_pos=self.left_tool_end_pos,
            tool_ref_pos=self.left_tool_ref_pos,
            base_frame_in_world=self.left_base_frame_in_world,
            joint_num=self.left_joint_num,
            transform_cache=self._left_transform_cache,
            cached_tool_end_pos=self._left_cached_tool_end_pos,
            cached_T_flan_end=self._left_cached_T_flan_end,
        )
        rokae_algo.de_init()

        right_rbv, right_min, right_max = self._effective_right_params()
        rokae_algo.cross_wrist7_init(right_rbv, right_min, right_max)
        self._right_cart_pose, right_q_sol, self._right_elbow, right_cart_pos_ref_end, self._right_cached_tool_end_pos, self._right_cached_T_flan_end = self._run_arm(
            q_current=right_q,
            cart_vel=right_cart_vel,
            cart_pose=self._right_cart_pose,
            elbow=self._right_elbow,
            cart_pos_end_in_base=right_cart_pos_end_in_base,
            psi=right_psi,
            tool_end_pos=self.right_tool_end_pos,
            tool_ref_pos=self.right_tool_ref_pos,
            base_frame_in_world=self.right_base_frame_in_world,
            joint_num=self.right_joint_num,
            transform_cache=self._right_transform_cache,
            cached_tool_end_pos=self._right_cached_tool_end_pos,
            cached_T_flan_end=self._right_cached_T_flan_end,
        )
        rokae_algo.de_init()

        # 总是输出 cart_pos、joint_pos 和 cart_vel（用于记录到数据集）
        # robot processor 会根据 callback_mode 选择发送哪个字段给机器人（在 robot_action_processor 中处理）
        for i in range(6):
            action[f"left_cart_pos{i}"] = float(left_cart_pos_ref_end[i])
            action[f"right_cart_pos{i}"] = float(right_cart_pos_ref_end[i])
        for i in range(self.left_joint_num):
            action[f"left_joint_pos{i}"] = float(left_q_sol[i])
        for i in range(self.right_joint_num):
            action[f"right_joint_pos{i}"] = float(right_q_sol[i])
        
        # 输出 cart_vel（end 相对于 ref，已经乘以速度限制）
        for i in range(6):
            action[f"left_cart_vel{i}"] = float(left_cart_vel[i])
            action[f"right_cart_vel{i}"] = float(right_cart_vel[i])
        
        # 总是输出 psi（用于记录到数据集），非7轴时为 0.0
        action["left_psi"] = float(left_psi)
        action["right_psi"] = float(right_psi)
        
        action["left_gripper_pos"] = float(self.left_gripper_state)
        action["right_gripper_pos"] = float(self.right_gripper_state)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 总是添加 cart_pos、joint_pos 和 cart_vel 特征（用于记录到数据集）
        action_features = features[PipelineFeatureType.ACTION]
        
        # joint_pos 特征
        for i in range(self.left_joint_num):
            action_features.setdefault(f"left_joint_pos{i}", float)
        for i in range(self.right_joint_num):
            action_features.setdefault(f"right_joint_pos{i}", float)
        
        # cart_pos 特征
        for i in range(6):
            action_features.setdefault(f"left_cart_pos{i}", float)
            action_features.setdefault(f"right_cart_pos{i}", float)
        
        # cart_vel 特征（end 相对于 ref）
        for i in range(6):
            action_features.setdefault(f"left_cart_vel{i}", float)
            action_features.setdefault(f"right_cart_vel{i}", float)
        
        # psi 特征（总是添加，非7轴时为 0.0）
        action_features.setdefault("left_psi", float)
        action_features.setdefault("right_psi", float)
        
        # gripper_pos 特征
        action_features.setdefault("left_gripper_pos", float)
        action_features.setdefault("right_gripper_pos", float)
        
        return features
