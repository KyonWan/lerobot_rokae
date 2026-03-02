from lerobot.processor.pipeline import RobotActionProcessorStep, ProcessorStepRegistry, ProcessorStep
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from math import pi as M_PI
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from rokae_python_wrapper.rokae_kinematics import rokae_algo
except ImportError:
    rokae_algo = None

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    TransformCache,
    inv_homogeneous,
    pose_to_transform,
    transform_to_pose,
)


def _end_effector_to_cart_pose_6(end_effector: list[float]) -> np.ndarray:
    """将正解返回值转为内部 [x,y,z,rx,ry,rz]（欧拉 xyz）。
    支持 end_effector 为 7 维 [x,y,z,qw,qx,qy,qz]（四元数 w first）或 6 维 [x,y,z,rx,ry,rz]。"""
    arr = np.array(end_effector, dtype=np.float64)
    if len(arr) >= 7:
        # [x, y, z, qw, qx, qy, qz] -> [x, y, z, rx, ry, rz]
        out = np.zeros(6, dtype=np.float64)
        out[:3] = arr[:3]
        # scipy Rotation.from_quat 约定为 (x, y, z, w)
        r = R.from_quat([arr[4], arr[5], arr[6], arr[3]])
        out[3:] = r.as_euler("xyz", degrees=False)
        return out
    elif len(arr) >= 6:
        return np.array(arr[:6], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def _cart_pose_6_to_quat_wfirst(cart_pose: np.ndarray) -> list[float]:
    """将内部 [x,y,z,rx,ry,rz] 转为逆解接口需要的 [x,y,z,qw,qx,qy,qz]（四元数 w first）。"""
    r = R.from_euler("xyz", cart_pose[3:], degrees=False)
    q = r.as_quat()  # scipy 返回 [x, y, z, w]
    return list(cart_pose[:3]) + [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def update_gripper_state_from_buttons(
    buttons: list | tuple,
    button_prev: int,
    gripper_state: int,
) -> tuple[int, int]:
    """
    根据按钮状态更新夹爪开关状态。

    逻辑：
    - 仅在按钮从 0 -> 1 的上升沿时切换一次夹爪状态
    - 返回新的 (button_prev, gripper_state)
    """
    button_curr = 0
    if isinstance(buttons, (list, tuple)) and len(buttons) > 0:
        button_curr = 1 if buttons[0] else 0

    # 检测按钮上升沿（从0变为1）：切换gripper状态
    if button_curr == 1 and button_prev == 0:
        gripper_state = 1 - gripper_state

    return button_curr, gripper_state


@ProcessorStepRegistry.register("inverse_kinematics_processor")
@dataclass
class InverseKinematicsProcessor(ProcessorStep):
    """
    使用 rokae_algo 做笛卡尔速度 -> 关节位置的逆解（正逆解均走 rokae_kinematics 的 C 扩展）。

    - teleop 侧从 SpaceMouse 读取 cart_vel0..5，本 Processor 积分得到目标笛卡尔位姿 [x,y,z,rx,ry,rz]，
      再调用 rokae_algo 的 cr6_* / cross_wrist7_* 做正逆解。
    - 6 轴：cr_init(rbv, min_joint, max_joint)，正解 cr6_jnt2cart，逆解 cr6_cart2jnt_nearest。
    - 7 轴：cross_wrist7_init(rbv, min_joint, max_joint)，正解 cross_wrist7_jnt2cart，逆解 cross_wrist7_cart2jnt_nearest。
    - rbv、min_joint、max_joint 在 rokae 配置中按 list 配置。
    """
    joint_num: int = 6
    control_period: float = 1.0 / 30  # 控制周期，秒
    trans_max_vel: float = 0.1  # m/s
    rot_max_vel: float = 0.2    # rad/s
    initial_gripper_state: int = 1  # 初始夹爪状态（0=close, 1=open），在 episode 开始前设置
    rbv: list[float] = field(default_factory=list)           # 机器人描述参数（RD），用于 cr_init / cross_wrist7_init
    min_joint: list[float] = field(default_factory=list)     # 关节下限（弧度）
    max_joint: list[float] = field(default_factory=list)    # 关节上限（弧度）

    # 由 rokae_robot 在创建时传入的底层配置（均来源于 ZMQ server），用于保证 IK 与实际机器人一致。
    tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # end 相对于 flange，在 flange 坐标系表述
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # ref 相对于 world 的位姿
    base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # base 相对于 world [x,y,z,rx,ry,rz]

    _cart_pose: np.ndarray | None = field(default=None, init=False, repr=False)  # [x, y, z, rx, ry, rz]
    _elbow: float = field(default=0.0, init=False, repr=False)  # 臂角 psi，7轴时为实际值，非7轴时为 0.0
    _inited: bool = field(default=False, init=False, repr=False)                  # rokae_algo 是否已初始化
    button_prev: int = field(default=0, init=False, repr=False)                   # 上一帧按钮状态，用于夹爪上升沿切换
    gripper_state: int = field(default=1, init=False, repr=False)                 # 当前夹爪状态（0=close, 1=open）

    def __post_init__(self) -> None:
        self.gripper_state = self.initial_gripper_state
        # 初始化缓存对象
        self._transform_cache = TransformCache()
        self._cached_tool_end_pos = None
        self._cached_T_flan_end = None
        
        if rokae_algo is None or not self.rbv or not self.min_joint or not self.max_joint:
            self._inited = False
            return
        try:
            if self.joint_num == 6:
                self._inited = rokae_algo.cr_init(self.rbv, self.min_joint, self.max_joint)
            elif self.joint_num == 7:
                self._inited = rokae_algo.cross_wrist7_init(self.rbv, self.min_joint, self.max_joint)
            else:
                self._inited = False
        except Exception:
            self._inited = False

    def __del__(self) -> None:
        """对象被回收时调用 rokae_algo.de_init()，释放 C 侧资源。"""
        if not getattr(self, "_inited", False):
            return
        try:
            if rokae_algo is not None:
                rokae_algo.de_init()
        except Exception:
            pass  # __del__ 中不宜再抛异常

    def reset(self, gripper_state: int | None = None) -> None:
        self._cart_pose = None
        self._elbow = 0.0  # 重置为 0.0
        self.button_prev = 0
        self.gripper_state = gripper_state if gripper_state is not None else self.initial_gripper_state
        self._transform_cache.clear_cache()
        self._cached_tool_end_pos = None
        self._cached_T_flan_end = None

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited or rokae_algo is None:
            return transition

        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        # SpaceMouse 按钮 -> 夹爪切换
        buttons = action.pop("buttons", [0, 0])
        self.button_prev, self.gripper_state = update_gripper_state_from_buttons(
            buttons=buttons,
            button_prev=self.button_prev,
            gripper_state=self.gripper_state,
        )

        q_current = np.array(
            [obs[f"joint_pos{i}"] for i in range(self.joint_num)],
            dtype=np.float64,
        )

        cart_pos_end_in_base = np.array([obs[f"cart_pos{i}"] for i in range(6)], dtype=np.float64)
        # 统一处理 psi：非7轴时为 0.0
        psi = float(obs.get("psi", 0.0))

        cart_vel = np.array(
            [action.get(f"cart_vel{i}", 0.0) for i in range(6)],
            dtype=np.float64,
        )
        cart_vel[:3] *= float(self.trans_max_vel)
        cart_vel[3:] *= float(self.rot_max_vel)

        q_solution, cart_pos_ref_end = self.process_coordinate_transform_and_ik(cart_vel, q_current, cart_pos_end_in_base, psi)
        
        # 总是输出 cart_pos、joint_pos 和 cart_vel（用于记录到数据集）
        # robot processor 会根据 callback_mode 选择发送哪个字段给机器人（在 robot_action_processor 中处理）
        for i in range(6):
            action[f"cart_pos{i}"] = float(cart_pos_ref_end[i])
        for i in range(self.joint_num):
            action[f"joint_pos{i}"] = float(q_solution[i])
        
        # 输出 cart_vel（end 相对于 ref，已经乘以速度限制）
        for i in range(6):
            action[f"cart_vel{i}"] = float(cart_vel[i])
        
        # 总是输出 psi（用于记录到数据集），非7轴时为 0.0
        action["psi"] = float(psi)

        action["gripper_pos"] = float(self.gripper_state)

        transition[TransitionKey.ACTION] = action
        return transition
    
    def process_coordinate_transform_and_ik(self, cart_vel, q_current, cart_pos_end_in_base, psi=0.0):
        """
        处理坐标系变换和逆解。
        
        Args:
            cart_vel: 笛卡尔速度 [vx, vy, vz, wx, wy, wz]
            q_current: 当前关节位置
            cart_pos_end_in_base: 笛卡尔位置（end相对于base），必须提供
            psi: psi值，7轴时为实际臂角，非7轴时为 0.0
        
        Returns:
            (q_solution, cart_pos_ref_end): 关节解和 end 相对于 ref 的位置
        """
        # 检查必需参数
        if cart_pos_end_in_base is None:
            raise ValueError("cart_pos_end_in_base must be provided")
        # psi 总是有值（非7轴时为 0.0），无需检查 None
        
        # 缓存 T_flan_end（tool_end_pos 通常不变）
        # tool_end_pos 是 end 相对于 flange，所以 T_flan_end = pose_to_transform(tool_end_pos) 是 end 相对于 flange
        # 但我们需要的是 flange 相对于 end，所以需要取逆
        if (
            self._cached_T_flan_end is None
            or self._cached_tool_end_pos is None
            or not np.allclose(self._cached_tool_end_pos, self.tool_end_pos, rtol=1e-10, atol=1e-10)
        ):
            T_end_flan = pose_to_transform(self.tool_end_pos)  # end 相对于 flange
            self._cached_T_flan_end = inv_homogeneous(T_end_flan)  # flange 相对于 end
            self._cached_tool_end_pos = self.tool_end_pos.copy()
        T_flan_end = self._cached_T_flan_end
        
        # 初始化当前笛卡尔位姿
        if self._cart_pose is None:
            # cart_pos_end_in_base 是 end 相对于 base
            T_base_end = pose_to_transform(cart_pos_end_in_base)
            
            # 转换为 flange 相对于 base: T_base_flan = T_base_end @ T_end_flan
            # T_end_flan = inv_homogeneous(T_flan_end)
            T_end_flan = inv_homogeneous(T_flan_end)
            T_base_flan = T_base_end @ T_end_flan
            
            self._cart_pose = transform_to_pose(T_base_flan)
            
            # 对于7轴，使用从观测中获取的psi
            if self.joint_num == 7:
                self._elbow = float(psi)

        # 使用缓存获取 T_base_ref 和 T_ref_base
        T_base_ref, _ = self._transform_cache.get_base_ref_transform(
            self.tool_ref_pos, self.base_frame_in_world
        )
        T_ref_base = inv_homogeneous(T_base_ref)

        # T_base_flan 从当前 _cart_pose 计算（会变化，不缓存）
        T_base_flan = pose_to_transform(self._cart_pose)

        # T_ref_end = T_ref_base @ T_base_flan @ T_flan_end
        T_ref_end = T_ref_base @ T_base_flan @ T_flan_end

        T_ref_end_new = T_ref_end.copy()
        T_ref_end_new[:3, 3] += cart_vel[:3] * self.control_period
        R_rot = R.from_rotvec(cart_vel[3:] * self.control_period).as_matrix()
        T_ref_end_new[:3, :3] = R_rot @ T_ref_end_new[:3, :3]

        # 提取 end 相对于 ref 的位置（用于 cart_pos mode）
        cart_pos_ref_end = np.zeros(6, dtype=np.float64)
        cart_pos_ref_end[:3] = T_ref_end_new[:3, 3]
        R_ref_end = R.from_matrix(T_ref_end_new[:3, :3])
        cart_pos_ref_end[3:] = R_ref_end.as_euler("xyz", degrees=False)

        T_end_flan = inv_homogeneous(T_flan_end)
        T_base_flan = T_base_ref @ T_ref_end_new @ T_end_flan

        # 更新当前位置 [x, y, z, rx, ry, rz]（位置 + 姿态）
        self._cart_pose[:3] = T_base_flan[:3, 3]
        R_new_base_flan = R.from_matrix(T_base_flan[:3, :3])
        self._cart_pose[3:] = R_new_base_flan.as_euler("xyz", degrees=False)

        # 逆解
        end_effector_list = _cart_pose_6_to_quat_wfirst(self._cart_pose)
        q_init = q_current.tolist()
        ec = 0
        if self.joint_num == 6:
            q_solution, ec = rokae_algo.cr6_cart2jnt(q_init, end_effector_list)
        elif self.joint_num == 7:
            q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(q_init, end_effector_list, self._elbow)
        if ec != 0:
            q_solution = q_current.tolist()
        return q_solution, cart_pos_ref_end


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 总是添加 cart_pos、joint_pos 和 cart_vel 特征（用于记录到数据集）
        action_features = features[PipelineFeatureType.ACTION]
        
        # joint_pos 特征
        for i in range(self.joint_num):
            action_features.setdefault(f"joint_pos{i}", float)
        
        # cart_pos 特征
        for i in range(6):
            action_features.setdefault(f"cart_pos{i}", float)
        
        # cart_vel 特征（end 相对于 ref）
        for i in range(6):
            action_features.setdefault(f"cart_vel{i}", float)
        
        # psi 特征（总是添加，非7轴时为 0.0）
        action_features.setdefault("psi", float)
        
        # gripper_pos 特征
        action_features.setdefault("gripper_pos", float)
        
        return features