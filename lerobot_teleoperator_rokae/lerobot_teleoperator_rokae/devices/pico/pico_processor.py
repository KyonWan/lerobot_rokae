from time import sleep
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose

try:
    from rokae_python_wrapper.rokae_kinematics import rokae_algo
except ImportError:
    rokae_algo = None

from ..cartesian_ik_helpers import transform_to_quat_wfirst as _transform_to_quat_wfirst

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    compute_base_ref_transform,
    inv_homogeneous,
    pose_to_transform,
    transform_to_pose,
)


@ProcessorStepRegistry.register("pico_bi_inverse_kinematics_processor")
@dataclass
class PicoBiInverseKinematicsProcessor(ProcessorStep):
    """
    pico的双臂版 InverseKinematicsProcessor：左右臂分别用 rokae_algo cross_wrist7 做逆解，
    逻辑与单臂一致，key 使用 left_* / right_*。
    """
    left_joint_num: int = 7
    right_joint_num: int = 7
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
    left_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64)) # end 相对于 flange
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64)) # ref 相对于 world
    left_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    right_tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    right_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    _left_elbow: float | None = field(default=None, init=False, repr=False) 
    _right_elbow: float | None = field(default=None, init=False, repr=False)
    _inited: bool = field(default=False, init=False, repr=False) # processor 是否具备运行条件（rokae_algo 初始化成功，左右臂参数有效）

    def __post_init__(self) -> None:
        self.left_gripper_state = self.initial_left_gripper_state
        self.right_gripper_state = self.initial_right_gripper_state
        self._inited = (
            rokae_algo is not None
            and bool(self.left_rbv and self.left_min_joint and self.left_max_joint)
            and bool(self.right_rbv and self.right_min_joint and self.right_max_joint)
        )
        self.reset()

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
        self._left_elbow = None
        self._right_elbow = None
        self.left_gripper_state = left_gripper_state if left_gripper_state is not None else self.initial_left_gripper_state
        self.right_gripper_state = right_gripper_state if right_gripper_state is not None else self.initial_right_gripper_state
        self.left_start_pos_end_in_world = None  # 左臂用于逆解的初始笛卡尔位置（end相对于world），形式为[x, y, z]
        self.right_start_pos_end_in_world = None # 左臂用于逆解的初始笛卡尔姿态（end相对于world），形式为[w, x, y, z]
        self.left_start_ori_end_in_world = None
        self.right_start_ori_end_in_world = None

    def _effective_left_params(self) -> tuple[list[float], list[float], list[float]]:
        return (self.left_rbv, self.left_min_joint, self.left_max_joint)

    def _effective_right_params(self) -> tuple[list[float], list[float], list[float]]:
        return (self.right_rbv, self.right_min_joint, self.right_max_joint)

    def _run_arm(
        self,
        side: Literal["left", "right"],
        q_current: np.ndarray,
        cart_delta_end_in_world: np.ndarray,
        elbow: float | None,
        cart_pos_real_end_in_base: np.ndarray,
        psi: float,
        tool_end_pos: np.ndarray,
        tool_ref_pos: np.ndarray,
        base_frame_in_world: np.ndarray,
        joint_num: int,
    ) -> tuple[np.ndarray | None, np.ndarray, float | None, np.ndarray]:
        """
        用于将 pico 返回的世界坐标系下的笛卡尔增量转化为关节空间的角度和在ref坐标系下的笛卡尔位姿。
        流程：
            pico cart_delta(world frame) -> cart_pos(ref frame) -> cart_pose(base frame) -> inverse kinematics -> joint_pos
        
        Args:
            side: 手臂名称，"left" 或 "right"
            q_current: 当前关节角度
            cart_delta_end_in_world: pico 返回的末端相对于世界坐标系的笛卡尔增量，形式为[x, y, z, wx, wy, wz]，旋转使用轴角表示
            elbow: 当前肘部角度
            cart_pos_real_end_in_base: 当前末端相对于基的笛卡尔位姿
            psi: 当前臂角
            tool_end_pos: end 相对于 flange 的位姿
            tool_ref_pos: ref 相对于 world 的位姿
            base_frame_in_world: base 相对于 world 的位姿
            joint_num: 关节数量
        
        Returns:
            (新 q_solution, 新 elbow, cart_pos_cmd_end_in_ref)。
        """

        T_end_in_flan = pose_to_transform(tool_end_pos)  # end 相对于 flange
        T_flan_in_end = inv_homogeneous(T_end_in_flan)  # flange 相对于 end

        # 计算 T_ref_in_base（ref 相对于 base）
        T_ref_in_base, _ = compute_base_ref_transform(
            tool_ref_pos, base_frame_in_world
        )
        T_base_in_ref = inv_homogeneous(T_ref_in_base)  # base 相对于 ref

        T_base_in_world = pose_to_transform(base_frame_in_world) # base 相对于 world
        T_world_in_base = inv_homogeneous(T_base_in_world)

        # 1. 用于逆解的初始笛卡尔位姿（end相对于world）
        start_attr_pos = f"{side}_start_pos_end_in_world" 
        start_attr_ori = f"{side}_start_ori_end_in_world"
        start_pos_end_in_world = getattr(self, start_attr_pos, None)
        if start_pos_end_in_world is None:
            # 组合用于apply_delta_pose的初始位姿：start_pos 和 start_ori
            start_T_end_in_base = pose_to_transform(cart_pos_real_end_in_base)
            start_T_end_in_world = T_base_in_world @ start_T_end_in_base
            start_cart_pos = transform_to_pose(start_T_end_in_world)
            start_pos = start_cart_pos[:3]
            start_ori = start_cart_pos[3:]
            start_ori_quat = R.from_euler('xyz', start_ori, degrees=False).as_quat()
            start_ori_quat = np.roll(start_ori_quat, 1) # 四元数：wxyz
            setattr(self, start_attr_pos, start_pos)
            setattr(self, start_attr_ori, start_ori_quat)
            start_pos_end_in_world = start_pos
            start_ori_end_in_world = start_ori_quat
            # 初始化臂角：对于7轴，使用从观测中获取的psi
            if joint_num == 7:
                elbow = float(psi)
        else:
            start_ori_end_in_world = getattr(self, start_attr_ori, None)

        # 2. 组合 pico 传入的增量坐标得到目标坐标(end相对于world)
        # target_pos_end_in_world: np.ndarry：[x, y, z]
        # target_ori_end_in_world: np.ndarray：[w, x, y, z]
        # 利用 pico 官方提供的 apply_delta_pose 函数计算目标坐标（end相对于world）
        target_pos_end_in_world, target_ori_end_in_world = apply_delta_pose(
            start_pos_end_in_world,
            start_ori_end_in_world,
            cart_delta_end_in_world[:3],
            cart_delta_end_in_world[3:],
        )
        target_ori_end_in_world = np.roll(target_ori_end_in_world, -1) # 四元数：wxyz -> xyzw
        target_ori_end_in_world_euler = R.from_quat(target_ori_end_in_world).as_euler('xyz', degrees=False)
        target_cart_pos_end_in_world = np.concatenate([target_pos_end_in_world, target_ori_end_in_world_euler])
        aim_T_end_in_world = pose_to_transform(target_cart_pos_end_in_world)

        # 3. 计算储存在数据集中的目标笛卡尔坐标（end相对于ref）
        aim_T_end_in_ref = T_base_in_ref @ T_world_in_base @ aim_T_end_in_world
        cart_pos_cmd_end_in_ref = transform_to_pose(aim_T_end_in_ref)

        # 4. 求解目标笛卡尔坐标（flan相对于base），并用于逆解
        aim_T_flan_in_base = T_ref_in_base @ aim_T_end_in_ref @ T_flan_in_end
        # 逆解：直接从 T_flan_in_base 生成四元数格式的笛卡尔坐标
        end_effector_list = _transform_to_quat_wfirst(aim_T_flan_in_base)
        psi = float(elbow) if elbow is not None else 0.0
        q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(q_current.tolist(), end_effector_list, psi)
        if ec != 0:
            # 当逆解失败时保持位置不动
            q_solution = q_current.tolist()

        return np.array(q_solution[:joint_num], dtype=np.float64), elbow, cart_pos_cmd_end_in_ref

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited or rokae_algo is None:
            return transition

        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        # 观测中的关节角度
        left_q = np.array([obs[f"left_joint_pos{i}"] for i in range(self.left_joint_num)], dtype=np.float64)
        right_q = np.array([obs[f"right_joint_pos{i}"] for i in range(self.right_joint_num)], dtype=np.float64)
        
        # 观测中的 cart_pos 是 end_in_base（机器人直接返回）
        left_cart_pos_real_end_in_base = np.array([obs.get(f"left_cart_pos{i}", 0.0) for i in range(6)], dtype=np.float64)
        right_cart_pos_real_end_in_base = np.array([obs.get(f"right_cart_pos{i}", 0.0) for i in range(6)], dtype=np.float64)
        
        # 统一处理 psi：非7轴时为 0.0
        left_psi = float(obs.get("left_psi", 0.0))
        right_psi = float(obs.get("right_psi", 0.0))
        
        left_delta_end_in_world = np.array([action.get(f"left_target_{axis}", 0.0) for axis in ["x", "y", "z", "wx", "wy", "wz"]])
        right_delta_end_in_world = np.array([action.get(f"right_target_{axis}", 0.0) for axis in ["x", "y", "z", "wx", "wy", "wz"]])

        left_rbv, left_min, left_max = self._effective_left_params()
        rokae_algo.cross_wrist7_init(left_rbv, left_min, left_max)
        left_q_sol, self._left_elbow, left_cart_pos_cmd_end_in_ref = self._run_arm(
            side="left",
            q_current=left_q,
            cart_delta_end_in_world=left_delta_end_in_world,
            elbow=self._left_elbow,
            cart_pos_real_end_in_base=left_cart_pos_real_end_in_base,
            psi=left_psi,
            tool_end_pos=self.left_tool_end_pos,
            tool_ref_pos=self.left_tool_ref_pos,
            base_frame_in_world=self.left_base_frame_in_world,
            joint_num=self.left_joint_num,
        )
        rokae_algo.de_init()

        right_rbv, right_min, right_max = self._effective_right_params()
        rokae_algo.cross_wrist7_init(right_rbv, right_min, right_max)
        right_q_sol, self._right_elbow, right_cart_pos_cmd_end_in_ref = self._run_arm(
            side="right",
            q_current=right_q,
            cart_delta_end_in_world=right_delta_end_in_world,
            elbow=self._right_elbow,
            cart_pos_real_end_in_base=right_cart_pos_real_end_in_base,
            psi=right_psi,
            tool_end_pos=self.right_tool_end_pos,
            tool_ref_pos=self.right_tool_ref_pos,
            base_frame_in_world=self.right_base_frame_in_world,
            joint_num=self.right_joint_num,
        )
        rokae_algo.de_init()

        # 总是输出 cart_pos、joint_pos 和 cart_vel（用于记录到数据集）
        # 保持 pico 端传入的 gripper_pos 原样透传，不在此处覆盖
        # robot processor 会根据 callback_mode 选择发送哪个字段给机器人（在 robot_action_processor 中处理）
        for i in range(6):
            action[f"left_cart_pos{i}"] = float(left_cart_pos_cmd_end_in_ref[i])
            action[f"right_cart_pos{i}"] = float(right_cart_pos_cmd_end_in_ref[i])
        for i in range(self.left_joint_num):
            action[f"left_joint_pos{i}"] = float(left_q_sol[i])
        for i in range(self.right_joint_num):
            action[f"right_joint_pos{i}"] = float(right_q_sol[i])
        
        # 总是输出 psi（用于记录到数据集），非7轴时为 0.0
        action["left_psi"] = float(left_psi)
        action["right_psi"] = float(right_psi)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 总是添加 cart_pos 和 joint_pos 特征（用于记录到数据集）
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
        
        # psi 特征（总是添加，非7轴时为 0.0）
        action_features.setdefault("left_psi", float)
        action_features.setdefault("right_psi", float)
        
        # gripper_pos 特征
        action_features.setdefault("left_gripper_pos", float)
        action_features.setdefault("right_gripper_pos", float)

        # 移除 pico 端返回的增量坐标特征
        keys_to_remove = [
            k
            for k in list(action_features.keys())
            if k.startswith("left_target_") or k.startswith("right_target_")
        ]
        for k in keys_to_remove:
            action_features.pop(k, None)
        
        return features
