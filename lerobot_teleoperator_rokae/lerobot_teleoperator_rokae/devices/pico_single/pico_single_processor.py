from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.transition import Transition

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


def _quat_wfirst_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """[w, x, y, z] 四元数转欧拉角 xyz。"""
    quat_xyzw = np.roll(np.array(quat_wxyz, dtype=np.float64), -1)
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


@ProcessorStepRegistry.register("pico_single_inverse_kinematics_processor")
@dataclass
class PicoSingleInverseKinematicsProcessor(ProcessorStep):
    """
    pico 单臂版 IK Processor。

    - 输入: `target_*` 增量（世界系）
    - 输出: 单臂键名 `joint_pos*` / `cart_pos*` / `psi` / `gripper_pos`
    """

    joint_num: int = 7
    initial_gripper_state: int = 1
    rbv: list[float] = field(default_factory=list)
    min_joint: list[float] = field(default_factory=list)
    max_joint: list[float] = field(default_factory=list)

    # 机器人几何配置（由 robot 实例注入）
    tool_end_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    _elbow: float | None = field(default=None, init=False, repr=False)
    _inited: bool = field(default=False, init=False, repr=False)
    start_pos_end_in_world: np.ndarray | None = field(default=None, init=False, repr=False)
    start_ori_end_in_world: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.gripper_state = self.initial_gripper_state
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
        self.reset()

    def __del__(self) -> None:
        if not getattr(self, "_inited", False):
            return
        try:
            if rokae_algo is not None:
                rokae_algo.de_init()
        except Exception:
            pass

    def reset(self, gripper_state: int | None = None) -> None:
        self._elbow = None
        self.gripper_state = gripper_state if gripper_state is not None else self.initial_gripper_state
        self.start_pos_end_in_world = None
        self.start_ori_end_in_world = None

    def _run_ik(
        self,
        q_current: np.ndarray,
        cart_delta_end_in_world: np.ndarray,
        cart_pos_real_end_in_base: np.ndarray,
        psi: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        流程:
        pico delta(world) -> target end@world -> end@ref -> flange@base -> IK -> joint_pos。
        """
        T_end_in_flan = pose_to_transform(self.tool_end_pos)
        T_flan_in_end = inv_homogeneous(T_end_in_flan)

        T_ref_in_base, _ = compute_base_ref_transform(self.tool_ref_pos, self.base_frame_in_world)
        T_base_in_ref = inv_homogeneous(T_ref_in_base)

        T_base_in_world = pose_to_transform(self.base_frame_in_world)
        T_world_in_base = inv_homogeneous(T_base_in_world)

        if self.start_pos_end_in_world is None:
            start_T_end_in_base = pose_to_transform(cart_pos_real_end_in_base)
            start_T_end_in_world = T_base_in_world @ start_T_end_in_base
            start_cart_pos = transform_to_pose(start_T_end_in_world)
            self.start_pos_end_in_world = start_cart_pos[:3]
            start_ori_quat = R.from_euler("xyz", start_cart_pos[3:], degrees=False).as_quat()
            self.start_ori_end_in_world = np.roll(start_ori_quat, 1)  # wxyz
            if self.joint_num == 7:
                self._elbow = float(psi)

        target_pos_end_in_world, target_ori_end_in_world = apply_delta_pose(
            self.start_pos_end_in_world,
            self.start_ori_end_in_world,
            cart_delta_end_in_world[:3],
            cart_delta_end_in_world[3:],
        )

        target_cart_pos_end_in_world = np.concatenate(
            [target_pos_end_in_world, _quat_wfirst_to_euler_xyz(target_ori_end_in_world)]
        )
        aim_T_end_in_world = pose_to_transform(target_cart_pos_end_in_world)

        aim_T_end_in_ref = T_base_in_ref @ T_world_in_base @ aim_T_end_in_world
        cart_pos_cmd_end_in_ref = transform_to_pose(aim_T_end_in_ref)

        aim_T_flan_in_base = T_ref_in_base @ aim_T_end_in_ref @ T_flan_in_end
        end_effector_list = _transform_to_quat_wfirst(aim_T_flan_in_base)

        # 6轴/7轴的唯一差异：逆解器不同；笛卡尔坐标变换流程完全一致。
        if self.joint_num == 6:
            q_solution, ec = rokae_algo.cr6_cart2jnt(q_current.tolist(), end_effector_list)
        else:  # joint_num == 7
            psi_input = float(self._elbow) if self._elbow is not None else 0.0
            q_solution, ec = rokae_algo.cross_wrist7_cart2jnt(
                q_current.tolist(),
                end_effector_list,
                psi_input,
            )

        if ec != 0:
            q_solution = q_current.tolist()

        return np.array(q_solution[: self.joint_num], dtype=np.float64), cart_pos_cmd_end_in_ref

    def __call__(self, transition: Transition) -> Transition:
        if not self._inited or rokae_algo is None:
            return transition

        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)

        q_current = np.array([obs[f"joint_pos{i}"] for i in range(self.joint_num)], dtype=np.float64)
        cart_pos_real_end_in_base = np.array([obs.get(f"cart_pos{i}", 0.0) for i in range(6)], dtype=np.float64)
        psi = float(obs.get("psi", 0.0)) if self.joint_num == 7 else 0.0

        cart_delta_end_in_world = np.array(
            [action.get(f"target_{axis}", 0.0) for axis in ["x", "y", "z", "wx", "wy", "wz"]],
            dtype=np.float64,
        )
        self.gripper_state = float(action.get("gripper_pos", self.gripper_state))

        q_solution, cart_pos_cmd_end_in_ref = self._run_ik(
            q_current=q_current,
            cart_delta_end_in_world=cart_delta_end_in_world,
            cart_pos_real_end_in_base=cart_pos_real_end_in_base,
            psi=psi,
        )

        for i in range(6):
            action[f"cart_pos{i}"] = float(cart_pos_cmd_end_in_ref[i])
        for i in range(self.joint_num):
            action[f"joint_pos{i}"] = float(q_solution[i])

        action["psi"] = float(psi)
        action["gripper_pos"] = float(self.gripper_state)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]

        for i in range(self.joint_num):
            action_features.setdefault(f"joint_pos{i}", float)

        for i in range(6):
            action_features.setdefault(f"cart_pos{i}", float)

        action_features.setdefault("psi", float)
        action_features.setdefault("gripper_pos", float)

        keys_to_remove = [
            k
            for k in list(action_features.keys())
            if k.startswith("target_") or k.startswith("left_target_") or k.startswith("right_target_")
        ]
        for k in keys_to_remove:
            action_features.pop(k, None)

        return features
