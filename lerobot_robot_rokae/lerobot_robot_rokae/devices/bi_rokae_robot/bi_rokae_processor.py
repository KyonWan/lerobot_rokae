from lerobot.processor.pipeline import RobotActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from dataclasses import dataclass, field
import numpy as np

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    compute_base_ref_transform,
    inv_homogeneous,
    transform_pose,
    transform_velocity,
)


@ProcessorStepRegistry.register("bi_cart_pos_ref_to_base")
@dataclass
class BiCartPosRefToBaseProcessor(RobotActionProcessorStep):
    """
    Processor for bimanual cart_pos callback mode: converts cart_pos from end-relative-to-ref 
    to end-relative-to-base for robot control.
    
    This processor runs in robot_action_processor pipeline (before sending to robot).
    The stored action still contains end-relative-to-ref, but the robot receives 
    end-relative-to-base.
    
    Left and right arms have independent tool_ref_pos and base_frame_in_world configurations.
    """
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # left ref 相对于 world
    left_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # left base 相对于 world
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # right ref 相对于 world
    right_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # right base 相对于 world

    def action(self, action: RobotAction) -> RobotAction:
        """
        将左右臂的 cart_pos (end相对于ref) 转换为 end相对于base。
        """
        result = dict(action)
        
        # 处理左臂
        if all(f"left_cart_pos{i}" in action for i in range(6)):
            left_cart_pos_ref = np.array([action[f"left_cart_pos{i}"] for i in range(6)], dtype=np.float64)
            T_ref_in_base, _ = compute_base_ref_transform(
                self.left_tool_ref_pos, self.left_base_frame_in_world
            )
            left_cart_pos_base = transform_pose(left_cart_pos_ref, T_ref_in_base)
            for i in range(6):
                result[f"left_cart_pos{i}"] = float(left_cart_pos_base[i])
        
        # 处理右臂
        if all(f"right_cart_pos{i}" in action for i in range(6)):
            right_cart_pos_ref = np.array([action[f"right_cart_pos{i}"] for i in range(6)], dtype=np.float64)
            T_ref_in_base, _ = compute_base_ref_transform(
                self.right_tool_ref_pos, self.right_base_frame_in_world
            )
            right_cart_pos_base = transform_pose(right_cart_pos_ref, T_ref_in_base)
            for i in range(6):
                result[f"right_cart_pos{i}"] = float(right_cart_pos_base[i])
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_pos 特征存在（本 Processor 会转换 cart_pos）
        action_features = features[PipelineFeatureType.ACTION]
        for i in range(6):
            action_features.setdefault(f"left_cart_pos{i}", float)
            action_features.setdefault(f"right_cart_pos{i}", float)
        return features


@ProcessorStepRegistry.register("bi_cart_vel_ref_to_base")
@dataclass
class BiCartVelRefToBaseProcessor(RobotActionProcessorStep):
    """
    Processor for bimanual cart_vel callback mode: converts cart_vel from end-relative-to-ref 
    to end-relative-to-base for robot control.
    
    This processor runs in robot_action_processor pipeline (before sending to robot).
    The stored action still contains end-relative-to-ref, but the robot receives 
    end-relative-to-base.
    
    Left and right arms have independent tool_ref_pos and base_frame_in_world configurations.
    """
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # left ref 相对于 world
    left_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # left base 相对于 world
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # right ref 相对于 world
    right_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # right base 相对于 world

    def action(self, action: RobotAction) -> RobotAction:
        """
        将左右臂的 cart_vel (end相对于ref) 转换为 end相对于base。
        速度转换只需要旋转矩阵，不需要平移。
        """
        result = dict(action)
        
        # 处理左臂
        if all(f"left_cart_vel{i}" in action for i in range(6)):
            left_cart_vel_ref = np.array([action[f"left_cart_vel{i}"] for i in range(6)], dtype=np.float64)
            _, R_ref_in_base = compute_base_ref_transform(
                self.left_tool_ref_pos, self.left_base_frame_in_world
            )
            left_cart_vel_base = transform_velocity(left_cart_vel_ref, R_ref_in_base)
            for i in range(6):
                result[f"left_cart_vel{i}"] = float(left_cart_vel_base[i])
        
        # 处理右臂
        if all(f"right_cart_vel{i}" in action for i in range(6)):
            right_cart_vel_ref = np.array([action[f"right_cart_vel{i}"] for i in range(6)], dtype=np.float64)
            _, R_ref_in_base = compute_base_ref_transform(
                self.right_tool_ref_pos, self.right_base_frame_in_world
            )
            right_cart_vel_base = transform_velocity(right_cart_vel_ref, R_ref_in_base)
            for i in range(6):
                result[f"right_cart_vel{i}"] = float(right_cart_vel_base[i])
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_vel 特征存在（本 Processor 会转换 cart_vel）
        action_features = features[PipelineFeatureType.ACTION]
        for i in range(6):
            action_features.setdefault(f"left_cart_vel{i}", float)
            action_features.setdefault(f"right_cart_vel{i}", float)
        return features


@ProcessorStepRegistry.register("bi_select_action_by_callback_mode")
@dataclass
class BiSelectActionByCallbackMode(RobotActionProcessorStep):
    """
    Processor that selects action fields based on callback mode for bimanual robots.
    For joint_pos mode: keeps joint_pos, removes cart_pos, cart_vel, psi
    For cart_pos mode: keeps cart_pos, removes joint_pos, cart_vel, psi
    For cart_vel mode: keeps cart_vel, removes joint_pos, cart_pos, psi
    
    Left and right arms may have different joint_num.
    """
    callback_mode: str = "joint_pos"  # "joint_pos", "cart_pos", or "cart_vel"
    left_joint_num: int = 7
    right_joint_num: int = 7

    def action(self, action: RobotAction) -> RobotAction:
        """
        根据 callback_mode 选择发送给机器人的字段。
        """
        result = dict(action)
        
        if self.callback_mode == "joint_pos":
            # joint_pos mode: 保留 joint_pos，移除 cart_pos, cart_vel, psi
            for i in range(6):
                result.pop(f"left_cart_pos{i}", None)
                result.pop(f"left_cart_vel{i}", None)
                result.pop(f"right_cart_pos{i}", None)
                result.pop(f"right_cart_vel{i}", None)
            result.pop("left_psi", None)
            result.pop("right_psi", None)
        elif self.callback_mode == "cart_pos":
            # cart_pos mode: 保留 cart_pos，移除 joint_pos, cart_vel, psi
            for i in range(self.left_joint_num):
                result.pop(f"left_joint_pos{i}", None)
            for i in range(self.right_joint_num):
                result.pop(f"right_joint_pos{i}", None)
            for i in range(6):
                result.pop(f"left_cart_vel{i}", None)
                result.pop(f"right_cart_vel{i}", None)
            result.pop("left_psi", None)
            result.pop("right_psi", None)
        elif self.callback_mode == "cart_vel":
            # cart_vel mode: 保留 cart_vel，移除 joint_pos, cart_pos, psi
            for i in range(self.left_joint_num):
                result.pop(f"left_joint_pos{i}", None)
            for i in range(self.right_joint_num):
                result.pop(f"right_joint_pos{i}", None)
            for i in range(6):
                result.pop(f"left_cart_pos{i}", None)
                result.pop(f"right_cart_pos{i}", None)
            result.pop("left_psi", None)
            result.pop("right_psi", None)
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 根据 callback_mode 移除不需要的特征
        action_features = features[PipelineFeatureType.ACTION]
        
        if self.callback_mode == "joint_pos":
            # 移除 cart_pos, cart_vel, psi 特征
            for i in range(6):
                action_features.pop(f"left_cart_pos{i}", None)
                action_features.pop(f"left_cart_vel{i}", None)
                action_features.pop(f"right_cart_pos{i}", None)
                action_features.pop(f"right_cart_vel{i}", None)
            action_features.pop("left_psi", None)
            action_features.pop("right_psi", None)
        elif self.callback_mode == "cart_pos":
            # 移除 joint_pos, cart_vel, psi 特征
            for i in range(self.left_joint_num):
                action_features.pop(f"left_joint_pos{i}", None)
            for i in range(self.right_joint_num):
                action_features.pop(f"right_joint_pos{i}", None)
            for i in range(6):
                action_features.pop(f"left_cart_vel{i}", None)
                action_features.pop(f"right_cart_vel{i}", None)
            action_features.pop("left_psi", None)
            action_features.pop("right_psi", None)
        elif self.callback_mode == "cart_vel":
            # 移除 joint_pos, cart_pos, psi 特征
            for i in range(self.left_joint_num):
                action_features.pop(f"left_joint_pos{i}", None)
            for i in range(self.right_joint_num):
                action_features.pop(f"right_joint_pos{i}", None)
            for i in range(6):
                action_features.pop(f"left_cart_pos{i}", None)
                action_features.pop(f"right_cart_pos{i}", None)
            action_features.pop("left_psi", None)
            action_features.pop("right_psi", None)
        
        return features


@ProcessorStepRegistry.register("bi_cart_pos_base_to_ref_observation")
@dataclass
class BiCartPosBaseToRefObservationProcessor(ObservationProcessorStep):
    """
    Processor for bimanual observation: converts cart_pos from end-relative-to-base
    to end-relative-to-ref for dataset recording.

    This processor runs in robot_observation_processor pipeline.
    The robot provides end-relative-to-base, but we want to store end-relative-to-ref.

    Left and right arms have independent tool_ref_pos and base_frame_in_world configurations.
    """
    left_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # left ref 相对于 world
    left_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # left base 相对于 world
    right_tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # right ref 相对于 world
    right_base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # right base 相对于 world

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        将左右臂的 cart_pos (end相对于base) 转换为 end相对于ref。
        """
        result = dict(observation)

        # 处理左臂
        if all(f"left_cart_pos{i}" in observation for i in range(6)):
            left_cart_pos_end_in_base = np.array([observation[f"left_cart_pos{i}"] for i in range(6)], dtype=np.float64)

            # 计算 ref 相对于 base 的变换矩阵
            T_ref_in_base, _ = compute_base_ref_transform(
                self.left_tool_ref_pos, self.left_base_frame_in_world
            )
            # 计算 base 相对于 ref 的变换矩阵
            T_base_in_ref = inv_homogeneous(T_ref_in_base)

            # 使用统一的转换函数将 cart_pos_end_in_base 转换为 cart_pos_end_in_ref
            left_cart_pos_end_in_ref = transform_pose(left_cart_pos_end_in_base, T_base_in_ref)

            for i in range(6):
                result[f"left_cart_pos{i}"] = float(left_cart_pos_end_in_ref[i])

        # 处理右臂
        if all(f"right_cart_pos{i}" in observation for i in range(6)):
            right_cart_pos_end_in_base = np.array([observation[f"right_cart_pos{i}"] for i in range(6)], dtype=np.float64)

            # 计算 ref 相对于 base 的变换矩阵
            T_ref_in_base, _ = compute_base_ref_transform(
                self.right_tool_ref_pos, self.right_base_frame_in_world
            )
            # 计算 base 相对于 ref 的变换矩阵
            T_base_in_ref = inv_homogeneous(T_ref_in_base)

            # 使用统一的转换函数将 cart_pos_end_in_base 转换为 cart_pos_end_in_ref
            right_cart_pos_end_in_ref = transform_pose(right_cart_pos_end_in_base, T_base_in_ref)

            for i in range(6):
                result[f"right_cart_pos{i}"] = float(right_cart_pos_end_in_ref[i])

        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_pos 特征存在（本 Processor 会转换 cart_pos）
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for i in range(6):
            observation_features.setdefault(f"left_cart_pos{i}", float)
            observation_features.setdefault(f"right_cart_pos{i}", float)
        return features

