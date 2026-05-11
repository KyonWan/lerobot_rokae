from lerobot.processor.pipeline import RobotActionProcessorStep, ObservationProcessorStep, PolicyActionProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_STATE, ACTION
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    compute_base_ref_transform,
    inv_homogeneous,
    transform_pose,
    transform_velocity,
)

@ProcessorStepRegistry.register("cart_pos_ref_to_base")
@dataclass
class CartPosRefToBaseProcessor(RobotActionProcessorStep):
    """
    Processor for cart_pos callback mode: converts cart_pos from end-relative-to-ref 
    to end-relative-to-base for robot control.
    
    This processor runs in robot_action_processor pipeline (before sending to robot).
    The stored action still contains end-relative-to-ref, but the robot receives 
    end-relative-to-base.
    """
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # ref 相对于 world
    base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # base 相对于 world

    def action(self, action: RobotAction) -> RobotAction:
        """
        将 cart_pos (end相对于ref) 转换为 end相对于base。
        """
        # 检查是否有 cart_pos 字段
        if not all(f"cart_pos{i}" in action for i in range(6)):
            return action
        
        cart_pos_ref = np.array([action[f"cart_pos{i}"] for i in range(6)], dtype=np.float64)
        
        # 计算 ref 相对于 base 的变换矩阵
        T_ref_in_base, _ = compute_base_ref_transform(
            self.tool_ref_pos, self.base_frame_in_world
        )
        
        # 使用统一的转换函数将 cart_pos_ref 转换为 cart_pos_base
        cart_pos_base = transform_pose(cart_pos_ref, T_ref_in_base)
        
        # 更新 action 中的 cart_pos 为 end相对于base
        result = dict(action)
        for i in range(6):
            result[f"cart_pos{i}"] = float(cart_pos_base[i])
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_pos 特征存在（本 Processor 会转换 cart_pos）
        action_features = features[PipelineFeatureType.ACTION]
        for i in range(6):
            action_features.setdefault(f"cart_pos{i}", float)
        return features


@ProcessorStepRegistry.register("cart_vel_ref_to_base")
@dataclass
class CartVelRefToBaseProcessor(RobotActionProcessorStep):
    """
    Processor for cart_vel callback mode: converts cart_vel from end-relative-to-ref 
    to end-relative-to-base for robot control.
    
    This processor runs in robot_action_processor pipeline (before sending to robot).
    The stored action still contains end-relative-to-ref, but the robot receives 
    end-relative-to-base.
    """
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # ref 相对于 world
    base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # base 相对于 world

    def action(self, action: RobotAction) -> RobotAction:
        """
        将 cart_vel (end相对于ref) 转换为 end相对于base。
        速度转换只需要旋转矩阵，不需要平移。
        """
        # 检查是否有 cart_vel 字段
        if not all(f"cart_vel{i}" in action for i in range(6)):
            return action
        
        cart_vel_ref = np.array([action[f"cart_vel{i}"] for i in range(6)], dtype=np.float64)
        
        # 计算 ref 相对于 base 的旋转矩阵
        _, R_ref_in_base = compute_base_ref_transform(
            self.tool_ref_pos, self.base_frame_in_world
        )
        
        # 使用统一的转换函数将 cart_vel_ref 转换为 cart_vel_base
        cart_vel_base = transform_velocity(cart_vel_ref, R_ref_in_base)
        
        # 更新 action 中的 cart_vel 为 end相对于base
        result = dict(action)
        for i in range(6):
            result[f"cart_vel{i}"] = float(cart_vel_base[i])
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_vel 特征存在（本 Processor 会转换 cart_vel）
        action_features = features[PipelineFeatureType.ACTION]
        for i in range(6):
            action_features.setdefault(f"cart_vel{i}", float)
        return features


@ProcessorStepRegistry.register("select_action_by_callback_mode")
@dataclass
class SelectActionByCallbackMode(RobotActionProcessorStep):
    """
    Processor that selects action fields based on callback mode.
    For joint_pos mode: keeps joint_pos, removes cart_pos, cart_vel, psi
    For cart_pos mode: keeps cart_pos, removes joint_pos, cart_vel, psi
    For cart_vel mode: keeps cart_vel, removes joint_pos, cart_pos, psi
    """
    callback_mode: str = "joint_pos"  # "joint_pos", "cart_pos", or "cart_vel"
    joint_num: int = 6

    def action(self, action: RobotAction) -> RobotAction:
        """
        根据 callback_mode 选择发送给机器人的字段。
        """
        result = dict(action)
        
        if self.callback_mode == "joint_pos":
            # joint_pos mode: 保留 joint_pos，移除 cart_pos, cart_vel, psi
            for i in range(6):
                result.pop(f"cart_pos{i}", None)
                result.pop(f"cart_vel{i}", None)
            result.pop("psi", None)
        elif self.callback_mode == "cart_pos":
            # cart_pos mode: 保留 cart_pos，移除 joint_pos, cart_vel, psi
            for i in range(self.joint_num):
                result.pop(f"joint_pos{i}", None)
            for i in range(6):
                result.pop(f"cart_vel{i}", None)
            result.pop("psi", None)
        elif self.callback_mode == "cart_vel":
            # cart_vel mode: 保留 cart_vel，移除 joint_pos, cart_pos, psi
            for i in range(self.joint_num):
                result.pop(f"joint_pos{i}", None)
            for i in range(6):
                result.pop(f"cart_pos{i}", None)
            result.pop("psi", None)
        
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 根据 callback_mode 移除不需要的特征
        action_features = features[PipelineFeatureType.ACTION]
        
        if self.callback_mode == "joint_pos":
            # 移除 cart_pos, cart_vel, psi 特征
            for i in range(6):
                action_features.pop(f"cart_pos{i}", None)
                action_features.pop(f"cart_vel{i}", None)
            action_features.pop("psi", None)
        elif self.callback_mode == "cart_pos":
            # 移除 joint_pos, cart_vel, psi 特征
            for i in range(self.joint_num):
                action_features.pop(f"joint_pos{i}", None)
            for i in range(6):
                action_features.pop(f"cart_vel{i}", None)
            action_features.pop("psi", None)
        elif self.callback_mode == "cart_vel":
            # 移除 joint_pos, cart_pos, psi 特征
            for i in range(self.joint_num):
                action_features.pop(f"joint_pos{i}", None)
            for i in range(6):
                action_features.pop(f"cart_pos{i}", None)
            action_features.pop("psi", None)
        
        return features


@ProcessorStepRegistry.register("cart_pos_base_to_ref_observation")
@dataclass
class CartPosBaseToRefObservationProcessor(ObservationProcessorStep):
    """
    Processor for observation: converts cart_pos from end-relative-to-base
    to end-relative-to-ref for dataset recording.

    This processor runs in robot_observation_processor pipeline.
    The robot provides end-relative-to-base, but we want to store end-relative-to-ref.
    """
    tool_ref_pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))   # ref 相对于 world
    base_frame_in_world: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))  # base 相对于 world

    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        将 cart_pos (end相对于base) 转换为 end相对于ref。
        """
        # 检查是否有 cart_pos 字段
        if not all(f"cart_pos{i}" in observation for i in range(6)):
            return observation

        cart_pos_end_in_base = np.array([observation[f"cart_pos{i}"] for i in range(6)], dtype=np.float64)

        # 计算 ref 相对于 base 的变换矩阵
        T_ref_in_base, _ = compute_base_ref_transform(
            self.tool_ref_pos, self.base_frame_in_world
        )
        # 计算 base 相对于 ref 的变换矩阵
        T_base_in_ref = inv_homogeneous(T_ref_in_base)

        # 使用统一的转换函数将 cart_pos_end_in_base 转换为 cart_pos_end_in_ref
        cart_pos_end_in_ref = transform_pose(cart_pos_end_in_base, T_base_in_ref)

        # 更新 observation 中的 cart_pos 为 end相对于ref
        result = dict(observation)
        for i in range(6):
            result[f"cart_pos{i}"] = float(cart_pos_end_in_ref[i])

        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 确保 cart_pos 特征存在（本 Processor 会转换 cart_pos）
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for i in range(6):
            observation_features.setdefault(f"cart_pos{i}", float)
        return features

@ProcessorStepRegistry.register("rokae_camera_crop_observation")
@dataclass
class RokaeCameraCropObservationProcessor(ObservationProcessorStep):
    """
    根据相机名称对观测图像进行裁剪。

    示例:
    - 原图 1920x1080，配置 (300, 640, 480, 640) 可裁成 640x480。

    `camera_crop_params` 格式:
    {
        "camera_front": (top, left, height, width),
        "camera_wrist": (top, left, height, width),
    }

    `camera_resize_params` 格式:
    {
        "camera_front": (height, width),
        "camera_wrist": (height, width),
    }
    """

    camera_crop_params: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    camera_resize_params: dict[str, tuple[int, int]] = field(default_factory=dict)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        if not self.camera_crop_params:
            return observation

        result = dict(observation)
        for camera_name, crop in self.camera_crop_params.items():
            if camera_name not in observation:
                continue

            image = observation[camera_name]
            if not isinstance(image, np.ndarray):
                continue
            if image.ndim < 2:
                continue

            top, left, height, width = crop
            src_h, src_w = image.shape[:2]

            # 将裁剪窗口限制在图像边界内，避免越界
            top = max(0, min(int(top), src_h))
            left = max(0, min(int(left), src_w))
            bottom = max(top, min(top + int(height), src_h))
            right = max(left, min(left + int(width), src_w))
            if bottom <= top or right <= left:
                continue

            cropped = image[top:bottom, left:right, ...]
            target_hw = self.camera_resize_params.get(camera_name)
            if target_hw is not None:
                target_h, target_w = int(target_hw[0]), int(target_hw[1])
                if target_h > 0 and target_w > 0:
                    interpolation = cv2.INTER_AREA if (
                        cropped.shape[0] > target_h or cropped.shape[1] > target_w
                    ) else cv2.INTER_LINEAR
                    cropped = cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)

            result[camera_name] = cropped

        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.camera_crop_params:
            return features

        observation_features = features[PipelineFeatureType.OBSERVATION]
        for camera_name, (_, _, height, width) in self.camera_crop_params.items():
            if camera_name not in observation_features:
                continue
            old_shape = observation_features[camera_name]
            if isinstance(old_shape, tuple) and len(old_shape) == 3:
                _, _, channel = old_shape
                if camera_name in self.camera_resize_params:
                    resize_h, resize_w = self.camera_resize_params[camera_name]
                    observation_features[camera_name] = (int(resize_h), int(resize_w), channel)
                    continue
                observation_features[camera_name] = (int(height), int(width), channel)
        return features