from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import RobotObservation
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from dataclasses import dataclass, field
import cv2
import numpy as np

from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.arm_frame_context import (
    ArmFrameContext,
)
from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.cartesian_processors import (
    CartPosBaseToRefObservationProcessor,
    CartPosRefToBaseProcessor,
    SelectActionByCallbackMode,
    bimanual_arm_specs,
    cart_pos_ref_to_base_arms,
    select_action_arms,
    single_arm_spec,
)

__all__ = [
    "ArmFrameContext",
    "CartPosRefToBaseProcessor",
    "SelectActionByCallbackMode",
    "CartPosBaseToRefObservationProcessor",
    "cart_pos_ref_to_base_arms",
    "select_action_arms",
    "single_arm_spec",
    "bimanual_arm_specs",
    "RokaeCameraCropObservationProcessor",
]


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
                    cropped = cv2.resize(
                        cropped, (target_w, target_h), interpolation=interpolation
                    )

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
                    observation_features[camera_name] = (
                        int(resize_h),
                        int(resize_w),
                        channel,
                    )
                    continue
                observation_features[camera_name] = (
                    int(height),
                    int(width),
                    channel,
                )
        return features
