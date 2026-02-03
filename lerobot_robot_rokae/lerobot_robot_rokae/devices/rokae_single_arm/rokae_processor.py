from lerobot.processor.pipeline import RobotActionProcessorStep, ObservationProcessorStep, PolicyActionProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_STATE, ACTION
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@ProcessorStepRegistry.register("space_mouse_vel_map")
@dataclass
class ExtractCartVelAndGripper(RobotActionProcessorStep):
    TRANS_MAX_VEL = 0.1
    ROT_MAX_VEL = 0.2

    def action(self, action: RobotAction) -> RobotAction:
        trans_vel = np.array([action[f"cart_vel{i}"] for i in range(3)])
        rot_vel = np.array([action[f"cart_vel{i+3}"] for i in range(3)])

        # 限制速度
        trans_vel *= self.TRANS_MAX_VEL
        rot_vel *= self.ROT_MAX_VEL

        return {
            **{f"cart_vel{i}": float(trans_vel[i]) for i in range(3)},
            **{f"cart_vel{i+3}": float(rot_vel[i]) for i in range(3)},
            "gripper_pos": action["gripper_pos"],
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for i in range(6): # todo
            features[PipelineFeatureType.ACTION].pop(f"joint_pos{i}", None)
        return features
