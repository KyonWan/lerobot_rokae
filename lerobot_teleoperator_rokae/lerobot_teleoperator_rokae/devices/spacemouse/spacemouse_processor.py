from lerobot.processor.pipeline import RobotActionProcessorStep, ProcessorStepRegistry, ProcessorStep
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from math import pi as M_PI


@ProcessorStepRegistry.register("generate_joint_pos_cmd")
class GenerateJointPosCmd(ProcessorStep):
    joint_num: int = 6
    
    def __call__(self, transition:Transition):
        # transition 一定是 EnvTransition / dict-like
        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        buttons = action["buttons"]
        action.pop("buttons")
        for i in range(self.joint_num):
            action[f"joint_pos{i}"] = obs[f"joint_pos{i}"]
            action["gripper_pos"] = buttons[0]
        transition[TransitionKey.ACTION] = action
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for i in range(self.joint_num): # todo
            features[PipelineFeatureType.ACTION][f"joint_pos{i}"] = float
        return features

# @ProcessorStepRegistry.register("generate_joint_pos_cmd")
# class GenerateJointPosCmd:
#     joint_num: int = 6

#     def __call__(
#         self,
#         data: tuple[dict, dict],  # (act, obs)
#     ) -> dict:
#         act, obs = data

#         act = act.copy()
#         for i in range(self.joint_num):
#             act[f"joint_pos{i}"] = obs[f"joint_pos{i}"]

#         return act

