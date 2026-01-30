from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.transition import Transition
from dataclasses import dataclass


@ProcessorStepRegistry.register("generate_bi_joint_pos_cmd")
@dataclass
class GenerateBiJointPosCmd(ProcessorStep):
    """
    Processor for bimanual spacemouse: generates joint position commands from cartesian velocities
    by maintaining current joint positions (similar to single arm GenerateJointPosCmd).
    This copies joint positions from observation to action, and extracts gripper states from buttons.
    """
    left_joint_num: int = 7
    right_joint_num: int = 7
    
    def __call__(self, transition: Transition) -> Transition:
        """
        Process transition: copy joint positions from observation to action,
        and extract gripper states from buttons.
        """
        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        
        # Extract buttons from action
        left_buttons = action.pop("left_buttons", [0, 0])
        right_buttons = action.pop("right_buttons", [0, 0])
        
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
        
        # Extract gripper states from buttons
        if isinstance(left_buttons, (list, tuple)) and len(left_buttons) > 0:
            action["left_gripper_pos"] = left_buttons[0]
        else:
            action["left_gripper_pos"] = 0
        
        if isinstance(right_buttons, (list, tuple)) and len(right_buttons) > 0:
            action["right_gripper_pos"] = right_buttons[0]
        else:
            action["right_gripper_pos"] = 0
        
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
