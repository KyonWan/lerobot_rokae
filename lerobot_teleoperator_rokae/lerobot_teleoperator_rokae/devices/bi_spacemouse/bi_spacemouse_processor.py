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
    This copies joint positions from observation to action, and converts button presses to gripper toggle states.
    """
    left_joint_num: int = 7
    right_joint_num: int = 7
    initial_left_gripper_state: int = 1  # 初始左夹爪状态（0=close, 1=open），在episode开始前设置
    initial_right_gripper_state: int = 1  # 初始右夹爪状态（0=close, 1=open），在episode开始前设置
    
    def __post_init__(self):
        # Gripper切换状态跟踪：记录上一次按钮状态和当前gripper状态
        self.left_button_prev = 0  # 上一次左按钮状态（0=未按下，1=按下）
        self.right_button_prev = 0  # 上一次右按钮状态（0=未按下，1=按下）
        self.left_gripper_state = self.initial_left_gripper_state  # 当前左gripper状态（0=close, 1=open）
        self.right_gripper_state = self.initial_right_gripper_state  # 当前右gripper状态（0=close, 1=open）
    
    def reset(self, left_gripper_state: int | None = None, right_gripper_state: int | None = None) -> None:
        """
        Reset gripper states to initial values or specified values.
        Called at the start of each episode to synchronize gripper states.
        
        Args:
            left_gripper_state: Left gripper state to set (0=close, 1=open). If None, use initial_left_gripper_state.
            right_gripper_state: Right gripper state to set (0=close, 1=open). If None, use initial_right_gripper_state.
        """
        self.left_button_prev = 0
        self.right_button_prev = 0
        self.left_gripper_state = left_gripper_state if left_gripper_state is not None else self.initial_left_gripper_state
        self.right_gripper_state = right_gripper_state if right_gripper_state is not None else self.initial_right_gripper_state
    
    def __call__(self, transition: Transition) -> Transition:
        """
        Process transition: copy joint positions from observation to action,
        and convert button presses to gripper toggle states.
        """
        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        
        # Extract buttons from action
        left_buttons = action.pop("left_buttons")
        right_buttons = action.pop("right_buttons")
        
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
        
        # Convert button presses to gripper toggle states
        # 检测按钮上升沿（从0变为1）：切换gripper状态
        left_button_curr = 0
        if isinstance(left_buttons, (list, tuple)) and len(left_buttons) > 0:
            left_button_curr = 1 if left_buttons[0] else 0
        
        right_button_curr = 0
        if isinstance(right_buttons, (list, tuple)) and len(right_buttons) > 0:
            right_button_curr = 1 if right_buttons[0] else 0
        
        # 检测左按钮上升沿（从0变为1）：切换gripper状态
        if left_button_curr == 1 and self.left_button_prev == 0:
            # 按钮从未按下变为按下，切换gripper状态
            self.left_gripper_state = 1 - self.left_gripper_state
        
        # 检测右按钮上升沿（从0变为1）：切换gripper状态
        if right_button_curr == 1 and self.right_button_prev == 0:
            # 按钮从未按下变为按下，切换gripper状态
            self.right_gripper_state = 1 - self.right_gripper_state
        
        # 更新上一次按钮状态
        self.left_button_prev = left_button_curr
        self.right_button_prev = right_button_curr
        
        # 使用切换后的gripper状态（0=close, 1=open）
        action["left_gripper_pos"] = float(self.left_gripper_state)
        action["right_gripper_pos"] = float(self.right_gripper_state)
        
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
