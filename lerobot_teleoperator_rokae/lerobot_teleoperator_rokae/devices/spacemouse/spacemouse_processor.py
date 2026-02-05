from lerobot.processor.pipeline import RobotActionProcessorStep, ProcessorStepRegistry, ProcessorStep
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.utils.transition import Transition
from dataclasses import dataclass, field
from math import pi as M_PI


@ProcessorStepRegistry.register("generate_joint_pos_cmd")
@dataclass
class GenerateJointPosCmd(ProcessorStep):
    """
    Processor for single arm spacemouse: generates joint position commands from cartesian velocities
    by maintaining current joint positions.
    This copies joint positions from observation to action, and converts button presses to gripper toggle states.
    """
    joint_num: int = 6
    initial_gripper_state: int = 1  # 初始夹爪状态（0=close, 1=open），在episode开始前设置
    
    def __post_init__(self):
        # Gripper切换状态跟踪：记录上一次按钮状态和当前gripper状态
        self.button_prev = 0  # 上一次按钮状态（0=未按下，1=按下）
        self.gripper_state = self.initial_gripper_state  # 当前gripper状态（0=close, 1=open）
    
    def reset(self, gripper_state: int | None = None) -> None:
        """
        Reset gripper state to initial value or specified value.
        Called at the start of each episode to synchronize gripper state.
        
        Args:
            gripper_state: Gripper state to set (0=close, 1=open). If None, use initial_gripper_state.
        """
        self.button_prev = 0
        self.gripper_state = gripper_state if gripper_state is not None else self.initial_gripper_state
    
    def __call__(self, transition: Transition) -> Transition:
        """
        Process transition: copy joint positions from observation to action,
        and convert button presses to gripper toggle states.
        """
        transition = transition.copy()
        obs = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        
        # Extract buttons from action
        buttons = action.pop("buttons", [0, 0])
        
        # Copy joint positions from observation to action
        for i in range(self.joint_num):
            action[f"joint_pos{i}"] = obs[f"joint_pos{i}"]
        
        # Convert button presses to gripper toggle states
        # 检测按钮上升沿（从0变为1）：切换gripper状态
        button_curr = 0
        if isinstance(buttons, (list, tuple)) and len(buttons) > 0:
            button_curr = 1 if buttons[0] else 0
        
        # 检测按钮上升沿（从0变为1）：切换gripper状态
        if button_curr == 1 and self.button_prev == 0:
            # 按钮从未按下变为按下，切换gripper状态
            self.gripper_state = 1 - self.gripper_state
        
        # 更新上一次按钮状态
        self.button_prev = button_curr
        
        # 使用切换后的gripper状态（0=close, 1=open）
        action["gripper_pos"] = float(self.gripper_state)
        
        transition[TransitionKey.ACTION] = action
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Ensure joint_pos features exist
        for i in range(self.joint_num):
            features[PipelineFeatureType.ACTION][f"joint_pos{i}"] = float
        # Ensure gripper_pos feature exists
        features[PipelineFeatureType.ACTION]["gripper_pos"] = float
        return features
