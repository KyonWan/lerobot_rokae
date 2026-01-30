from functools import cached_property
from lerobot.teleoperators.teleoperator import Teleoperator
from .config_bi_spacemouse import BiSpacemouseConfig
from ..spacemouse.spacemouse import Spacemouse
from ..spacemouse.config_spacemouse import SpacemouseConfig
from typing import Any


class BiSpacemouse(Teleoperator):
    """
    Bimanual SpaceMouse teleoperator for controlling two single-arm robots.
    Uses two SpaceMouse devices, one for each arm.
    """
    
    config_class = BiSpacemouseConfig
    name = "bi_spacemouse"
    
    def __init__(self, config: BiSpacemouseConfig):
        super().__init__(config)
        self.config = config
        
        left_config = SpacemouseConfig(
            id=f"{config.id}_left" if config.id else None,
            device_index=config.left_device_index,
        )
        
        right_config = SpacemouseConfig(
            id=f"{config.id}_right" if config.id else None,
            device_index=config.right_device_index,
        )
        
        self.left_spacemouse = Spacemouse(left_config)
        self.right_spacemouse = Spacemouse(right_config)
    
    @cached_property
    def action_features(self) -> dict:
        left_features = {f"left_cart_vel{i}": float for i in range(6)}
        left_features["left_buttons"] = (2,)  # Tuple for shape, not (2, 1)
        right_features = {f"right_cart_vel{i}": float for i in range(6)}
        right_features["right_buttons"] = (2,)  # Tuple for shape, not (2, 1)
        return {**left_features, **right_features}
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self.left_spacemouse.is_connected and self.right_spacemouse.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        self.left_spacemouse.connect(calibrate)
        self.right_spacemouse.connect(calibrate)
    
    @property
    def is_calibrated(self) -> bool:
        return self.left_spacemouse.is_calibrated and self.right_spacemouse.is_calibrated
    
    def calibrate(self) -> None:
        self.left_spacemouse.calibrate()
        self.right_spacemouse.calibrate()
    
    def configure(self) -> None:
        self.left_spacemouse.configure()
        self.right_spacemouse.configure()
    
    def get_action(self) -> dict[str, Any]:
        action_dict = {}
        
        # Get left arm action and add "left_" prefix
        left_action = self.left_spacemouse.get_action()
        for key, value in left_action.items():
            if key == "buttons":
                action_dict["left_buttons"] = value
            else:
                # key is like "cart_vel0", "cart_vel1", etc.
                action_dict[f"left_{key}"] = value
        
        # Get right arm action and add "right_" prefix
        right_action = self.right_spacemouse.get_action()
        for key, value in right_action.items():
            if key == "buttons":
                action_dict["right_buttons"] = value
            else:
                # key is like "cart_vel0", "cart_vel1", etc.
                action_dict[f"right_{key}"] = value
        
        return action_dict
    
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value 
            for key, value in feedback.items() 
            if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value 
            for key, value in feedback.items() 
            if key.startswith("right_")
        }
        
        if left_feedback:
            self.left_spacemouse.send_feedback(left_feedback)
        if right_feedback:
            self.right_spacemouse.send_feedback(right_feedback)
    
    def disconnect(self) -> None:
        self.left_spacemouse.disconnect()
        self.right_spacemouse.disconnect()
