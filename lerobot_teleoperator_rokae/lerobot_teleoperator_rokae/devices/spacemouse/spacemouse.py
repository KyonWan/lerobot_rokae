from lerobot.teleoperators.teleoperator import Teleoperator
from .config_spacemouse import SpacemouseConfig
from .spacemouse_expert import SpaceMouseExpert
from typing import Any

class Spacemouse(Teleoperator):
    config_class = SpacemouseConfig
    name = "spacemouse"
    
    def __init__(self, config:SpacemouseConfig):
        super().__init__(config)
        self.config = config
        device_index = getattr(config, 'device_index', None)
        # self.space_mouse_expert = SpaceMouseExpert(device_index=device_index)
    
    @property
    def action_features(self) -> dict:
        return {**{"cart_vel{i}": float for i in range(6)}, "buttons": (2,1)}

    @property
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        # cart_vels, buttons = self.space_mouse_expert.get_action()
        # cart_vels = cart_vels.tolist()
        cart_vels = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0]
        buttons = [0, 0]
        return {**{f"cart_vel{i}":cart_vels[i] for i in range(6)}, "buttons": buttons}
    
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        pass