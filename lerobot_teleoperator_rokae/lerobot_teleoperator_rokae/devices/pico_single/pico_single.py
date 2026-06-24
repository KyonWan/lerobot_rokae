from __future__ import annotations

from typing import Any

from ..pico.base import PicoInputSpec, PicoTeleopBase
from .config_pico_single import PicoSingleConfig

SIDE_MANIPULATOR_CONFIG = {
    "left": PicoInputSpec(
        pose_source="left_controller",
        clutch_button="left_grip",
        gripper_trigger_source="left_trigger",
    ),
    "right": PicoInputSpec(
        pose_source="right_controller",
        clutch_button="right_grip",
        gripper_trigger_source="right_trigger",
    ),
}


class PicoSingle(PicoTeleopBase):
    """
    Pico 单臂遥操器。

    通过 `config.side` 选择左/右手柄作为输入，仅输出单臂 action 键：
    `target_x/y/z/wx/wy/wz` 与作为夹爪控制输入的 `gripper_trigger`。
    """

    config_class = PicoSingleConfig
    name = "pico_single"

    def __init__(self, config: PicoSingleConfig):
        self.side_config = SIDE_MANIPULATOR_CONFIG[config.side]
        super().__init__(
            config,
            input_specs={"single_arm": self.side_config},
            euler_sequence="ZYX",
        )

    def get_action(self) -> dict[str, Any]:
        with self._state_lock:
            state = self.arm_states["single_arm"]
            return {
                "target_x": float(state.current_delta_xyz[0]),
                "target_y": float(state.current_delta_xyz[1]),
                "target_z": float(state.current_delta_xyz[2]),
                "target_wx": float(state.current_delta_rot[0]),
                "target_wy": float(state.current_delta_rot[1]),
                "target_wz": float(state.current_delta_rot[2]),
                "gripper_trigger": float(state.raw_gripper_trigger),
            }
