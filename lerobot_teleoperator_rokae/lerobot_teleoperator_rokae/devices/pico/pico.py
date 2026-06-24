from __future__ import annotations

import logging
import time
from typing import Any

from xrobotoolkit_teleop.common.xr_client import XrClient

from .base import PicoInputSpec, PicoTeleopBase
from .config_pico import PicoConfig

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

DEFAULT_MANIPULATOR_CONFIG = {
    "left_arm": PicoInputSpec(
        pose_source="left_controller",
        clutch_button="left_grip",
        gripper_trigger_source="left_trigger",
        action_prefix="left",
    ),
    "right_arm": PicoInputSpec(
        pose_source="right_controller",
        clutch_button="right_grip",
        gripper_trigger_source="right_trigger",
        action_prefix="right",
    ),
}


class Pico(PicoTeleopBase):
    """
    Pico VR Teleop class for controlling robot arms via Cartesian pose deltas.
    Outputs Cartesian pose deltas instead of joint positions, allowing the framework
    to handle inverse kinematics through processor pipelines.
    """

    config_class = PicoConfig
    name = "pico"

    def __init__(self, config: PicoConfig):
        super().__init__(
            config,
            input_specs=DEFAULT_MANIPULATOR_CONFIG,
            euler_sequence="xyz",
        )

    def get_action(self) -> dict[str, Any]:
        """
        Returns Cartesian pose deltas for each arm.
        Format compatible with EEReferenceAndDelta processor:
        - target_x, target_y, target_z: position deltas
        - target_wx, target_wy, target_wz: rotation deltas (rotation vector)
        - gripper_trigger: raw trigger input for PicoGripperProcessor
        """
        action = {}

        with self._state_lock:
            for arm_name, input_spec in self.input_specs.items():
                state = self.arm_states[arm_name]
                prefix = input_spec.action_prefix

                action[f"{prefix}_target_x"] = float(state.current_delta_xyz[0])
                action[f"{prefix}_target_y"] = float(state.current_delta_xyz[1])
                action[f"{prefix}_target_z"] = float(state.current_delta_xyz[2])
                action[f"{prefix}_target_wx"] = float(state.current_delta_rot[0])
                action[f"{prefix}_target_wy"] = float(state.current_delta_rot[1])
                action[f"{prefix}_target_wz"] = float(state.current_delta_rot[2])
                action[f"{prefix}_gripper_trigger"] = float(state.raw_gripper_trigger)

        return action


# 单独运行 python pico.py 来测试 VR 设备是否连接成功并打印数据
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    xr_client = XrClient()
    teleop_config = PicoConfig(
        xr_client=xr_client,
        fps=60.0,
        xyz_scale_factor=0.5,
        rot_scale_factor=0.5,
    )

    print(f"[TEST] Initializing Pico with config: {teleop_config}")
    teleop = Pico(teleop_config)
    teleop.connect()

    i = 0
    try:
        print("[TEST] Pico Connected. Printing actions... (Press Ctrl+C to stop)")
        while True:
            action = teleop.get_action()
            print(
                f"当前运行时长：{i * 0.5}s\n"
                f"right_target_x: {action['right_target_x']:.4f}\n"
                f"right_target_y: {action['right_target_y']:.4f}\n"
                f"right_target_z: {action['right_target_z']:.4f}\n"
            )
            time.sleep(0.5)
            i += 1
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user.")
    finally:
        teleop.disconnect()
