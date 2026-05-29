"""Pico IK pipeline 构建。"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    ArmConfig,
    ArmPipeline,
    PosFlanInBaseToEndInRefProcessor,
    SolveArmProcessor,
    init_arm_pipeline,
)

from .processors import (
    PicoDeltaPosEndInRefToFlanInBaseProcessor,
    PicoGripperProcessor,
    PicoIntegrateFlanInBaseFromStartProcessor,
)
from .runtime import PicoArmRuntime


@dataclass
class PicoArmPipeline(ArmPipeline[PicoArmRuntime]):
    def reset_kinematics(self) -> None:
        super().reset_kinematics()
        for rt in self.runtimes:
            rt.session_start_pos_flan_in_base = None
            rt.session_start_ori_flan_in_base = None


def _runtime_factory(initial_gripper_state: int) -> PicoArmRuntime:
    return PicoArmRuntime(gripper_state=initial_gripper_state)


def build_pico_arm_pipeline(
    *,
    arms: list[ArmConfig],
    control_period: float,
) -> PicoArmPipeline:
    pipeline = PicoArmPipeline(
        arms=arms,
        control_period=control_period,
    )
    init_arm_pipeline(pipeline, _runtime_factory)

    gripper = PicoGripperProcessor(pipeline)
    pipeline.gripper_processor = gripper
    pipeline.steps = [
        gripper,
        PicoDeltaPosEndInRefToFlanInBaseProcessor(pipeline),
        PicoIntegrateFlanInBaseFromStartProcessor(pipeline),
        SolveArmProcessor(pipeline),
        PosFlanInBaseToEndInRefProcessor(pipeline),
    ]
    return pipeline
