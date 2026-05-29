"""SpaceMouse IK pipeline 构建。"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.processor.pipeline import ProcessorStep

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    ArmConfig,
    ArmPipeline,
    CoreArmRuntime,
    PosFlanInBaseToEndInRefProcessor,
    SolveArmProcessor,
    init_arm_pipeline,
)

from .processors import (
    DeltaPosEndInRefToFlanInBaseProcessor,
    IntegrateFlanInBaseProcessor,
    LimitAndIntegrateVelEndInRefProcessor,
    SpaceMouseGripperProcessor,
)
from .runtime import SpaceMouseArmRuntime


@dataclass
class SpaceMouseArmPipeline(ArmPipeline[SpaceMouseArmRuntime]):
    def reset_kinematics(self) -> None:
        super().reset_kinematics()
        for rt in self.runtimes:
            rt.button_prev = 0


def _runtime_factory(initial_gripper_state: int) -> SpaceMouseArmRuntime:
    return SpaceMouseArmRuntime(gripper_state=initial_gripper_state)


def build_spacemouse_arm_pipeline(
    *,
    arms: list[ArmConfig],
    control_period: float,
    include_spacemouse_gripper: bool = True,
) -> SpaceMouseArmPipeline:
    pipeline = SpaceMouseArmPipeline(
        arms=arms,
        control_period=control_period,
    )
    init_arm_pipeline(pipeline, _runtime_factory)

    steps: list[ProcessorStep] = []
    if include_spacemouse_gripper:
        gripper = SpaceMouseGripperProcessor(pipeline)
        steps.append(gripper)
        pipeline.gripper_processor = gripper

    steps.extend(
        [
            LimitAndIntegrateVelEndInRefProcessor(pipeline),
            DeltaPosEndInRefToFlanInBaseProcessor(pipeline),
            IntegrateFlanInBaseProcessor(pipeline),
            SolveArmProcessor(pipeline),
            PosFlanInBaseToEndInRefProcessor(pipeline),
        ]
    )
    pipeline.steps = steps
    return pipeline


def find_arm_pipeline(steps: list[ProcessorStep]) -> ArmPipeline[CoreArmRuntime] | None:
    for step in steps:
        pipeline = getattr(step, "pipeline", None)
        if isinstance(pipeline, ArmPipeline):
            return pipeline
    return None


# Backward-compat alias
build_arm_pipeline = build_spacemouse_arm_pipeline
