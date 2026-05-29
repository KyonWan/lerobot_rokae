"""各 teleop 设备共用的 arm pipeline 基础定义与初始化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from lerobot.processor.pipeline import ProcessorStep

from rokae_python_wrapper.rokae_kinematics.cartesian_ik_solver import (
    Pink7AxisIkSolver,
    create_cartesian_ik_solver,
)

from .config import ArmConfig, CoreArmRuntime

RT = TypeVar("RT", bound=CoreArmRuntime)


@dataclass
class ArmPipeline(Generic[RT]):
    arms: list[ArmConfig]
    control_period: float
    gripper_processor: Any = None
    steps: list[ProcessorStep] = field(default_factory=list)
    runtimes: list[RT] = field(default_factory=list, repr=False)

    @property
    def is_bimanual(self) -> bool:
        return len(self.arms) > 1

    @property
    def initial_gripper_state(self) -> int:
        if self.gripper_processor is None:
            return self.arms[0].initial_gripper_state
        return self.gripper_processor.initial_gripper_state

    @property
    def initial_left_gripper_state(self) -> int:
        if self.gripper_processor is None:
            return self.arms[0].initial_gripper_state
        return self.gripper_processor.initial_left_gripper_state

    @property
    def initial_right_gripper_state(self) -> int:
        if self.gripper_processor is None:
            return self.arms[1].initial_gripper_state if len(self.arms) > 1 else 1
        return self.gripper_processor.initial_right_gripper_state

    def reset_kinematics(self) -> None:
        for rt in self.runtimes:
            rt.cart_pos_flan_in_base = None
            if rt.ik_solver is not None:
                rt.ik_solver.request_reset()

    def reset_all(
        self,
        gripper_state: int | None = None,
        left_gripper_state: int | None = None,
        right_gripper_state: int | None = None,
    ) -> None:
        if self.gripper_processor is not None:
            self.gripper_processor.reset(
                gripper_state=gripper_state,
                left_gripper_state=left_gripper_state,
                right_gripper_state=right_gripper_state,
            )
        self.reset_kinematics()


def init_arm_pipeline(
    pipeline: ArmPipeline[RT],
    runtime_factory,
) -> None:
    if not pipeline.arms:
        raise ValueError("ArmPipeline 需要至少一个 ArmConfig")
    pipeline.runtimes = [
        runtime_factory(cfg.initial_gripper_state) for cfg in pipeline.arms
    ]
    for cfg in pipeline.arms:
        if cfg.joint_num == 7 and not str(cfg.robot_type).strip():
            raise ValueError(
                f"{cfg.key_prefix!r} 7 轴逆解需要 robot_type（get_robot_info().type）"
            )
        if cfg.joint_num == 6 and not str(cfg.robot_ip).strip():
            raise ValueError(
                f"{cfg.key_prefix!r} 6 轴逆解需要 robot_ip（xCore model）"
            )
        if cfg.joint_num not in (6, 7):
            raise ValueError(
                f"{cfg.key_prefix!r} 仅支持 6 或 7 轴，got {cfg.joint_num}"
            )
    init_kinematics_backend(pipeline)


def init_kinematics_backend(pipeline: ArmPipeline[RT]) -> None:
    for cfg, rt in zip(pipeline.arms, pipeline.runtimes):
        rt.ik_solver = create_cartesian_ik_solver(
            cfg.joint_num,
            robot_type=cfg.robot_type,
            robot_ip=cfg.robot_ip,
        )
        if isinstance(rt.ik_solver, Pink7AxisIkSolver):
            rt.ik_solver.request_reset()
