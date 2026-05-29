"""SpaceMouse 设备专属 processors（cart_vel 笛卡尔速度链）。"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.transition import Transition

from rokae_python_wrapper.rokae_kinematics.action_fields import (
    scale_action_cart_vel,
    write_action_cart_pos,
)
from rokae_python_wrapper.rokae_kinematics.cartesian_frames import (
    apply_delta_pos_flan_in_base,
    delta_pos_end_in_ref_to_flan_in_base,
)

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    ArmPipeline,
    ensure_cart_pos_flan_in_base,
    pop_pipeline_scratch,
    read_pipeline_vec6,
    register_gripper_action_feature,
    write_pipeline_vec6,
)

from .runtime import SpaceMouseArmRuntime


def update_gripper_state_from_buttons(
    buttons: list | tuple,
    button_prev: int,
    gripper_state: int,
) -> tuple[int, int]:
    button_curr = 0
    if isinstance(buttons, (list, tuple)) and len(buttons) > 0:
        button_curr = 1 if buttons[0] else 0
    if button_curr == 1 and button_prev == 0:
        gripper_state = 1 - gripper_state
    return button_curr, gripper_state


@ProcessorStepRegistry.register("spacemouse_gripper_processor")
@dataclass
class SpaceMouseGripperProcessor(ProcessorStep):
    pipeline: ArmPipeline[SpaceMouseArmRuntime]

    @property
    def is_bimanual(self) -> bool:
        return self.pipeline.is_bimanual

    @property
    def initial_gripper_state(self) -> int:
        return self.pipeline.arms[0].initial_gripper_state

    @property
    def initial_left_gripper_state(self) -> int:
        return self.pipeline.arms[0].initial_gripper_state

    @property
    def initial_right_gripper_state(self) -> int:
        return (
            self.pipeline.arms[1].initial_gripper_state
            if len(self.pipeline.arms) > 1
            else 1
        )

    def reset(
        self,
        gripper_state: int | None = None,
        left_gripper_state: int | None = None,
        right_gripper_state: int | None = None,
    ) -> None:
        overrides = [gripper_state, left_gripper_state, right_gripper_state]
        for i, (cfg, rt) in enumerate(zip(self.pipeline.arms, self.pipeline.runtimes)):
            override = overrides[i] if i < len(overrides) else None
            if override is None and len(self.pipeline.arms) == 1:
                override = gripper_state
            rt.button_prev = 0
            rt.gripper_state = (
                override if override is not None else cfg.initial_gripper_state
            )

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        for cfg, rt in zip(self.pipeline.arms, self.pipeline.runtimes):
            p = cfg.key_prefix
            buttons = action.pop(f"{p}buttons", [0, 0])
            rt.button_prev, rt.gripper_state = update_gripper_state_from_buttons(
                buttons, rt.button_prev, rt.gripper_state
            )
            action[f"{p}gripper_pos"] = float(rt.gripper_state)
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            register_gripper_action_feature(action_features, cfg.key_prefix)
        return features


@ProcessorStepRegistry.register("limit_and_integrate_vel_end_in_ref_processor")
@dataclass
class LimitAndIntegrateVelEndInRefProcessor(ProcessorStep):
    pipeline: ArmPipeline[SpaceMouseArmRuntime]

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        pipeline = self.pipeline

        for cfg, _rt in zip(pipeline.arms, pipeline.runtimes):
            cart_vel = scale_action_cart_vel(
                action, cfg.trans_max_vel, cfg.rot_max_vel, cfg.key_prefix
            )
            write_pipeline_vec6(
                action,
                cfg.key_prefix,
                "delta_pos_end_in_ref",
                cart_vel * pipeline.control_period,
            )

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("delta_pos_end_in_ref_to_flan_in_base_processor")
@dataclass
class DeltaPosEndInRefToFlanInBaseProcessor(ProcessorStep):
    pipeline: ArmPipeline[SpaceMouseArmRuntime]

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        pipeline = self.pipeline

        for cfg, rt in zip(pipeline.arms, pipeline.runtimes):
            ensure_cart_pos_flan_in_base(cfg, rt, obs)
            assert rt.cart_pos_flan_in_base is not None
            delta_pos_end_in_ref = read_pipeline_vec6(
                action, cfg.key_prefix, "delta_pos_end_in_ref"
            )
            delta_pos_flan_in_base = delta_pos_end_in_ref_to_flan_in_base(
                delta_pos_end_in_ref,
                cfg.tool_end_pos,
                cfg.tool_ref_pos,
                cfg.base_frame_in_world,
                rt.cart_pos_flan_in_base,
            )
            write_pipeline_vec6(
                action,
                cfg.key_prefix,
                "delta_pos_flan_in_base",
                delta_pos_flan_in_base,
            )

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("integrate_flan_in_base_processor")
@dataclass
class IntegrateFlanInBaseProcessor(ProcessorStep):
    pipeline: ArmPipeline[SpaceMouseArmRuntime]

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        pipeline = self.pipeline

        for cfg, rt in zip(pipeline.arms, pipeline.runtimes):
            ensure_cart_pos_flan_in_base(cfg, rt, obs)
            assert rt.cart_pos_flan_in_base is not None
            delta_pos_flan = read_pipeline_vec6(
                action, cfg.key_prefix, "delta_pos_flan_in_base"
            )
            cart_pos_flan_in_base = apply_delta_pos_flan_in_base(
                rt.cart_pos_flan_in_base,
                delta_pos_flan,
            )
            write_action_cart_pos(
                action, cfg.key_prefix, cart_pos_flan_in_base
            )
            pop_pipeline_scratch(action, cfg.key_prefix)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            for i in range(6):
                action_features.setdefault(f"{cfg.key_prefix}cart_pos{i}", float)
        return features


__all__ = [
    "DeltaPosEndInRefToFlanInBaseProcessor",
    "IntegrateFlanInBaseProcessor",
    "LimitAndIntegrateVelEndInRefProcessor",
    "SpaceMouseGripperProcessor",
    "update_gripper_state_from_buttons",
]
