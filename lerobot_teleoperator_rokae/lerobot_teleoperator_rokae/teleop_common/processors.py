"""各 teleop 设备共用的 processor（SolveArm、末端位姿写回）。"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.transition import Transition

from rokae_python_wrapper.rokae_kinematics.action_fields import (
    obs_joint_vector,
    read_action_cart_pos,
    write_action_cart_pos,
)
from rokae_python_wrapper.rokae_kinematics.cartesian_frames import (
    pos_flan_in_base_to_end_in_ref,
)
from rokae_python_wrapper.rokae_kinematics.transform_utils import pose_to_transform

from .config import (
    ArmConfig,
    CoreArmRuntime,
    register_arm_kinematics_features,
    write_arm_joints,
)
from .pipeline_base import ArmPipeline


@ProcessorStepRegistry.register("solve_arm_processor")
@dataclass
class SolveArmProcessor(ProcessorStep):
    pipeline: ArmPipeline[CoreArmRuntime]

    def _solve_arm(
        self,
        cfg: ArmConfig,
        rt: CoreArmRuntime,
        action: dict,
        obs: dict,
    ) -> None:
        pipeline = self.pipeline
        q_current = obs_joint_vector(obs, cfg.joint_num, cfg.key_prefix)

        cart_pos_flan_in_base = read_action_cart_pos(action, cfg.key_prefix)
        T_flan = pose_to_transform(cart_pos_flan_in_base)

        prev_flan = rt.cart_pos_flan_in_base
        cart_vel = (
            (cart_pos_flan_in_base - prev_flan) / pipeline.control_period
            if prev_flan is not None
            else None
        )

        assert rt.ik_solver is not None
        q_solution, ik_ok = rt.ik_solver.solve(
            T_flan, q_current, pipeline.control_period, cart_vel=cart_vel
        )

        q_out = q_solution if ik_ok else q_current.tolist()
        write_arm_joints(action, cfg.key_prefix, q_out, cfg.joint_num)

        if ik_ok:
            rt.cart_pos_flan_in_base = cart_pos_flan_in_base.copy()
        elif rt.cart_pos_flan_in_base is not None:
            write_action_cart_pos(
                action, cfg.key_prefix, rt.cart_pos_flan_in_base
            )

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        pipeline = self.pipeline

        for cfg, rt in zip(pipeline.arms, pipeline.runtimes):
            self._solve_arm(cfg, rt, action, obs)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            for i in range(cfg.joint_num):
                action_features.setdefault(f"{cfg.key_prefix}joint_pos{i}", float)
        return features


@ProcessorStepRegistry.register("pos_flan_in_base_to_end_in_ref_processor")
@dataclass
class PosFlanInBaseToEndInRefProcessor(ProcessorStep):
    pipeline: ArmPipeline[CoreArmRuntime]

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        pipeline = self.pipeline

        for cfg, _rt in zip(pipeline.arms, pipeline.runtimes):
            p = cfg.key_prefix
            cart_pos_flan_in_base = read_action_cart_pos(action, p)
            cart_pos_end_in_ref = pos_flan_in_base_to_end_in_ref(
                cart_pos_flan_in_base,
                cfg.tool_end_pos,
                cfg.tool_ref_pos,
                cfg.base_frame_in_world,
            )
            write_action_cart_pos(action, p, cart_pos_end_in_ref)
            action[f"{p}psi"] = float(obs.get(f"{p}psi", 0.0))

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            register_arm_kinematics_features(
                action_features, cfg.key_prefix, cfg.joint_num
            )
        return features
