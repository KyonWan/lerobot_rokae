"""Pico 设备专属 processors。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.transition import Transition

from rokae_python_wrapper.rokae_kinematics.cartesian_frames import (
    apply_delta_pos_flan_in_base,
    delta_pos_end_in_ref_to_flan_in_base,
)

from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common import (
    ArmPipeline,
    ensure_cart_pos_flan_in_base,
    register_gripper_action_feature,
)

from .runtime import PicoArmRuntime


def _pico_delta_from_action(action: dict, key_prefix: str) -> np.ndarray:
    axes = ["x", "y", "z", "wx", "wy", "wz"]
    return np.array(
        [action.get(f"{key_prefix}target_{axis}", 0.0) for axis in axes],
        dtype=np.float64,
    )


def _quat_wfirst_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.roll(np.asarray(quat_wxyz, dtype=np.float64), -1)
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


@ProcessorStepRegistry.register("pico_delta_pos_end_in_ref_to_flan_in_base_processor")
@dataclass
class PicoDeltaPosEndInRefToFlanInBaseProcessor(ProcessorStep):
    pipeline: ArmPipeline[PicoArmRuntime]

    def _ensure_session_start_flan_in_base(self, cfg, rt, obs: dict) -> None:
        if (
            rt.session_start_pos_flan_in_base is not None
            and rt.session_start_ori_flan_in_base is not None
        ):
            return

        ensure_cart_pos_flan_in_base(cfg, rt, obs)
        assert rt.cart_pos_flan_in_base is not None
        start_flan = rt.cart_pos_flan_in_base
        quat_xyzw = R.from_euler("xyz", start_flan[3:], degrees=False).as_quat()
        rt.session_start_pos_flan_in_base = start_flan[:3].copy()
        rt.session_start_ori_flan_in_base = np.roll(quat_xyzw, 1)

    def _start_flan_pose(self, rt) -> np.ndarray:
        assert rt.session_start_pos_flan_in_base is not None
        assert rt.session_start_ori_flan_in_base is not None
        euler = _quat_wfirst_to_euler_xyz(rt.session_start_ori_flan_in_base)
        return np.concatenate([rt.session_start_pos_flan_in_base, euler])

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        pipeline = self.pipeline

        for cfg, rt in zip(pipeline.arms, pipeline.runtimes):
            p = cfg.key_prefix
            cart_delta_end_in_ref = _pico_delta_from_action(action, p)
            self._ensure_session_start_flan_in_base(cfg, rt, obs)
            start_flan = self._start_flan_pose(rt)
            delta_pos_flan_in_base = delta_pos_end_in_ref_to_flan_in_base(
                cart_delta_end_in_ref,
                cfg.tool_end_pos,
                cfg.tool_ref_pos,
                cfg.base_frame_in_world,
                start_flan,
            )
            for i in range(6):
                action[f"{p}cart_pos{i}"] = float(delta_pos_flan_in_base[i])

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            p = cfg.key_prefix
            for i in range(6):
                action_features.setdefault(f"{p}cart_pos{i}", float)
        for k in list(action_features.keys()):
            if "target_" in k:
                action_features.pop(k, None)
        return features


@ProcessorStepRegistry.register("integrate_flan_in_base_from_start_processor")
@dataclass
class PicoIntegrateFlanInBaseFromStartProcessor(ProcessorStep):
    pipeline: ArmPipeline[PicoArmRuntime]

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION)
        for cfg, rt in zip(self.pipeline.arms, self.pipeline.runtimes):
            p = cfg.key_prefix
            delta_pos_flan_in_base = np.array(
                [action.get(f"{p}cart_pos{i}", 0.0) for i in range(6)],
                dtype=np.float64,
            )
            ensure_cart_pos_flan_in_base(cfg, rt, obs)
            assert rt.session_start_pos_flan_in_base is not None
            assert rt.session_start_ori_flan_in_base is not None
            start_flan = np.concatenate(
                [
                    rt.session_start_pos_flan_in_base,
                    _quat_wfirst_to_euler_xyz(rt.session_start_ori_flan_in_base),
                ]
            )
            cart_pos_flan_in_base = apply_delta_pos_flan_in_base(
                start_flan,
                delta_pos_flan_in_base,
            )
            for i in range(6):
                action[f"{p}cart_pos{i}"] = float(cart_pos_flan_in_base[i])
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            p = cfg.key_prefix
            for i in range(6):
                action_features.setdefault(f"{p}cart_pos{i}", float)
        for k in list(action_features.keys()):
            if "target_" in k:
                action_features.pop(k, None)
        return features


@ProcessorStepRegistry.register("pico_gripper_processor")
@dataclass
class PicoGripperProcessor(ProcessorStep):
    pipeline: ArmPipeline[PicoArmRuntime]
    trigger_reverse: bool = True
    trigger_threshold: float = 0.5
    close_position: float = 0.0
    open_position: float = 1.0

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
            rt.gripper_state = (
                override if override is not None else cfg.initial_gripper_state
            )
            rt.last_gripper_trigger = self.open_position

    @property
    def raw_open_trigger(self) -> float:
        return 0.0 if self.trigger_reverse else 1.0

    def _binarize_trigger(self, raw_trigger: float) -> float:
        trigger = 1.0 - raw_trigger if self.trigger_reverse else raw_trigger
        if trigger < self.trigger_threshold:
            return self.close_position
        return self.open_position

    def __call__(self, transition: Transition) -> Transition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        for cfg, rt in zip(self.pipeline.arms, self.pipeline.runtimes):
            p = cfg.key_prefix
            raw_trigger = float(action.get(f"{p}gripper_trigger", self.raw_open_trigger))
            gripper_trigger = self._binarize_trigger(raw_trigger)
            if (
                rt.last_gripper_trigger == self.open_position
                and gripper_trigger == self.close_position
            ):
                rt.gripper_state = 0 if rt.gripper_state == 1 else 1
            rt.last_gripper_trigger = gripper_trigger
            action[f"{p}gripper_pos"] = float(rt.gripper_state)
            action.pop(f"{p}gripper_trigger", None)
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for cfg in self.pipeline.arms:
            register_gripper_action_feature(action_features, cfg.key_prefix)
            action_features.pop(f"{cfg.key_prefix}gripper_trigger", None)
        return features


__all__ = [
    "PicoDeltaPosEndInRefToFlanInBaseProcessor",
    "PicoIntegrateFlanInBaseFromStartProcessor",
    "PicoGripperProcessor",
]
