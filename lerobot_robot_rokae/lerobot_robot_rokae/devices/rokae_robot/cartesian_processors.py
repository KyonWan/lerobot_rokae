"""ref/base 笛卡尔变换与 callback 字段筛选（单臂/双臂共用 key_prefix）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
)

from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.arm_frame_context import (
    ArmFrameContext,
)
from lerobot_robot_rokae.lerobot_robot_rokae.utils.transform_utils import (
    compute_base_ref_transform,
    inv_homogeneous,
    transform_pose,
)

ArmSpec = tuple[ArmFrameContext, str, int]  # (frame, key_prefix, joint_num)


def _has_keys(mapping: dict, prefix: str, stem: str) -> bool:
    return all(f"{prefix}{stem}{i}" in mapping for i in range(6))


def _read_vec6(mapping: dict, prefix: str, stem: str) -> np.ndarray:
    return np.array([mapping[f"{prefix}{stem}{i}"] for i in range(6)], dtype=np.float64)


def _write_vec6(result: dict, prefix: str, stem: str, values: np.ndarray) -> None:
    for i in range(6):
        result[f"{prefix}{stem}{i}"] = float(values[i])


@ProcessorStepRegistry.register("cart_pos_ref_to_base")
@dataclass
class CartPosRefToBaseProcessor(RobotActionProcessorStep):
    """
    cart_pos：end@ref → end@base（下发真机前）。
    ``arms`` 为 ``(ArmFrameContext, key_prefix)``；单臂 ``[("",)]``，双臂 ``[("left_",), ("right_",)]``。
    """

    arms: Sequence[tuple[ArmFrameContext, str]] = field(default_factory=list)

    def action(self, action: RobotAction) -> RobotAction:
        result = dict(action)
        for ctx, prefix in self.arms:
            if not _has_keys(action, prefix, "cart_pos"):
                continue
            cart_ref = _read_vec6(action, prefix, "cart_pos")
            T_ref_in_base, _ = compute_base_ref_transform(
                ctx.tool_ref_pos, ctx.base_frame_in_world
            )
            _write_vec6(result, prefix, "cart_pos", transform_pose(cart_ref, T_ref_in_base))
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for _, prefix in self.arms:
            for i in range(6):
                action_features.setdefault(f"{prefix}cart_pos{i}", float)
        return features


@ProcessorStepRegistry.register("select_action_by_callback_mode")
@dataclass
class SelectActionByCallbackMode(RobotActionProcessorStep):
    """
    按 callback_mode 保留下发字段。
    ``arms`` 为 ``(key_prefix, joint_num)`` 列表。
    """

    callback_mode: str = "joint_pos"
    arms: Sequence[tuple[str, int]] = field(default_factory=list)

    def action(self, action: RobotAction) -> RobotAction:
        result = dict(action)
        for prefix, joint_num in self.arms:
            if self.callback_mode == "joint_pos":
                for i in range(6):
                    result.pop(f"{prefix}cart_pos{i}", None)
                result.pop(f"{prefix}psi", None)
            elif self.callback_mode == "cart_pos":
                for i in range(joint_num):
                    result.pop(f"{prefix}joint_pos{i}", None)
                result.pop(f"{prefix}psi", None)
            else:
                raise ValueError(
                    f"Unsupported callback_mode {self.callback_mode!r}; "
                    "use joint_pos or cart_pos."
                )
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for prefix, joint_num in self.arms:
            if self.callback_mode == "joint_pos":
                for i in range(6):
                    action_features.pop(f"{prefix}cart_pos{i}", None)
                action_features.pop(f"{prefix}psi", None)
            elif self.callback_mode == "cart_pos":
                for i in range(joint_num):
                    action_features.pop(f"{prefix}joint_pos{i}", None)
                action_features.pop(f"{prefix}psi", None)
            else:
                raise ValueError(
                    f"Unsupported callback_mode {self.callback_mode!r}; "
                    "use joint_pos or cart_pos."
                )
        return features


@ProcessorStepRegistry.register("cart_pos_base_to_ref_observation")
@dataclass
class CartPosBaseToRefObservationProcessor(ObservationProcessorStep):
    """观测 cart_pos：end@base → end@ref（写入数据集）。"""

    arms: Sequence[tuple[ArmFrameContext, str]] = field(default_factory=list)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        result = dict(observation)
        for ctx, prefix in self.arms:
            if not _has_keys(observation, prefix, "cart_pos"):
                continue
            cart_base = _read_vec6(observation, prefix, "cart_pos")
            T_ref_in_base, _ = compute_base_ref_transform(
                ctx.tool_ref_pos, ctx.base_frame_in_world
            )
            T_base_in_ref = inv_homogeneous(T_ref_in_base)
            _write_vec6(
                result, prefix, "cart_pos", transform_pose(cart_base, T_base_in_ref)
            )
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for _, prefix in self.arms:
            for i in range(6):
                observation_features.setdefault(f"{prefix}cart_pos{i}", float)
        return features


def cart_pos_ref_to_base_arms(
    specs: Sequence[ArmSpec],
) -> list[tuple[ArmFrameContext, str]]:
    return [(ctx, prefix) for ctx, prefix, _ in specs]


def select_action_arms(specs: Sequence[ArmSpec]) -> list[tuple[str, int]]:
    return [(prefix, joint_num) for _, prefix, joint_num in specs]


def single_arm_spec(robot: Any, joint_num: int) -> list[ArmSpec]:
    return [(ArmFrameContext.from_robot(robot), "", joint_num)]


def bimanual_arm_specs(
    left_arm: Any, right_arm: Any, left_joint_num: int, right_joint_num: int
) -> list[ArmSpec]:
    return [
        (ArmFrameContext.from_robot(left_arm), "left_", left_joint_num),
        (ArmFrameContext.from_robot(right_arm), "right_", right_joint_num),
    ]
