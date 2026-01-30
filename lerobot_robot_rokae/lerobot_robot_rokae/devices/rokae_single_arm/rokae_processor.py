from lerobot.processor.pipeline import RobotActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@ProcessorStepRegistry.register("space_mouse_vel_map")
@dataclass
class ExtractCartVelAndGripper(RobotActionProcessorStep):
    TRANS_MAX_VEL = 0.1
    ROT_MAX_VEL = 0.2

    def action(self, action: RobotAction) -> RobotAction:
        trans_vel = np.array([action[f"cart_vel{i}"] for i in range(3)])
        rot_vel = np.array([action[f"cart_vel{i+3}"] for i in range(3)])
        trans_vel *= self.TRANS_MAX_VEL
        rot_vel *= self.ROT_MAX_VEL

        return {
            **{f"cart_vel{i}": float(trans_vel) for i in range(3)},
            **{f"cart_vel{i+3}": float(rot_vel) for i in range(3)},
            "gripper_pos": action["gripper_pos"],
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for i in range(6): # todo
            features[PipelineFeatureType.ACTION].pop(f"joint_pos{i}", None)
        return features




@ProcessorStepRegistry.register("map_joint8_to_joint7")
@dataclass
class MapJoint8toJoint7(RobotActionProcessorStep):
    """
    将策略输出的 joint_1, joint_2, ..., joint_7, gripper_state 做8维->7维机械臂动作映射，
    并输出目标名 joint_pos1, ..., joint_pos7, gripper_pos。
    """
    def action(self, action: RobotAction) -> RobotAction:
        # 策略输出 key: joint_1 ... joint_7, gripper_state
        # 目标 key: joint_pos1 ... joint_pos7, gripper_pos
        # 按顺序获取输入（确保是float类型）
        s8 = np.array([
            float(action.get(f"action_{i+1}", 0.0)) for i in range(7)
        ] + [float(action.get("gripper_action", 0.0))], dtype=np.float32)

        # 兼容单个或批量（此处只考虑单个，符合robot action用例）
        if s8.ndim == 1:
            s8 = s8.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        s7 = np.zeros((s8.shape[0], 7), dtype=s8.dtype)
        s7[:, 0] = s8[:, 0] + np.pi / 3
        s7[:, 1] = s8[:, 1]
        s7[:, 2] = -s8[:, 3]          # 3轴取反
        s7[:, 3] = s8[:, 4]

        positive_mask = s8[:, 5] > 0
        negative_mask = s8[:, 5] < 0
        zero_mask = s8[:, 5] == 0
        s7[positive_mask, 4] = np.pi - s8[positive_mask, 5]
        s7[negative_mask, 4] = -np.pi - s8[negative_mask, 5]
        s7[zero_mask, 4] = 0.0
        s7[:, 5] = s8[:, 6]
        s7[:, 6] = s8[:, 7]  # gripper

        # 输出名转换
        output = {f"joint_pos{i+1}": float(s7[0, i]) for i in range(7)}
        output["gripper_pos"] = float(s7[0, 6])

        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 输入特征名 joint_1...joint_7, gripper_state，输出 joint_pos1...joint_pos7, gripper_pos
        act_feats = features.get(PipelineFeatureType.ACTION, {})
        for i in range(7):
            act_feats.pop(f"action_{i+1}", None)
        act_feats.pop("gripper_action", None)
        for i in range(7):
            act_feats[f"joint_pos{i+1}"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(),
            )
        act_feats["gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(),
        )
        features[PipelineFeatureType.ACTION] = act_feats
        return features


@ProcessorStepRegistry.register("map_joint7_to_joint8_observation")
@dataclass
class MapJoint7toJoint8Observation(ObservationProcessorStep):
    """
    将观测到的 joint_pos1, joint_pos2, ..., joint_pos7, gripper_pos 做7维->8维映射，
    并输出模型需要的 joint_1, joint_2, ..., joint_7, gripper_state。
    这是 MapJoint8toJoint7 的反向映射，用于观测数据处理。
    """
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        # 输入观测 key: joint_pos1 ... joint_pos7, gripper_pos
        # 输出 key: joint_1 ... joint_7, gripper_state
        # 按顺序获取输入（确保是float类型）
        s7 = np.array([
            float(observation.get(f"joint_pos{i+1}", 0.0)) for i in range(7)
        ], dtype=np.float32)

        # 兼容单个或批量
        if s7.ndim == 1:
            s7 = s7.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        # 初始化8维数组
        s8 = np.zeros((s7.shape[0], 8), dtype=s7.dtype)
        
        # 反向映射
        s8[:, 0] = s7[:, 0] - np.pi / 3  # 反向：s7[0] = s8[0] + π/3
        s8[:, 1] = s7[:, 1]              # 直接映射
        s8[:, 2] = 0.0                   # s8[2] 在原始映射中未使用，设为0
        s8[:, 3] = -s7[:, 2]             # 反向：s7[2] = -s8[3]
        s8[:, 4] = s7[:, 3]              # 直接映射
        
        # 处理第5维的反向映射（s7[4] 到 s8[5]）
        # 原始映射：
        # - s7[4] = π - s8[5] (if s8[5] > 0) => s8[5] = π - s7[4]
        # - s7[4] = -π - s8[5] (if s8[5] < 0) => s8[5] = -π - s7[4]
        # - s7[4] = 0 (if s8[5] == 0) => s8[5] = 0
        zero_mask = np.abs(s7[:, 4]) < 1e-6  # 接近0
        positive_mask = s7[:, 4] > 0
        negative_mask = s7[:, 4] < 0
        
        s8[zero_mask, 5] = 0.0
        s8[positive_mask, 5] = np.pi - s7[positive_mask, 4]
        s8[negative_mask, 5] = -np.pi - s7[negative_mask, 4]
        
        s8[:, 6] = s7[:, 5]              # 直接映射
        s8[:, 7] = float(observation.get("gripper_pos", 0.0))  # gripper

        # 输出名转换
        output = observation.copy()
        # 移除旧的7维键
        for i in range(7):
            output.pop(f"joint_pos{i+1}", None)
        output.pop("gripper_pos", None)
        
        # 添加新的8维键
        for i in range(7):
            output[f"joint_{i+1}"] = float(s8[0, i])
        output["gripper_state"] = float(s8[0, 7])

        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 输入特征名 joint_pos1...joint_pos7, gripper_pos，输出 joint_1...joint_7, gripper_state
        obs_feats = features.get(PipelineFeatureType.OBSERVATION, {})
        
        # 移除旧的7维特征
        for i in range(7):
            obs_feats.pop(f"joint_pos{i+1}", None)
        obs_feats.pop("gripper_pos", None)
        
        # 添加新的8维特征
        for i in range(7):
            obs_feats[f"joint_{i+1}"] = PolicyFeature(
                type=FeatureType.OBSERVATION,
                shape=(),
            )
        obs_feats["gripper_state"] = PolicyFeature(
            type=FeatureType.OBSERVATION,
            shape=(),
        )
        features[PipelineFeatureType.OBSERVATION] = obs_feats
        return features
