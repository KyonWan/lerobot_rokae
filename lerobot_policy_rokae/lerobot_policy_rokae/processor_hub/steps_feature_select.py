#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rokae/feature_select")
class RokaeFeatureSelectStep(ProcessorStep):
    """Select observation/action sub-dimensions via precomputed index masks.

    This step is designed for standard LeRobot vectorized samples:
    - `transition["observation"]["observation.state"]` is a tensor
    - `transition["action"]` is a tensor
    """

    observation_keep_keys: tuple[str, ...] = field(default_factory=tuple)
    observation_keep_prefixes: tuple[str, ...] = field(default_factory=tuple)
    action_keep_keys: tuple[str, ...] = field(default_factory=tuple)
    action_keep_prefixes: tuple[str, ...] = field(default_factory=tuple)
    observation_state_names: tuple[str, ...] = field(default_factory=tuple)
    action_names: tuple[str, ...] = field(default_factory=tuple)
    observation_state_indices: tuple[int, ...] = field(default_factory=tuple)
    action_indices: tuple[int, ...] = field(default_factory=tuple)

    @staticmethod
    def _keep_key(key: str, keep_keys: tuple[str, ...], keep_prefixes: tuple[str, ...]) -> bool:
        return key in keep_keys or any(key.startswith(prefix) for prefix in keep_prefixes)

    @staticmethod
    def _build_indices(
        names: tuple[str, ...], keep_keys: tuple[str, ...], keep_prefixes: tuple[str, ...], field_name: str
    ) -> tuple[int, ...]:
        if not names:
            raise ValueError(f"{field_name} names are required to build index masks.")
        indices = tuple(
            i
            for i, name in enumerate(names)
            if RokaeFeatureSelectStep._keep_key(name, keep_keys, keep_prefixes)
        )
        if not indices:
            raise ValueError(
                f"No indices selected for {field_name}. "
                f"keep_keys={keep_keys}, keep_prefixes={keep_prefixes}, names={names}"
            )
        return indices

    def __post_init__(self) -> None:
        if not self.observation_state_indices:
            self.observation_state_indices = self._build_indices(
                self.observation_state_names,
                self.observation_keep_keys,
                self.observation_keep_prefixes,
                "observation.state",
            )
        if not self.action_indices:
            self.action_indices = self._build_indices(
                self.action_names,
                self.action_keep_keys,
                self.action_keep_prefixes,
                "action",
            )

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, None)
        if not isinstance(obs, dict) or "observation.state" not in obs:
            raise ValueError(
                "RokaeFeatureSelectStep expects vectorized observation with key 'observation.state'."
            )
        obs_state = obs["observation.state"]
        if not isinstance(obs_state, torch.Tensor):
            raise ValueError(
                f"'observation.state' must be torch.Tensor, got {type(obs_state)}."
            )
        obs_dim = obs_state.shape[-1]
        if max(self.observation_state_indices) >= obs_dim:
            raise ValueError(
                f"observation_state_indices out of range. max_idx={max(self.observation_state_indices)}, obs_dim={obs_dim}"
            )
        obs["observation.state"] = obs_state[..., list(self.observation_state_indices)]
        new_transition[TransitionKey.OBSERVATION] = obs

        action = new_transition.get(TransitionKey.ACTION, None)
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"'action' must be torch.Tensor, got {type(action)}.")
        act_dim = action.shape[-1]
        if max(self.action_indices) >= act_dim:
            raise ValueError(
                f"action_indices out of range. max_idx={max(self.action_indices)}, act_dim={act_dim}"
            )
        new_transition[TransitionKey.ACTION] = action[..., list(self.action_indices)]
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "observation_keep_keys": list(self.observation_keep_keys),
            "observation_keep_prefixes": list(self.observation_keep_prefixes),
            "action_keep_keys": list(self.action_keep_keys),
            "action_keep_prefixes": list(self.action_keep_prefixes),
            "observation_state_names": list(self.observation_state_names),
            "action_names": list(self.action_names),
            "observation_state_indices": list(self.observation_state_indices),
            "action_indices": list(self.action_indices),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = features.copy()
        obs_features = new_features[PipelineFeatureType.OBSERVATION]
        if "observation.state" in obs_features:
            obs_ft = obs_features["observation.state"]
            obs_features["observation.state"] = PolicyFeature(
                type=obs_ft.type,
                shape=(len(self.observation_state_indices),),
            )

        action_features = new_features[PipelineFeatureType.ACTION]
        if "action" in action_features:
            act_ft = action_features["action"]
            action_features["action"] = PolicyFeature(
                type=act_ft.type,
                shape=(len(self.action_indices),),
            )
        return new_features
