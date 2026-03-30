#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rokae/state_action_padding")
class RokaeStateActionPaddingStep(ProcessorStep):
    """Placeholder for policy-specific state/action dimensional padding."""

    max_state_dim: int = 64
    max_action_dim: int = 32

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement tensor padding/trimming by policy profile.
        _ = (self.max_state_dim, self.max_action_dim)
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim, "max_action_dim": self.max_action_dim}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # TODO: update shapes in feature metadata after padding logic is introduced.
        return features
