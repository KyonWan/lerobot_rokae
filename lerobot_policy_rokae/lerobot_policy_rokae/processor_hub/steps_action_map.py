#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rokae/action_contract_map")
class RokaeActionContractMapStep(ProcessorStep):
    """Placeholder for joint<->ee action contract mapping."""

    source_contract: str = "joint"
    target_contract: str = "joint"

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement actual mapping once canonical action schema is finalized.
        _ = (self.source_contract, self.target_contract)
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"source_contract": self.source_contract, "target_contract": self.target_contract}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # TODO: adapt feature metadata when action schema conversion is implemented.
        return features
