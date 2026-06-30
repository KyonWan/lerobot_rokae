#!/usr/bin/env python

from typing import Any

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import IdentityProcessorStep

from .configuration_rokae_custom import RokaeCustomConfig


def make_rokae_custom_pre_post_processors(
    config: RokaeCustomConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create placeholder pre/post pipelines for rokae_custom policy.

    This intentionally starts with identity transformations so the package can be installed,
    discovered, and iterated on without coupling unfinished logic to runtime behavior.
    """
    _ = (config, dataset_stats)

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[IdentityProcessorStep()],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[IdentityProcessorStep()],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
