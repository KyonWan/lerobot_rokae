#!/usr/bin/env python

from typing import Any

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

from .contracts import DUAL_CONTRACTS, validate_contract_name
from .registry import get_policy_profile
from .steps_action_map import RokaeActionContractMapStep
from .steps_feature_select import RokaeFeatureSelectStep
from .steps_padding import RokaeStateActionPaddingStep


def make_rokae_policy_adaptation_pipelines(
    *,
    policy_type: str,
    observation_state_names: tuple[str, ...] | None = None,
    action_names: tuple[str, ...] | None = None,
    source_contract: str = "joint",
    target_contract: str | None = None,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    enable_action_contract_map: bool = False,
    enable_padding: bool = False,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build placeholder adaptation pipelines for a target policy.

    This function is intentionally light-weight in the scaffold phase: it wires
    deterministic step order and policy profile routing without implementing heavy
    transformation math yet.
    """
    profile = get_policy_profile(policy_type)
    effective_target = target_contract or profile.preferred_contract

    validate_contract_name(source_contract)
    validate_contract_name(effective_target)
    target_spec = DUAL_CONTRACTS[effective_target]

    observation_keep_prefixes = (
        *target_spec.observation_prefixes,
        *target_spec.optional_prefixes,
        *profile.observation_extra_prefixes,
    )
    action_keep_prefixes = (
        *target_spec.action_prefixes,
        *profile.action_extra_prefixes,
    )

    def _matches_keep(name: str, keep_keys: tuple[str, ...], keep_prefixes: tuple[str, ...]) -> bool:
        name_variants = (
            name,
            f"observation.{name}",
            name.replace("observation.", "", 1) if name.startswith("observation.") else name,
        )
        return any(
            variant in keep_keys or any(variant.startswith(prefix) for prefix in keep_prefixes)
            for variant in name_variants
        )

    def _build_indices(
        names: tuple[str, ...] | None,
        keep_keys: tuple[str, ...],
        keep_prefixes: tuple[str, ...],
        field_name: str,
    ) -> tuple[int, ...]:
        if not names:
            raise ValueError(
                f"{field_name} names are required for Rokae vectorized feature selection. "
                "Please pass names from dataset.meta.info['features']."
            )
        indices = tuple(i for i, n in enumerate(names) if _matches_keep(n, keep_keys, keep_prefixes))
        if not indices:
            raise ValueError(
                f"No {field_name} indices matched for policy='{policy_type}', target_contract='{effective_target}'. "
                f"keep_keys={keep_keys}, keep_prefixes={keep_prefixes}, names={names}"
            )
        return indices

    resolved_observation_names = tuple(observation_state_names or ())
    resolved_action_names = tuple(action_names or ())
    observation_state_indices = _build_indices(
        resolved_observation_names,
        profile.observation_keep_keys,
        observation_keep_prefixes,
        "observation.state",
    )
    action_indices = _build_indices(
        resolved_action_names,
        profile.action_keep_keys,
        action_keep_prefixes,
        "action",
    )

    pre_steps: list[Any] = [
        RokaeFeatureSelectStep(
            observation_keep_keys=profile.observation_keep_keys,
            observation_keep_prefixes=observation_keep_prefixes,
            action_keep_keys=profile.action_keep_keys,
            action_keep_prefixes=action_keep_prefixes,
            observation_state_names=resolved_observation_names,
            action_names=resolved_action_names,
            observation_state_indices=observation_state_indices,
            action_indices=action_indices,
        ),
    ]
    if enable_action_contract_map:
        pre_steps.append(
            RokaeActionContractMapStep(source_contract=source_contract, target_contract=effective_target)
        )
    if enable_padding:
        pre_steps.append(RokaeStateActionPaddingStep(max_state_dim=max_state_dim, max_action_dim=max_action_dim))

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=pre_steps,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    post_steps: list[Any] = []
    if enable_action_contract_map:
        post_steps.append(RokaeActionContractMapStep(source_contract=effective_target, target_contract=source_contract))

    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=post_steps,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
