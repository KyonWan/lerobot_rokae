#!/usr/bin/env python

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RokaePolicyProfile:
    """Policy-specific adaptation profile used by the Rokae processor hub.

    Contract-specific prefixes are defined in `contracts.py`.
    Profiles should only declare contract choice and optional policy-specific overrides.
    """

    policy_type: str
    preferred_contract: str
    observation_keep_keys: tuple[str, ...] = field(default_factory=tuple)  # exact keys
    action_keep_keys: tuple[str, ...] = field(default_factory=tuple)  # exact keys
    observation_extra_prefixes: tuple[str, ...] = field(default_factory=tuple)
    action_extra_prefixes: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


ROKAE_POLICY_PROFILES: dict[str, RokaePolicyProfile] = {
    "act": RokaePolicyProfile(
        policy_type="act",
        preferred_contract="joint",
        observation_keep_keys=(),
        action_keep_keys=(),
        observation_extra_prefixes=(),
        action_extra_prefixes=(),
        notes="Prefer joint-space states/actions for ACT.",
    ),
    "diffusion": RokaePolicyProfile(
        policy_type="diffusion",
        preferred_contract="joint",
        observation_keep_keys=(),
        action_keep_keys=(),
        observation_extra_prefixes=(),
        action_extra_prefixes=(),
        notes="Prefer joint-space states/actions for Diffusion.",
    ),
    "smolvla": RokaePolicyProfile(
        policy_type="smolvla",
        preferred_contract="ee",
        observation_keep_keys=(),
        action_keep_keys=(),
        observation_extra_prefixes=(),
        action_extra_prefixes=(),
        notes="Prefer end-effector/cartesian channels for SmolVLA.",
    ),
    "xvla": RokaePolicyProfile(
        policy_type="xvla",
        preferred_contract="ee",
        observation_keep_keys=(),
        action_keep_keys=(),
        observation_extra_prefixes=(),
        action_extra_prefixes=(),
        notes="Prefer end-effector/cartesian channels for XVLA.",
    ),
}


def get_policy_profile(policy_type: str) -> RokaePolicyProfile:
    if policy_type not in ROKAE_POLICY_PROFILES:
        raise ValueError(
            f"Policy '{policy_type}' is not registered in Rokae policy profiles. "
            f"Known policies: {tuple(ROKAE_POLICY_PROFILES.keys())}"
        )
    return ROKAE_POLICY_PROFILES[policy_type]
