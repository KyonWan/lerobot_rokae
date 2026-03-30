#!/usr/bin/env python

from __future__ import annotations

from typing import Any

import numpy as np
from lerobot.processor import PolicyProcessorPipeline

from .processor_hub.factory import make_rokae_policy_adaptation_pipelines
from .processor_hub.steps_feature_select import RokaeFeatureSelectStep


def _slice_stats_block(
    block: dict[str, np.ndarray], indices: tuple[int, ...]
) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for k, v in block.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[-1] > int(idx.max(initial=-1)):
            out[k] = np.take(arr, idx, axis=-1)
        else:
            out[k] = arr.copy()
    return out


def apply_rokae_slices_to_metadata(meta: Any, policy_type: str) -> bool:
    """Align dataset feature specs + normalization stats with Rokae feature selection.

    `RokaeFeatureSelectStep` slices raw dataset vectors before the policy; dataset
    stats and policy `input_features` shapes must use the same index subset so
    MEAN_STD normalization and ACT linear layers match tensor dimensions.

    Returns:
        True if observation/action metadata and stats were sliced; False if no subset selection applies.
    """
    if policy_type not in {"act", "diffusion", "smolvla", "xvla"}:
        return False

    feats = meta.info["features"]
    obs_names_full = tuple(feats["observation.state"]["names"])
    act_names_full = tuple(feats["action"]["names"])

    rokae_preprocessor, _ = make_rokae_policy_adaptation_pipelines(
        policy_type=policy_type,
        observation_state_names=obs_names_full,
        action_names=act_names_full,
    )
    first = rokae_preprocessor.steps[0]
    if not isinstance(first, RokaeFeatureSelectStep):
        raise TypeError("Expected RokaeFeatureSelectStep as first Rokae preprocessor step.")
    obs_idx = first.observation_state_indices
    act_idx = first.action_indices

    if len(obs_idx) == len(obs_names_full) and len(act_idx) == len(act_names_full):
        return False

    obs_names = [obs_names_full[i] for i in obs_idx]
    act_names = [act_names_full[i] for i in act_idx]
    feats["observation.state"] = {**feats["observation.state"], "names": obs_names, "shape": [len(obs_names)]}
    feats["action"] = {**feats["action"], "names": act_names, "shape": [len(act_names)]}

    if stats := getattr(meta, "stats", None):
        if "observation.state" in stats:
            stats["observation.state"] = _slice_stats_block(stats["observation.state"], obs_idx)
        if "action" in stats:
            stats["action"] = _slice_stats_block(stats["action"], act_idx)
    return True


def compose_rokae_preprocessor_if_needed(
    *,
    policy_type: str,
    dataset_meta_info: dict[str, Any],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    logger: Any,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """Compose Rokae feature adapter before native preprocessor when applicable."""
    if policy_type not in {"act", "diffusion", "smolvla", "xvla"}:
        return preprocessor

    # Resume path can load a preprocessor already containing Rokae feature selection.
    # Avoid composing twice, otherwise the second selector can index out of range.
    if any(isinstance(step, RokaeFeatureSelectStep) for step in preprocessor.steps):
        logger.info("Detected existing RokaeFeatureSelectStep in preprocessor; skip duplicate compose.")
        return preprocessor

    features_info = dataset_meta_info["features"]
    observation_state_names = tuple(features_info["observation.state"]["names"])
    action_names = tuple(features_info["action"]["names"])

    rokae_preprocessor, _ = make_rokae_policy_adaptation_pipelines(
        policy_type=policy_type,
        observation_state_names=observation_state_names,
        action_names=action_names,
    )
    composed = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[*rokae_preprocessor.steps, *preprocessor.steps],
        # Keep canonical pipeline name so checkpoint filename remains policy_preprocessor.json.
        name=preprocessor.name,
        to_transition=preprocessor.to_transition,
        to_output=preprocessor.to_output,
    )
    first = rokae_preprocessor.steps[0]
    assert isinstance(first, RokaeFeatureSelectStep)
    logger.info(
        "Composed Rokae preprocessor for '%s' with selected obs/action dims: %d/%d (dataset names: %d/%d)",
        policy_type,
        len(first.observation_state_indices),
        len(first.action_indices),
        len(observation_state_names),
        len(action_names),
    )
    return composed
