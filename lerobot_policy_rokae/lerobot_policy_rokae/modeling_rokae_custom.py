#!/usr/bin/env python

from typing import Any

from torch import Tensor, nn

from .configuration_rokae_custom import RokaeCustomConfig
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy


class RokaeCustomPolicy(PreTrainedPolicy):
    """Skeleton policy reserved for future Rokae-specific model implementation."""

    config_class = RokaeCustomConfig
    name = "rokae_custom"

    def __init__(self, config: RokaeCustomConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config)
        self.dataset_stats = dataset_stats
        # Minimal placeholder module to keep the class structurally valid.
        self._placeholder = nn.Identity()

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def reset(self) -> None:
        # TODO: clear recurrent/chunk caches once implemented.
        return None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "RokaeCustomPolicy is a scaffold. Implement training forward logic before use."
        )

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise NotImplementedError(
            "RokaeCustomPolicy is a scaffold. Implement chunk prediction before use."
        )

    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        raise NotImplementedError(
            "RokaeCustomPolicy is a scaffold. Implement online action selection before use."
        )
