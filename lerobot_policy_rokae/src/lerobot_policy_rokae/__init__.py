"""Rokae policy plugin package for LeRobot."""

try:
    from . import processor_hub
    from .configuration_rokae_custom import RokaeCustomConfig
    from .modeling_rokae_custom import RokaeCustomPolicy
    from .processor_rokae_custom import make_rokae_custom_pre_post_processors
except ModuleNotFoundError:
    # Keep policy package importable if optional policy deps are missing.
    processor_hub = None
    RokaeCustomConfig = None
    RokaeCustomPolicy = None
    make_rokae_custom_pre_post_processors = None

__all__ = [
    "RokaeCustomConfig",
    "RokaeCustomPolicy",
    "make_rokae_custom_pre_post_processors",
    "processor_hub",
]
