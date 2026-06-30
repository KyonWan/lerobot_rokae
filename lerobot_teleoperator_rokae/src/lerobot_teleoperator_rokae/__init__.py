# Register teleoperator config subclasses (processors load via pipeline imports).
from .devices.bi_spacemouse.config_bi_spacemouse import BiSpacemouseConfig  # noqa: F401
from .devices.pico.config_pico import PicoConfig  # noqa: F401
from .devices.pico_single.config_pico_single import PicoSingleConfig  # noqa: F401
from .devices.spacemouse.config_spacemouse import SpacemouseConfig  # noqa: F401
