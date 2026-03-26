"""Pico 单臂包：避免在 import config 时加载 pico_single 实现（meshcat 等可选依赖）。"""


def __getattr__(name: str):
    if name == "PicoSingle":
        from .pico_single import PicoSingle

        return PicoSingle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
