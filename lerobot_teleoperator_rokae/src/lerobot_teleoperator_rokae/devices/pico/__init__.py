"""Pico 遥操作包：避免在 import 子模块（如 config_pico）时加载 pico.py（meshcat 等可选依赖）。"""


def __getattr__(name: str):
    if name == "Pico":
        from .pico import Pico

        return Pico
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
