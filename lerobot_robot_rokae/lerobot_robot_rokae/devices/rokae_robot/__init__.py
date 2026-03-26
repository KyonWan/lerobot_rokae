"""仅在需要实例化机器人时再加载 rokae_robot（依赖 rokae_python_wrapper）。"""


def __getattr__(name: str):
    if name == "RokaeRobot":
        from .rokae_robot import RokaeRobot

        return RokaeRobot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
