"""仅在需要实例化时再加载 bi_rokae_robot（经 rokae_robot 依赖 rokae_python_wrapper）。"""


def __getattr__(name: str):
    if name == "BiRokaeRobot":
        from .bi_rokae_robot import BiRokaeRobot

        return BiRokaeRobot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
