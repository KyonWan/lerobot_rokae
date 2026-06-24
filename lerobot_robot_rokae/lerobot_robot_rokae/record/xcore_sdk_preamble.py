"""在 pyrealsense2 导入前预热 xCore SDK（6 轴 xMateRobot 构造）。"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cli_flag(name: str, default: Any = None) -> Any:
    """轻量解析 CLI 标志，支持 ``--flag=value`` 与 ``--flag value``。"""
    prefix_eq = f"{name}="
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(prefix_eq):
            return arg[len(prefix_eq) :]
    return default


def _record_yaml_config() -> dict[str, Any]:
    """从 ``--config_path`` 加载录制 YAML（无文件或解析失败时返回空 dict）。"""
    config_path = _cli_flag("--config_path")
    if not config_path:
        return {}

    path = Path(str(config_path))
    if not path.is_file():
        logger.warning("xCore preamble: config_path 不存在: %s", path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning(
            "xCore preamble: 未安装 PyYAML，无法从 config_path 读取 robot.type；"
            "请 pip install pyyaml 或使用 --robot.type= 命令行参数"
        )
        return {}

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _robot_section() -> dict[str, Any]:
    robot = _record_yaml_config().get("robot")
    return robot if isinstance(robot, dict) else {}


def _resolve_robot_type() -> str | None:
    cli_type = _cli_flag("--robot.type")
    if cli_type:
        return str(cli_type)
    yaml_type = _robot_section().get("type")
    return str(yaml_type) if yaml_type else None


def _yaml_int(key: str, default: int) -> int:
    value = _robot_section().get(key, default)
    return int(value)


def _port_to_zmq_address(port: int) -> str:
    if platform.system() == "Windows":
        return f"tcp://127.0.0.1:{port}"
    return f"ipc:///tmp/rokae_server_{port}"


def _single_arm_zmq_address() -> str:
    zmq_address = _cli_flag("--robot.zmq_address")
    if zmq_address:
        return str(zmq_address)
    zmq_port = _cli_flag("--robot.zmq_port")
    if zmq_port is None:
        zmq_port = _yaml_int("zmq_port", 5555)
    return _port_to_zmq_address(int(zmq_port))


def _bimanual_zmq_addresses() -> list[str]:
    addresses: list[str] = []
    for addr_flag in ("--robot.left_zmq_address", "--robot.right_zmq_address"):
        addr = _cli_flag(addr_flag)
        if addr:
            addresses.append(str(addr))
    if addresses:
        return addresses

    left_port = _cli_flag("--robot.left_zmq_port")
    right_port = _cli_flag("--robot.right_zmq_port")
    if left_port is None:
        left_port = _yaml_int("left_zmq_port", 5555)
    if right_port is None:
        right_port = _yaml_int("right_zmq_port", 5556)
    return [_port_to_zmq_address(int(left_port)), _port_to_zmq_address(int(right_port))]


def _resolve_zmq_addresses(robot_type: str) -> list[str]:
    if robot_type == "rokae_robot":
        return [_single_arm_zmq_address()]
    if robot_type == "bi_rokae_robot":
        return _bimanual_zmq_addresses()
    return []


def _preload_xcore_for_address(zmq_address: str) -> None:
    from rokae_python_wrapper.rokae_zmq_client import RokaeZmqClient, RokaeZmqClientError

    client = RokaeZmqClient(zmq_address, timeout=1.0)
    try:
        info = client.get_robot_info()
    except RokaeZmqClientError as e:
        raise RuntimeError(
            f"无法从 ZMQ ({zmq_address}) 获取 robot_info，请先启动 rokae_zmq_server: {e}"
        ) from e
    finally:
        client.close()

    if int(info.get("joint_num", 0)) != 6:
        return

    robot_ip = str(info.get("robot_ip", "")).strip()
    if not robot_ip:
        raise RuntimeError(
            f"6 轴机器人 ({zmq_address}) 的 get_robot_info 未返回 robot_ip"
        )

    from rokae_python_wrapper.rokae_sdk import xCoreSDK_python as rokae_sdk_api

    logger.info(
        "xCore SDK preamble: 6 轴臂在 RealSense 导入前预热 xMateRobot(%s) via %s",
        robot_ip,
        zmq_address,
    )
    rokae_sdk_api.xMateRobot(robot_ip)


def maybe_init_xcore_sdk_before_realsense() -> None:
    """若当前为 rokae 录制且存在 6 轴臂，在 RealSense 导入前调用 xMateRobot。"""
    robot_type = _resolve_robot_type()
    if robot_type not in ("rokae_robot", "bi_rokae_robot"):
        return

    for zmq_address in _resolve_zmq_addresses(str(robot_type)):
        _preload_xcore_for_address(zmq_address)
