"""在 pyrealsense2 导入前预热 xCore SDK（6 轴 xMateRobot 构造）。"""

from __future__ import annotations

import platform
import sys
from typing import Any


def _cli_flag(name: str, default: Any = None) -> Any:
    """轻量解析 CLI 标志，支持 ``--flag=value`` 与 ``--flag value``。"""
    prefix_eq = f"{name}="
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(prefix_eq):
            return arg[len(prefix_eq) :]
    return default


def _port_to_zmq_address(port: int) -> str:
    if platform.system() == "Windows":
        return f"tcp://127.0.0.1:{port}"
    return f"ipc:///tmp/rokae_server_{port}"


def _single_arm_zmq_address() -> str:
    zmq_address = _cli_flag("--robot.zmq_address")
    if zmq_address:
        return str(zmq_address)
    zmq_port = int(_cli_flag("--robot.zmq_port", 5555))
    return _port_to_zmq_address(zmq_port)


def _bimanual_zmq_addresses() -> list[str]:
    addresses: list[str] = []
    for addr_flag in ("--robot.left_zmq_address", "--robot.right_zmq_address"):
        addr = _cli_flag(addr_flag)
        if addr:
            addresses.append(str(addr))
    if addresses:
        return addresses

    left_port = int(_cli_flag("--robot.left_zmq_port", 5555))
    right_port = int(_cli_flag("--robot.right_zmq_port", 5556))
    return [_port_to_zmq_address(left_port), _port_to_zmq_address(right_port)]


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

    rokae_sdk_api.xMateRobot(robot_ip)


def maybe_init_xcore_sdk_before_realsense() -> None:
    """若当前为 rokae 录制且存在 6 轴臂，在 RealSense 导入前调用 xMateRobot。"""
    robot_type = _cli_flag("--robot.type")
    if robot_type not in ("rokae_robot", "bi_rokae_robot"):
        return

    for zmq_address in _resolve_zmq_addresses(str(robot_type)):
        _preload_xcore_for_address(zmq_address)
