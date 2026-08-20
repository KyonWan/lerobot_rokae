#!/usr/bin/env python3
"""CLI entrypoint for Rokae 双臂三相机 <-> OpenPI bridge."""

import argparse
import logging
import sys


def _configure_logging(level: str, log_file: str | None) -> None:
    """在导入 bi_bridge 之前配置 root logger，便于控制级别与文件输出。"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(lvl, int):
        lvl = logging.INFO
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=lvl, format=fmt, handlers=handlers, force=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rokae 双臂机器人（三相机）<-> OpenPI 策略服务器桥接",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_host", default="localhost", help="策略服务器主机地址")
    parser.add_argument("--policy_port", type=int, default=8000, help="策略服务器端口")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="运行模式: autonomous（真实执行）或 test（仅打印不运动）",
    )
    parser.add_argument(
        "--task_prompt",
        default="Put the two white parts into the gray box and the two blue parts into the blue box.",
        help="传给策略的任务描述",
    )
    parser.add_argument("--max_steps", type=int, default=10000, help="每次 Episode 的最大步数")
    parser.add_argument("--control_freq", type=int, default=30, help="控制频率 (Hz)")

    # ZMQ
    parser.add_argument("--left_zmq_port", type=int, default=5555, help="左臂 ZMQ 服务器端口")
    parser.add_argument("--right_zmq_port", type=int, default=5556, help="右臂 ZMQ 服务器端口")
    parser.add_argument(
        "--left_zmq_address",
        type=str,
        default=None,
        help="左臂 ZMQ 完整地址（优先级高于 left_zmq_port）",
    )
    parser.add_argument(
        "--right_zmq_address",
        type=str,
        default=None,
        help="右臂 ZMQ 完整地址（优先级高于 right_zmq_port）",
    )

    # 相机
    parser.add_argument("--cam_high_serial", type=str, default=None, help="顶部外部 RealSense 相机序列号")
    parser.add_argument("--cam_left_wrist_serial", type=str, default=None, help="左腕 RealSense 相机序列号")
    parser.add_argument("--cam_right_wrist_serial", type=str, default=None, help="右腕 RealSense 相机序列号")

    # 推理参数
    parser.add_argument("--action_chunk_size", type=int, default=50, help="每次推理返回的动作步数")
    parser.add_argument(
        "--rate_of_inference",
        type=int,
        default=30,
        help="每隔多少控制步重新请求一次策略（chunk 用完后才用新观测）；"
        "若感觉轨迹重复或闭环差，可改为 1～5 以更高频重算",
    )
    parser.add_argument(
        "--rtc_delay",
        type=int,
        default=None,
        help="手动指定 OpenPI RTC delay 步数；不指定时使用策略服务器 metadata 的 max_delay",
    )
    parser.add_argument(
        "--disable_dynamic_delay",
        action="store_true",
        help="关闭动态延迟估计，始终使用 --rtc_delay 或服务器 metadata 的 max_delay",
    )
    parser.add_argument(
        "--delay_window",
        type=int,
        default=8,
        help="动态 delay 估计使用的最近推理耗时窗口大小",
    )
    parser.add_argument(
        "--delay_safety_steps",
        type=int,
        default=1,
        help="动态 delay 估计额外增加的安全步数",
    )
    parser.add_argument(
        "--temporal_ensemble_coefficient",
        type=float,
        default=None,
        help="Temporal ensemble 系数（None 表示关闭，建议值 0.01）",
    )
    parser.add_argument(
        "--save_rgb_dir",
        type=str,
        default=None,
        help="若指定目录，则在每次推理取图时把预处理后的 RGB（224×224）存为 PNG，"
        "文件名形如 external_step000042.png",
    )
    parser.add_argument(
        "--debug_model_io",
        action="store_true",
        help="每次 infer 除默认摘要外，多打印 action chunk 的第 1～2 步向量",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO；调试时可设 DEBUG）",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="若指定路径，则同时写入该文件（UTF-8）；控制台仍会输出",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level, args.log_file)
    from .bi_bridge import BiOpenPIPolicyBridge

    bridge = BiOpenPIPolicyBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        mode=args.mode,
        max_steps=args.max_steps,
        left_zmq_port=args.left_zmq_port,
        right_zmq_port=args.right_zmq_port,
        left_zmq_address=args.left_zmq_address,
        right_zmq_address=args.right_zmq_address,
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        action_chunk_size=args.action_chunk_size,
        rate_of_inference=args.rate_of_inference,
        rtc_delay=args.rtc_delay,
        dynamic_delay=not args.disable_dynamic_delay,
        delay_window=args.delay_window,
        delay_safety_steps=args.delay_safety_steps,
        temporal_ensemble_coefficient=args.temporal_ensemble_coefficient,
        save_rgb_dir=args.save_rgb_dir,
        debug_model_io=args.debug_model_io,
    )

    try:
        bridge.run_episode(task_prompt=args.task_prompt)
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
