#!/usr/bin/env python3
"""CLI entrypoint for Rokae <-> OpenPI bridge."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rokae 单臂机器人 <-> OpenPI 策略服务器桥接",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_host", default="localhost", help="策略服务器主机地址")
    parser.add_argument("--policy_port", type=int, default=8000, help="策略服务器端口")
    parser.add_argument(
        "--bridge",
        choices=["async_bridge", "bridge"],
        default="async_bridge",
        help="选择桥接实现: async_bridge（异步推理/RTC）或 bridge（同步推理）",
    )
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="运行模式: autonomous（真实执行）或 test（仅打印不运动）",
    )
    parser.add_argument("--task_prompt", default="move the arm to the left", help="传给策略的任务描述")
    parser.add_argument("--max_steps", type=int, default=10000, help="每次 Episode 的最大步数")
    parser.add_argument("--control_freq", type=int, default=30, help="控制频率 (Hz)")
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=50,
        help="策略返回的动作 chunk 长度（仅 async_bridge 使用）",
    )
    parser.add_argument(
        "--rate_of_inference",
        type=int,
        default=None,
        help="每执行多少步发起下一次 OpenPI 推理；默认读取服务器 metadata 的 rtc_rate_of_inference（仅 async_bridge 使用）",
    )
    parser.add_argument(
        "--rtc_delay",
        type=int,
        default=None,
        help="RTC 最大/初始延迟步数；默认读取服务器 metadata 的 max_delay，设为 0 可禁用机器人端 RTC 切换（仅 async_bridge 使用）",
    )
    parser.add_argument(
        "--disable_dynamic_delay",
        action="store_true",
        help="关闭动态延迟估计，始终使用 --rtc_delay 或服务器 metadata 的 max_delay（仅 async_bridge 使用）",
    )
    parser.add_argument(
        "--delay_window",
        type=int,
        default=8,
        help="动态延迟估计使用的最近推理耗时窗口（仅 async_bridge 使用）",
    )
    parser.add_argument(
        "--delay_safety_steps",
        type=int,
        default=1,
        help="动态延迟估计额外增加的安全步数（仅 async_bridge 使用）",
    )
    parser.add_argument("--zmq_port", type=int, default=5555, help="ZMQ 服务器端口（本地通信）")
    parser.add_argument("--zmq_address", type=str, default=None, help="ZMQ 完整地址（优先级高于 zmq_port）")
    parser.add_argument("--cam_high_serial", type=str, default=None, help="顶部 RealSense 相机序列号")
    parser.add_argument("--cam_wrist_serial", type=str, default=None, help="腕部 RealSense 相机序列号")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.bridge == "async_bridge":
        from .async_bridge import OpenPIPolicyBridge
    else:
        from .bridge import OpenPIPolicyBridge

    bridge_kwargs = {
        "policy_server_host": args.policy_host,
        "policy_server_port": args.policy_port,
        "control_frequency": args.control_freq,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "zmq_port": args.zmq_port,
        "zmq_address": args.zmq_address,
        "cam_high_serial": args.cam_high_serial,
        "cam_wrist_serial": args.cam_wrist_serial,
    }
    if args.bridge == "async_bridge":
        bridge_kwargs.update(
            {
                "action_chunk_size": args.action_chunk_size,
                "rate_of_inference": args.rate_of_inference,
                "rtc_delay": args.rtc_delay,
                "dynamic_delay": not args.disable_dynamic_delay,
                "delay_window": args.delay_window,
                "delay_safety_steps": args.delay_safety_steps,
            }
        )

    bridge = OpenPIPolicyBridge(**bridge_kwargs)

    try:
        bridge.run_episode(task_prompt=args.task_prompt)
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
