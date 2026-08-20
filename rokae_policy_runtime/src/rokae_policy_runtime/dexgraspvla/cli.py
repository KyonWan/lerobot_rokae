#!/usr/bin/env python3
"""CLI entrypoint for Rokae 双臂 <-> DexGraspVLA bridge."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def _default_log_file() -> Path:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"rokae_dexgraspvla_{ts}.log"


def _default_latency_log_file() -> Path:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"rokae_dexgraspvla_latency_{ts}.csv"


def _configure_logging(level: str, log_file: str | None) -> Path:
    lvl = getattr(logging, level.upper(), logging.INFO)
    log_path = Path(log_file).expanduser().resolve() if log_file else _default_log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info("Logging initialized: %s", log_path)
    return log_path


def _import_prompt_presets():
    try:
        from planner.task_prompts import USER_PROMPT_PRESETS, resolve_user_prompt

        return USER_PROMPT_PRESETS, resolve_user_prompt
    except ImportError:
        dexgrasp_root = Path(__file__).resolve().parents[4] / "DexGraspVLA"
        if dexgrasp_root.is_dir():
            sys.path.insert(0, str(dexgrasp_root))
            from planner.task_prompts import USER_PROMPT_PRESETS, resolve_user_prompt

            return USER_PROMPT_PRESETS, resolve_user_prompt
        raise ImportError(
            "Cannot import planner.task_prompts. Add DexGraspVLA to PYTHONPATH or use --task_prompt."
        ) from None


def _prompt_preset_choices() -> list[str]:
    try:
        presets, _ = _import_prompt_presets()
        return sorted(presets.keys())
    except ImportError:
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rokae 双臂机器人（三相机）<-> DexGraspVLA Controller 直接推理桥接",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dexgrasp_host", default="127.0.0.1", help="DexGraspVLA WebSocket policy server 主机")
    parser.add_argument("--dexgrasp_port", type=int, default=8008, help="DexGraspVLA WebSocket policy server 端口")
    presets = _prompt_preset_choices()
    parser.add_argument(
        "--prompt_preset",
        choices=presets or None,
        default=None,
        help=(
            "Built-in task prompt from DexGraspVLA planner/task_prompts.py "
            "(overrides --task_prompt). Use mask_eval_dual_object_* for Level B 双物体采集."
        ),
    )
    parser.add_argument(
        "--task_prompt",
        default="grasp the target object",
        help="传给 DexGraspVLA 的自定义任务描述（未指定 --prompt_preset 时使用）",
    )
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="autonomous 发送真机动作；test 只打印动作",
    )
    parser.add_argument("--max_steps", type=int, default=10000, help="最大控制步数")
    parser.add_argument("--control_freq", type=int, default=30, help="控制频率 Hz")
    parser.add_argument(
        "--action_stride",
        type=int,
        default=6,
        help="每次 DexGraspVLA 推理后执行 chunk 前多少步；官方 inference.py 默认执行前 6 步",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=8,
        choices=[8, 16],
        help="策略输出动作维度；左臂新模型用 8，老双臂模型用 16",
    )
    parser.add_argument(
        "--control_arm",
        default="left",
        choices=["left", "right", "dual"],
        help="8 维动作控制哪只手臂；新左臂模型用 left",
    )

    parser.add_argument("--left_zmq_port", type=int, default=5555, help="左臂 ZMQ 端口")
    parser.add_argument("--right_zmq_port", type=int, default=5556, help="右臂 ZMQ 端口")
    parser.add_argument("--left_zmq_address", default=None, help="左臂完整 ZMQ 地址，优先于端口")
    parser.add_argument("--right_zmq_address", default=None, help="右臂完整 ZMQ 地址，优先于端口")

    parser.add_argument("--cam_high_serial", required=True, help="顶部 external（裁下半幅后 resize 640x480）")
    parser.add_argument("--cam_left_wrist_serial", required=True, help="左腕 RealSense（原分辨率，不 resize）")
    parser.add_argument(
        "--cam_right_wrist_serial",
        default=None,
        help="右腕 RealSense（16 维双臂模型需要；8 维左臂模型可不传）",
    )
    parser.add_argument("--no_reset_on_start", action="store_true", help="启动 episode 时不调用 robot.reset_position()")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument("--log_file", default=None, help="日志文件路径；为空则写到 ./logs/rokae_dexgraspvla_*.log")
    parser.add_argument(
        "--latency_log_file",
        default=None,
        help="阶段耗时 CSV 路径；为空则写到 ./logs/rokae_dexgraspvla_latency_*.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt_preset:
        _, resolve_user_prompt = _import_prompt_presets()
        task_prompt = resolve_user_prompt(args.prompt_preset)
    else:
        task_prompt = args.task_prompt
    log_path = _configure_logging(args.log_level, args.log_file)
    latency_log_path = (
        Path(args.latency_log_file).expanduser().resolve()
        if args.latency_log_file
        else _default_latency_log_file()
    )
    logging.info(
        "rokae-dexgraspvla 启动 | mode=%s ws=%s:%s prompt=%r log=%s latency_csv=%s",
        args.mode,
        args.dexgrasp_host,
        args.dexgrasp_port,
        task_prompt,
        log_path,
        latency_log_path,
    )

    from .bridge import DexGraspVLABridge

    bridge = DexGraspVLABridge(
        dexgrasp_host=args.dexgrasp_host,
        dexgrasp_port=args.dexgrasp_port,
        task_prompt=task_prompt,
        mode=args.mode,
        max_steps=args.max_steps,
        control_frequency=args.control_freq,
        action_stride=args.action_stride,
        action_dim=args.action_dim,
        control_arm=args.control_arm,
        left_zmq_port=args.left_zmq_port,
        right_zmq_port=args.right_zmq_port,
        left_zmq_address=args.left_zmq_address,
        right_zmq_address=args.right_zmq_address,
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        reset_on_start=not args.no_reset_on_start,
        latency_log_file=latency_log_path,
    )
    try:
        bridge.run_episode()
    finally:
        bridge.cleanup()


if __name__ == "__main__":
    main()
