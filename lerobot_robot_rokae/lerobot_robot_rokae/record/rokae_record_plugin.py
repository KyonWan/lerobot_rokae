from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import Robot

from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.rokae_processor import (
    CartPosBaseToRefObservationProcessor,
    CartPosRefToBaseProcessor,
    RokaeCameraCropObservationProcessor,
    SelectActionByCallbackMode,
    bimanual_arm_specs,
    cart_pos_ref_to_base_arms,
    select_action_arms,
    single_arm_spec,
)
from lerobot_robot_rokae.lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import (
    infer_callback_mode,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common.config import (
    ArmConfig,
    CoreArmRuntime,
    arm_config,
    ensure_cart_pos_flan_in_base,
    write_arm_joints,
)
from rokae_python_wrapper.rokae_kinematics.action_fields import obs_joint_vector
from rokae_python_wrapper.rokae_kinematics.transform_utils import pose_to_transform
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.spacemouse.pipeline import (
    build_arm_pipeline,
    find_arm_pipeline,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.pico.pipeline import (
    build_pico_arm_pipeline,
)


POSTURE_RESET_SETTLE_STEPS = 20
POSTURE_RESET_SETTLE_CART_SPEED = 0.004
POSTURE_RESET_SETTLE_COST_SCALE = 0.01


def _callback_mode_from_control_mode(control_mode) -> str:
    """Derive callback_mode from control_mode enum/string."""
    cm = getattr(control_mode, "value", control_mode)
    return infer_callback_mode(cm).value


def _ik_robot_ip(client, joint_num: int) -> str:
    if joint_num == 6:
        return client.get_robot_info().get("robot_ip", "")
    return ""


def _warn_if_rokae_vel_limits_exceeded(
    trans_max_vel: float,
    rot_max_vel: float,
    processor_name: str,
) -> None:
    """
    在采集时检查上层 processor 设定的最大笛卡尔速度是否比底层 RokaeServer 更宽松。
    若超出，则打印 warning，但不中断录制。
    """
    from rokae_python_wrapper.rokae_server import RokaeServer  # type: ignore

    if trans_max_vel > RokaeServer.MAX_LINEAR_VEL or rot_max_vel > RokaeServer.MAX_ANGULAR_VEL:
        logger = logging.getLogger(__name__)
        logger.warning(
            "%s velocity limits (trans_max_vel=%.3f m/s, rot_max_vel=%.3f rad/s) "
            "are more permissive than RokaeServer limits (MAX_LINEAR_VEL=%.3f m/s, MAX_ANGULAR_VEL=%.3f rad/s). "
            "During recording, low-level control will saturate velocities; consider lowering processor limits.",
            processor_name,
            float(trans_max_vel),
            float(rot_max_vel),
            float(RokaeServer.MAX_LINEAR_VEL),
            float(RokaeServer.MAX_ANGULAR_VEL),
        )


def _make_external_lower_half_crop_params(robot: Robot) -> dict[str, tuple[int, int, int, int]]:
    """
    Build crop params for the head camera (`external`) to keep only the lower half.
    Returns empty dict when `external` camera is unavailable.
    """
    cameras_cfg = getattr(getattr(robot, "cfg", None), "cameras", {}) or {}
    external_cfg = cameras_cfg.get("external")
    if external_cfg is None:
        return {}

    height = getattr(external_cfg, "height", None)
    width = getattr(external_cfg, "width", None)
    if height is None or width is None:
        # Fallback to feature shape when camera config doesn't expose width/height.
        obs_features = getattr(robot, "observation_features", {}) or {}
        shape = obs_features.get("external")
        if not (isinstance(shape, tuple) and len(shape) == 3):
            return {}
        height, width, _ = shape

    top = int(height) // 2
    left = 0
    crop_h = int(height) - top
    crop_w = int(width)
    return {"external": (top, left, crop_h, crop_w)}


def _make_external_resize_params() -> dict[str, tuple[int, int]]:
    """Resize head camera (`external`) output to fixed 640x480 (height, width)."""
    return {"external": (480, 640)}


def _make_bimanual_pipelines(
    cfg,
    robot: Robot,
) -> Optional[Tuple[RobotProcessorPipeline, RobotProcessorPipeline, RobotProcessorPipeline]]:
    """
    为双臂 rokae 机器人创建 processor pipelines。
    支持双 spacemouse 和 pico。
    
    返回 (teleop_action_processor, robot_action_processor, robot_observation_processor)，如果不支持则返回 None。
    """
    left_joint_num = robot.left_arm.joint_num
    right_joint_num = robot.right_arm.joint_num
    left_cm = cfg.robot.left_control_mode
    right_cm = cfg.robot.right_control_mode
    left_mode = _callback_mode_from_control_mode(left_cm)
    right_mode = _callback_mode_from_control_mode(right_cm)
    
    # 检查左右臂的 callback_mode 是否相同（通常应该相同）
    if left_mode != right_mode:
        raise ValueError(
            f"Left and right arms must have the same callback_mode. "
            f"Got left={left_mode}, right={right_mode}"
        )
    
    cb_mode = left_mode  # 使用相同的 mode
    
    left_arm = robot.left_arm
    right_arm = robot.right_arm
    left_tool_end_pos = left_arm.tool_end_pos
    left_tool_ref_pos = left_arm.tool_ref_pos
    left_base_frame_in_world = left_arm.base_frame_in_world
    right_tool_end_pos = right_arm.tool_end_pos
    right_tool_ref_pos = right_arm.tool_ref_pos
    right_base_frame_in_world = right_arm.base_frame_in_world
    arm_specs = bimanual_arm_specs(
        left_arm, right_arm, left_joint_num, right_joint_num
    )
    cart_arms = cart_pos_ref_to_base_arms(arm_specs)
    select_arms = select_action_arms(arm_specs)
    
    # 运动学参数
    left_robot_ip = _ik_robot_ip(robot.left_arm.client, left_joint_num)
    right_robot_ip = _ik_robot_ip(robot.right_arm.client, right_joint_num)

    if cfg.teleop.type == "bi_spacemouse":
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        _left_type = robot.left_arm.client.get_robot_info()["type"]
        _right_type = robot.right_arm.client.get_robot_info()["type"]
        _arm_pipeline = build_arm_pipeline(
            arms=[
                arm_config(
                    "left_",
                    left_joint_num,
                    _left_type,
                    left_tool_end_pos,
                    left_tool_ref_pos,
                    left_base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=left_robot_ip,
                ),
                arm_config(
                    "right_",
                    right_joint_num,
                    _right_type,
                    right_tool_end_pos,
                    right_tool_ref_pos,
                    right_base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=right_robot_ip,
                ),
            ],
            control_period=1.0 / cfg.dataset.fps,
        )
        teleop_action_processor_steps = _arm_pipeline.steps
    elif cfg.teleop.type == "pico":
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        _left_type = robot.left_arm.client.get_robot_info()["type"]
        _right_type = robot.right_arm.client.get_robot_info()["type"]

        _pico_pipeline = build_pico_arm_pipeline(
            arms=[
                arm_config(
                    "left_",
                    left_joint_num,
                    _left_type,
                    left_tool_end_pos,
                    left_tool_ref_pos,
                    left_base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=left_robot_ip,
                ),
                arm_config(
                    "right_",
                    right_joint_num,
                    _right_type,
                    right_tool_end_pos,
                    right_tool_ref_pos,
                    right_base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=right_robot_ip,
                ),
            ],
            control_period=1.0 / cfg.dataset.fps,
            trigger_reverse=cfg.teleop.trigger_reverse,
            trigger_threshold=cfg.teleop.trigger_threshold,
            close_position=cfg.teleop.close_position,
            open_position=cfg.teleop.open_position,
        )
        teleop_action_processor_steps = _pico_pipeline.steps
    else:
        raise ValueError(f"Unsupported teleop type: {cfg.teleop.type}")

    _warn_if_rokae_vel_limits_exceeded(
        trans_max_vel=trans_max_vel,
        rot_max_vel=rot_max_vel,
        processor_name="ArmPipeline",
    )

    if cb_mode == "cart_vel":
        raise ValueError(
            "callback_mode='cart_vel' is no longer supported; use 'joint_pos' or 'cart_pos'."
        )

    # 根据 callback_mode 配置不同的 robot_action_processor_steps
    if cb_mode == "cart_pos":
        robot_action_processor_steps = [
            CartPosRefToBaseProcessor(arms=cart_arms),
            SelectActionByCallbackMode(callback_mode="cart_pos", arms=select_arms),
        ]
    elif cb_mode == "joint_pos":
        robot_action_processor_steps = [
            SelectActionByCallbackMode(callback_mode="joint_pos", arms=select_arms),
        ]
    else:
        # 非 rokae 的 callback_mode，使用默认 processors
        return None

    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=teleop_action_processor_steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=robot_action_processor_steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor_steps = [
        CartPosBaseToRefObservationProcessor(arms=cart_arms),
    ]
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=robot_observation_processor_steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    return teleop_action_processor, robot_action_processor, robot_observation_processor


def _make_single_arm_pipelines(
    cfg,
    robot: Robot,
) -> Optional[Tuple[RobotProcessorPipeline, RobotProcessorPipeline, RobotProcessorPipeline]]:
    """
    为单臂 rokae 机器人创建 processor pipelines。
    
    返回 (teleop_action_processor, robot_action_processor, robot_observation_processor)，如果不支持则返回 None。
    """
    joint_num = robot.joint_num
    cb_mode = _callback_mode_from_control_mode(cfg.robot.control_mode)
    robot_ip = _ik_robot_ip(robot.client, joint_num)

    if cfg.teleop.type == "spacemouse":
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        _robot_type = robot.client.get_robot_info()["type"]
        _arm_pipeline = build_arm_pipeline(
            arms=[
                arm_config(
                    "",
                    joint_num,
                    _robot_type,
                    robot.tool_end_pos,
                    robot.tool_ref_pos,
                    robot.base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=robot_ip,
                )
            ],
            control_period=1.0 / cfg.dataset.fps,
        )
        teleop_action_processor_steps = _arm_pipeline.steps
    elif cfg.teleop.type == "pico_single":
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        _robot_type = robot.client.get_robot_info()["type"]
        _pico_pipeline = build_pico_arm_pipeline(
            arms=[
                arm_config(
                    "",
                    joint_num,
                    _robot_type,
                    robot.tool_end_pos,
                    robot.tool_ref_pos,
                    robot.base_frame_in_world,
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=robot_ip,
                )
            ],
            control_period=1.0 / cfg.dataset.fps,
            trigger_reverse=cfg.teleop.trigger_reverse,
            trigger_threshold=cfg.teleop.trigger_threshold,
            close_position=cfg.teleop.close_position,
            open_position=cfg.teleop.open_position,
        )
        teleop_action_processor_steps = _pico_pipeline.steps
    else:
        raise ValueError(f"Unsupported single-arm teleop type: {cfg.teleop.type}")

    _warn_if_rokae_vel_limits_exceeded(
        trans_max_vel=trans_max_vel,
        rot_max_vel=rot_max_vel,
        processor_name="ArmPipeline",
    )

    arm_specs = single_arm_spec(robot, joint_num)
    cart_arms = cart_pos_ref_to_base_arms(arm_specs)
    select_arms = select_action_arms(arm_specs)

    if cb_mode == "cart_vel":
        raise ValueError(
            "callback_mode='cart_vel' is no longer supported; use 'joint_pos' or 'cart_pos'."
        )

    # 根据 callback_mode 配置不同的 robot_action_processor_steps
    if cb_mode == "cart_pos":
        robot_action_processor_steps = [
            CartPosRefToBaseProcessor(arms=cart_arms),
            SelectActionByCallbackMode(callback_mode="cart_pos", arms=select_arms),
        ]
    elif cb_mode == "joint_pos":
        robot_action_processor_steps = [
            SelectActionByCallbackMode(callback_mode="joint_pos", arms=select_arms),
        ]
    else:
        # 非 rokae 的 callback_mode，使用默认 processors
        return None

    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=teleop_action_processor_steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=robot_action_processor_steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor_steps = [
        CartPosBaseToRefObservationProcessor(arms=cart_arms),
    ]
    crop_params = _make_external_lower_half_crop_params(robot)
    if crop_params:
        robot_observation_processor_steps.append(
            RokaeCameraCropObservationProcessor(
                camera_crop_params=crop_params,
                camera_resize_params=_make_external_resize_params(),
            )
        )
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=robot_observation_processor_steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_action_processor, robot_action_processor, robot_observation_processor

def maybe_make_rokae_pipelines(
    cfg,
    robot: Robot,
    teleop,
    default_teleop_action_processor: RobotProcessorPipeline,
    default_robot_action_processor: RobotProcessorPipeline,
    default_robot_observation_processor: RobotProcessorPipeline,
) -> Optional[Tuple[RobotProcessorPipeline, RobotProcessorPipeline, RobotProcessorPipeline]]:
    """
    根据当前 cfg/robot/teleop，按 rokae 相关规则覆盖 teleop / robot action / robot observation processor。

    如果不是 rokae 相关场景，返回 None，调用方应继续使用默认 processor。
    """
    # 双臂：bi_spacemouse + bi_rokae_robot
    is_bimanual = (cfg.teleop is not None and (cfg.teleop.type == "bi_spacemouse" or cfg.teleop.type == "pico")) or (
        cfg.robot.type == "bi_rokae_robot"
    )
    if is_bimanual:
        return _make_bimanual_pipelines(cfg, robot)

    # 单臂：rokae_robot
    if cfg.robot.type == "rokae_robot":
        return _make_single_arm_pipelines(cfg, robot)

    return None


def maybe_reset_rokae_processors(
    robot: Robot,
    teleop_action_processor: RobotProcessorPipeline,
) -> bool:
    """
    处理 rokae 相关 Processor 的夹爪 reset 逻辑。

    如果识别并完成了 reset，返回 True；否则返回 False，调用方可以继续执行通用逻辑。
    """
    arm_pipeline = find_arm_pipeline(teleop_action_processor.steps)
    if arm_pipeline is not None:
        if arm_pipeline.is_bimanual:
            left_gripper_state = arm_pipeline.initial_left_gripper_state
            right_gripper_state = arm_pipeline.initial_right_gripper_state
            if hasattr(robot, "set_gripper_states"):
                robot.set_gripper_states(left_gripper_state, right_gripper_state)
            arm_pipeline.reset_all(
                left_gripper_state=left_gripper_state,
                right_gripper_state=right_gripper_state,
            )
        else:
            gripper_state = arm_pipeline.initial_gripper_state
            if hasattr(robot, "set_gripper_state"):
                robot.set_gripper_state(gripper_state)
            arm_pipeline.reset_all(gripper_state=gripper_state)
        return True

    return False


def reset_gripper_states(
    robot: Robot,
    teleop_action_processor: RobotProcessorPipeline,
) -> None:
    """
    Reset gripper states to initial values before episode starts.
    Sets physical gripper to initial state and synchronizes processor state.
    Supports both single arm and bimanual systems.
    
    Args:
        robot: Robot instance
        teleop_action_processor: Teleoperator action processor pipeline
    """
    # 先让 rokae 插件有机会处理（包括单臂 / 双臂 IK、Generate* 等）。
    # 非 rokae 机器人目前不在这里做额外处理，保持与官方脚本更接近，
    # 后续如官方增加通用 reset 逻辑，可以直接同步覆盖本函数。
    handled = maybe_reset_rokae_processors(robot, teleop_action_processor)
    if handled:
        return


def _callback_mode_value(obj) -> str:
    callback_mode = getattr(obj, "callback_mode", None)
    return str(getattr(callback_mode, "value", callback_mode))


def _supports_joint_pos_reset_settle(robot: Robot, arm_pipeline) -> bool:
    if arm_pipeline.is_bimanual:
        left_arm = getattr(robot, "left_arm", None)
        right_arm = getattr(robot, "right_arm", None)
        return (
            _callback_mode_value(left_arm) == "joint_pos"
            and _callback_mode_value(right_arm) == "joint_pos"
        )
    return _callback_mode_value(robot) == "joint_pos"


def _write_reset_settle_gripper_action(action: dict, arm_pipeline) -> None:
    if arm_pipeline.is_bimanual:
        action["left_gripper_pos"] = float(arm_pipeline.initial_left_gripper_state)
        action["right_gripper_pos"] = float(arm_pipeline.initial_right_gripper_state)
    else:
        action["gripper_pos"] = float(arm_pipeline.initial_gripper_state)


def _solve_reset_settle_arm(
    cfg: ArmConfig,
    rt: CoreArmRuntime,
    obs: dict,
    dt: float,
) -> list[float] | None:
    if cfg.joint_num != 7 or rt.ik_solver is None:
        return None

    ensure_cart_pos_flan_in_base(cfg, rt, obs)
    assert rt.cart_pos_flan_in_base is not None
    q_current = obs_joint_vector(obs, cfg.joint_num, cfg.key_prefix)
    settle_cart_vel = np.zeros(6, dtype=np.float64)
    settle_cart_vel[0] = POSTURE_RESET_SETTLE_CART_SPEED
    q_solution, ik_ok = rt.ik_solver.solve(
        pose_to_transform(rt.cart_pos_flan_in_base),
        q_current,
        dt,
        cart_vel=settle_cart_vel,
        posture_active=True,
        posture_cost_scale=POSTURE_RESET_SETTLE_COST_SCALE,
    )
    return q_solution if ik_ok else None


def settle_rokae_posture_before_record(
    robot: Robot,
    teleop_action_processor: RobotProcessorPipeline,
) -> None:
    arm_pipeline = find_arm_pipeline(teleop_action_processor.steps)
    if arm_pipeline is None:
        return
    if not _supports_joint_pos_reset_settle(robot, arm_pipeline):
        return
    if not any(cfg.joint_num == 7 for cfg in arm_pipeline.arms):
        return
    if not hasattr(robot, "get_observation") or not hasattr(robot, "send_action"):
        return

    logger = logging.getLogger(__name__)
    dt = float(arm_pipeline.control_period)
    for step_idx in range(POSTURE_RESET_SETTLE_STEPS):
        obs = robot.get_observation()
        action: dict = {}
        for cfg, rt in zip(arm_pipeline.arms, arm_pipeline.runtimes):
            q_solution = _solve_reset_settle_arm(cfg, rt, obs, dt)
            if q_solution is None:
                q_solution = obs_joint_vector(obs, cfg.joint_num, cfg.key_prefix).tolist()
            write_arm_joints(action, cfg.key_prefix, q_solution, cfg.joint_num)
        _write_reset_settle_gripper_action(action, arm_pipeline)
        try:
            robot.send_action(action)
        except Exception as exc:
            logger.warning("Posture reset settle stopped at step %d: %s", step_idx, exc)
            return
        time.sleep(max(dt, 0.0))


def reset_robot_and_grippers(
    robot: Robot,
    teleop,
    teleop_action_processor: RobotProcessorPipeline,
) -> None:
    """
    Convenience helper: reset robot to drag pose (if supported) and synchronize gripper state
    for (bi_)spacemouse teleoperation.
    """
    import logging
    
    if hasattr(robot, "reset_position"):
        robot.reset_position()
        logging.info("Robot reset to drag position")

    # Reset teleop internal accumulation buffers; otherwise the next control frame
    # may replay previous episode deltas and pull the robot away from drag pose.
    if teleop is not None and hasattr(teleop, "reset_for_new_episode"):
        teleop.reset_for_new_episode()

    # Reset gripper states for supported teleop systems
    if teleop is not None and getattr(teleop, "name", None) in ("spacemouse", "bi_spacemouse", "pico_single", "pico"):
        reset_gripper_states(robot, teleop_action_processor)
        settle_rokae_posture_before_record(robot, teleop_action_processor)
