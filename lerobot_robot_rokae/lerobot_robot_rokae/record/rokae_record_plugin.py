from __future__ import annotations

import logging
from typing import Optional, Tuple

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
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.teleop_common.config import (
    arm_config,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.spacemouse.pipeline import (
    build_arm_pipeline,
    find_arm_pipeline,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.pico.pipeline import (
    build_pico_arm_pipeline,
)


def _callback_mode_value(cb) -> Optional[str]:
    """Helper to normalize enum / string callback_mode."""
    if cb is None:
        return None
    return getattr(cb, "value", cb)


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
    left_joint_num = getattr(cfg.robot, "left_joint_num", 7)
    right_joint_num = getattr(cfg.robot, "right_joint_num", 7)
    left_cb = getattr(cfg.robot, "left_callback_mode", None)
    right_cb = getattr(cfg.robot, "right_callback_mode", None)
    left_mode = _callback_mode_value(left_cb)
    right_mode = _callback_mode_value(right_cb)
    
    # 检查左右臂的 callback_mode 是否相同（通常应该相同）
    if left_mode != right_mode:
        raise ValueError(
            f"Left and right arms must have the same callback_mode. "
            f"Got left={left_mode}, right={right_mode}"
        )
    
    cb_mode = left_mode  # 使用相同的 mode
    
    # 几何参数来自 BiRokaeRobot 内部的左右 RokaeRobot（它们在构造时已通过 ZMQ 读取）
    left_arm = getattr(robot, "left_arm", None)
    right_arm = getattr(robot, "right_arm", None)
    left_tool_end_pos = getattr(left_arm, "tool_end_pos", None)
    left_tool_ref_pos = getattr(left_arm, "tool_ref_pos", None)
    left_base_frame_in_world = getattr(left_arm, "base_frame_in_world", None)
    right_tool_end_pos = getattr(right_arm, "tool_end_pos", None)
    right_tool_ref_pos = getattr(right_arm, "tool_ref_pos", None)
    right_base_frame_in_world = getattr(right_arm, "base_frame_in_world", None)
    arm_specs = bimanual_arm_specs(
        left_arm, right_arm, left_joint_num, right_joint_num
    )
    cart_arms = cart_pos_ref_to_base_arms(arm_specs)
    select_arms = select_action_arms(arm_specs)
    
    # 运动学参数
    left_robot_ip = getattr(cfg.robot, "left_robot_ip", "")
    right_robot_ip = getattr(cfg.robot, "right_robot_ip", "")

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
    joint_num = getattr(cfg.robot, "joint_num")
    cb_mode = _callback_mode_value(getattr(cfg.robot, "callback_mode", None))

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
                    getattr(robot, "tool_end_pos", None),
                    getattr(robot, "tool_ref_pos", None),
                    getattr(robot, "base_frame_in_world", None),
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=getattr(cfg.robot, "robot_ip", ""),
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
                    getattr(robot, "tool_end_pos", None),
                    getattr(robot, "tool_ref_pos", None),
                    getattr(robot, "base_frame_in_world", None),
                    trans_max_vel,
                    rot_max_vel,
                    robot_ip=getattr(cfg.robot, "robot_ip", ""),
                )
            ],
            control_period=1.0 / cfg.dataset.fps,
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


