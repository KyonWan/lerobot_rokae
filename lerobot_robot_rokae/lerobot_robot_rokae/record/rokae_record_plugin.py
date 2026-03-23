from __future__ import annotations

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
    CartVelRefToBaseProcessor,
    SelectActionByCallbackMode,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.spacemouse.spacemouse_processor import (
    InverseKinematicsProcessor,
)


from lerobot_robot_rokae.lerobot_robot_rokae.devices.bi_rokae_robot.bi_rokae_processor import (
    BiCartPosBaseToRefObservationProcessor,
    BiCartPosRefToBaseProcessor,
    BiCartVelRefToBaseProcessor,
    BiSelectActionByCallbackMode,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.bi_spacemouse.bi_spacemouse_processor import (
    BiInverseKinematicsProcessor,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.pico.pico_processor import (
    PicoBiInverseKinematicsProcessor,
)
from lerobot_teleoperator_rokae.lerobot_teleoperator_rokae.devices.pico_single.pico_single_processor import (
    PicoSingleInverseKinematicsProcessor,
)


def _callback_mode_value(cb) -> Optional[str]:
    """Helper to normalize enum / string callback_mode."""
    if cb is None:
        return None
    return getattr(cb, "value", cb)


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
    
    # 运动学参数
    left_rbv = getattr(cfg.robot, "left_rbv", [])
    right_rbv = getattr(cfg.robot, "right_rbv", [])
    left_min_joint = getattr(cfg.robot, "left_min_joint", [])
    left_max_joint = getattr(cfg.robot, "left_max_joint", [])
    right_min_joint = getattr(cfg.robot, "right_min_joint", [])
    right_max_joint = getattr(cfg.robot, "right_max_joint", [])

    if cfg.teleop.type == "bi_spacemouse":
        # 所有模式都使用相同的 BiInverseKinematicsProcessor
        teleop_action_processor_steps = [
            BiInverseKinematicsProcessor(
                left_joint_num=left_joint_num,
                right_joint_num=right_joint_num,
                control_period=1.0 / cfg.dataset.fps,
                trans_max_vel=0.1,
                rot_max_vel=0.2,
                left_rbv=left_rbv,
                right_rbv=right_rbv,
                left_min_joint=left_min_joint,
                left_max_joint=left_max_joint,
                right_min_joint=right_min_joint,
                right_max_joint=right_max_joint,
                left_tool_end_pos=left_tool_end_pos,
                left_tool_ref_pos=left_tool_ref_pos,
                left_base_frame_in_world=left_base_frame_in_world,
                right_tool_end_pos=right_tool_end_pos,
                right_tool_ref_pos=right_tool_ref_pos,
                right_base_frame_in_world=right_base_frame_in_world,
            )
        ]
    elif cfg.teleop.type == "pico":
        # 所有模式都使用相同的 PicoBiInverseKinematicsProcessor
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        teleop_action_processor_steps = [
            PicoBiInverseKinematicsProcessor(
                left_joint_num=left_joint_num,
                right_joint_num=right_joint_num,
                left_rbv=left_rbv,
                right_rbv=right_rbv,
                left_min_joint=left_min_joint,
                left_max_joint=left_max_joint,
                right_min_joint=right_min_joint,
                right_max_joint=right_max_joint,
                left_tool_end_pos=left_tool_end_pos,
                left_tool_ref_pos=left_tool_ref_pos,
                left_base_frame_in_world=left_base_frame_in_world,
                right_tool_end_pos=right_tool_end_pos,
                right_tool_ref_pos=right_tool_ref_pos,
                right_base_frame_in_world=right_base_frame_in_world,
            )
        ]
    else:
        raise ValueError(f"Unsupported teleop type: {cfg.teleop.type}")

    # 根据 callback_mode 配置不同的 robot_action_processor_steps
    if cb_mode == "cart_vel":
        # pico 不支持 cart_vel
        if cfg.teleop.type == "pico":
            raise ValueError(f"cart_vel is not supported for pico")
        robot_action_processor_steps = [
            BiCartVelRefToBaseProcessor(
                left_tool_ref_pos=left_tool_ref_pos,
                left_base_frame_in_world=left_base_frame_in_world,
                right_tool_ref_pos=right_tool_ref_pos,
                right_base_frame_in_world=right_base_frame_in_world,
            ),
            BiSelectActionByCallbackMode(callback_mode="cart_vel", left_joint_num=left_joint_num, right_joint_num=right_joint_num),
        ]
    elif cb_mode == "cart_pos":
        robot_action_processor_steps = [
            BiCartPosRefToBaseProcessor(
                left_tool_ref_pos=left_tool_ref_pos,
                left_base_frame_in_world=left_base_frame_in_world,
                right_tool_ref_pos=right_tool_ref_pos,
                right_base_frame_in_world=right_base_frame_in_world,
            ),
            BiSelectActionByCallbackMode(callback_mode="cart_pos", left_joint_num=left_joint_num, right_joint_num=right_joint_num),
        ]
    elif cb_mode == "joint_pos":
        robot_action_processor_steps = [
            BiSelectActionByCallbackMode(callback_mode="joint_pos", left_joint_num=left_joint_num, right_joint_num=right_joint_num),
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
        BiCartPosBaseToRefObservationProcessor(
            left_tool_ref_pos=left_tool_ref_pos,
            left_base_frame_in_world=left_base_frame_in_world,
            right_tool_ref_pos=right_tool_ref_pos,
            right_base_frame_in_world=right_base_frame_in_world,
        )
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
        teleop_action_processor_steps = [
            InverseKinematicsProcessor(
                joint_num=joint_num,
                control_period=1.0 / cfg.dataset.fps,
                trans_max_vel=0.1,
                rot_max_vel=0.2,
                rbv=getattr(cfg.robot, "rbv", []),
                min_joint=getattr(cfg.robot, "min_joint", []),
                max_joint=getattr(cfg.robot, "max_joint", []),
                tool_end_pos=getattr(robot, "tool_end_pos", None),
                tool_ref_pos=getattr(robot, "tool_ref_pos", None),
                base_frame_in_world=getattr(robot, "base_frame_in_world", None),
            )
        ]
    elif cfg.teleop.type == "pico_single":
        trans_max_vel = getattr(cfg.teleop, "trans_max_vel")
        rot_max_vel = getattr(cfg.teleop, "rot_max_vel")
        teleop_action_processor_steps = [
            PicoSingleInverseKinematicsProcessor(
                joint_num=joint_num,
                rbv=getattr(cfg.robot, "rbv", []),
                min_joint=getattr(cfg.robot, "min_joint", []),
                max_joint=getattr(cfg.robot, "max_joint", []),
                tool_end_pos=getattr(robot, "tool_end_pos", None),
                tool_ref_pos=getattr(robot, "tool_ref_pos", None),
                base_frame_in_world=getattr(robot, "base_frame_in_world", None),
            )
        ]
    else:
        raise ValueError(f"Unsupported single-arm teleop type: {cfg.teleop.type}")

    # 根据 callback_mode 配置不同的 robot_action_processor_steps
    if cb_mode == "cart_vel":
        # pico_single 不支持 cart_vel
        if cfg.teleop.type == "pico_single":
            raise ValueError("cart_vel is not supported for pico_single")
        robot_action_processor_steps = [
            CartVelRefToBaseProcessor(
                tool_ref_pos=getattr(robot, "tool_ref_pos", None),
                base_frame_in_world=getattr(robot, "base_frame_in_world", None),
            ),
            SelectActionByCallbackMode(callback_mode="cart_vel", joint_num=joint_num),
        ]
    elif cb_mode == "cart_pos":
        robot_action_processor_steps = [
            CartPosRefToBaseProcessor(
                tool_ref_pos=getattr(robot, "tool_ref_pos", None),
                base_frame_in_world=getattr(robot, "base_frame_in_world", None),
            ),
            SelectActionByCallbackMode(callback_mode="cart_pos", joint_num=joint_num),
        ]
    elif cb_mode == "joint_pos":
        robot_action_processor_steps = [
            SelectActionByCallbackMode(callback_mode="joint_pos", joint_num=joint_num),
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
        CartPosBaseToRefObservationProcessor(
            tool_ref_pos=getattr(robot, "tool_ref_pos", None),
            base_frame_in_world=getattr(robot, "base_frame_in_world", None),
        )
    ]
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=robot_observation_processor_steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_action_processor, robot_action_processor, robot_observation_processor

# TODO：支持 pico 
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
    processor_step = None
    is_bimanual = False

    # 识别 rokae 的单臂 / 双臂 Processor（Generate* 或 IK）
    for step in teleop_action_processor.steps:
        if isinstance(step, BiInverseKinematicsProcessor):
            processor_step = step
            is_bimanual = True
            break
        if isinstance(step, PicoBiInverseKinematicsProcessor):
            processor_step = step
            is_bimanual = True
            break
        if isinstance(step, InverseKinematicsProcessor):
            processor_step = step
            is_bimanual = False
            break
        if isinstance(step, PicoSingleInverseKinematicsProcessor):
            processor_step = step
            is_bimanual = False
            break

    if processor_step is None:
        return False

    if is_bimanual:
        # 双臂：使用 initial_left/right_gripper_state
        left_gripper_state = getattr(processor_step, "initial_left_gripper_state", 1)
        right_gripper_state = getattr(processor_step, "initial_right_gripper_state", 1)

        if hasattr(robot, "set_gripper_states"):
            robot.set_gripper_states(left_gripper_state, right_gripper_state)
        if hasattr(processor_step, "reset"):
            processor_step.reset(
                left_gripper_state=left_gripper_state,
                right_gripper_state=right_gripper_state,
            )
    else:
        # 单臂：使用 initial_gripper_state
        gripper_state = getattr(processor_step, "initial_gripper_state", 1)

        if hasattr(robot, "set_gripper_state"):
            robot.set_gripper_state(gripper_state)
        if hasattr(processor_step, "reset"):
            processor_step.reset(gripper_state=gripper_state)

    return True


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
        try:
            robot.reset_position()
            logging.info("Robot reset to drag position")
        except Exception as e:
            logging.warning(f"Failed to reset robot position: {e}")

    # Reset teleop internal accumulation buffers; otherwise the next control frame
    # may replay previous episode deltas and pull the robot away from drag pose.
    if teleop is not None and hasattr(teleop, "reset_for_new_episode"):
        try:
            teleop.reset_for_new_episode()
        except Exception as e:
            logging.warning(f"Failed to reset teleop state for new episode: {e}")

    # Reset gripper states for supported teleop systems
    if teleop is not None and getattr(teleop, "name", None) in ("spacemouse", "bi_spacemouse", "pico_single", "pico"):
        reset_gripper_states(robot, teleop_action_processor)


