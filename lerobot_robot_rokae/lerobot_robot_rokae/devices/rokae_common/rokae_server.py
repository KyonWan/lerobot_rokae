"""Rokae 机器人服务器

提供 Rokae 机器人的实时控制接口，支持关节位置、笛卡尔位置和速度控制。
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from math import pi as M_PI
from threading import Lock
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R

from .dahuan_gripper_server import DahuanGripperServer
from .linkerhand_v10_server import LinkerhandV10Server
from .rokae_sdk import xCoreSDK_python as rokae_sdk_api
from .rokae_sdk.xCoreSDK_python import (
    CartesianPosition,
    JointPosition,
    PyTypeDouble,
    PyTypeVectorDouble,
    RtControllerMode,
    Toolset,
    Load,
    Frame,
)


def pose_to_T(xyz:Optional[list[float], np.ndarray], rpy:Optional[list[float], np.ndarray], degrees:bool=False) -> np.ndarray:
    if xyz is None:
        xyz = [0.0, 0.0, 0.0]
    if rpy is None:
        rpy = [0.0, 0.0, 0.0]
    return R.from_euler('xyz', rpy, degrees=degrees).as_matrix() @ np.eye(4)

def T_to_pose(T:np.ndarray, degrees:bool=False) -> tuple[np.ndarray, np.ndarray]:
    xyz = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    rpy = rot.as_euler('xyz', degrees=degrees)
    return xyz, rpy


# 配置日志
logger = logging.getLogger(__name__)


class ArmSide(str, Enum):
    """手臂侧枚举"""
    LEFT = "left"
    RIGHT = "right"


class CallbackMode(str, Enum):
    """回调模式枚举"""
    CART_POS = "cart_pos"
    CART_VEL = "cart_vel"
    JOINT_POS = "joint_pos"


@dataclass
class RobotState:
    """机器人状态数据类
    
    Attributes:
        joint_pos_cmd: 关节位置命令（弧度）
        cart_pos_cmd: 笛卡尔位置命令 [x, y, z, rx, ry, rz]
    """
    joint_pos_cmd: List[float]
    cart_pos_cmd: List[float]


class RokaeServer:
    """Rokae 机器人服务器类
    
    提供实时控制接口，支持多种控制模式和回调模式。
    
    Attributes:
        robot_ip: 机器人 IP 地址
        host_ip: 主机 IP 地址
        joint_num: 关节数量（6 或 7）
        arm_side: 手臂侧（"left" 或 "right"）
        q: 当前关节角度（弧度，numpy array）
        q_cmd: 关节角度命令（弧度，numpy array）
        dq: 关节速度（弧度/秒，numpy array）
        pos: 笛卡尔位置 [x, y, z, rx, ry, rz]（米，弧度）
        twist: 末端速度 [vx, vy, vz, wx, wy, wz]（米/秒，弧度/秒）
        wrench: 力和力矩 [fx, fy, fz, mx, my, mz]（牛顿，牛米）
        psi: 臂角（仅7轴机器人，弧度）
    """
    
    # 常量定义
    CONTROL_PERIOD: float = 0.001  # 控制周期（秒）
    CART_VEL_TIMEOUT: float = 0.2  # 笛卡尔速度命令超时（秒）
    MAX_LINEAR_VEL: float = 0.2  # 最大线速度（米/秒）
    MAX_ANGULAR_VEL: float = 0.5  # 最大角速度（弧度/秒）
    MAX_JOINT_VEL: float = M_PI / 2  # 最大关节速度（弧度/秒）
    
    # 6轴机器人拖拽位姿（弧度）
    DRAG_POSE_6DOF: np.ndarray = np.array(
        [M_PI / 3, M_PI / 6, -M_PI / 2, 0.0, -M_PI / 3, M_PI],
        dtype=np.float64
    )
    
    # 7轴机器人拖拽位姿（度转弧度）
    DRAG_POSE_7DOF_LEFT: np.ndarray = np.array(
        [-70, 34, -64, 105, 50, 0, -10],
        dtype=np.float64
    ) * M_PI / 180
    
    DRAG_POSE_7DOF_RIGHT: np.ndarray = np.array(
        [30, 75, -74, 90, 12, 8, -6],
        dtype=np.float64
    ) * M_PI / 180
    
    # 阻抗参数
    JOINT_IMPEDANCE_6DOF: List[float] = [3000, 3000, 3000, 1000, 1000, 1000]
    JOINT_IMPEDANCE_7DOF: List[float] = [3000, 3000, 3000, 1000, 1000, 1000, 1000]
    CART_IMPEDANCE: List[float] = [4000, 4000, 4000, 1000, 1000, 1000]

    TOOL_MASS: float = 0.0
    TOOL_CENTER_OF_MASS: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    TOOL_INERTIA_TENSOR: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    TOOL_END_POS: np.ndarray = np.zeros(6, dtype=np.float64)
    TOOL_REF_POS: np.ndarray = np.zeros(6, dtype=np.float64)

    def __init__(
        self,
        robot_ip: str,
        host_ip: str,
        joint_num: Literal[6, 7],
        arm_side: Union[str, ArmSide] = ArmSide.RIGHT,
    ) -> None:
        """初始化 RokaeServer
        
        Args:
            robot_ip: 机器人 IP 地址
            host_ip: 主机 IP 地址
            joint_num: 关节数量，必须是 6 或 7
            arm_side: 手臂侧，"left" 或 "right"，默认为 "right"
        
        Raises:
            ValueError: 如果关节数量不是 6 或 7，或 arm_side 无效
            RuntimeError: 如果机器人初始化失败
        """
        if joint_num not in (6, 7):
            raise ValueError(f"关节数量必须是 6 或 7，当前值: {joint_num}")
        
        # 标准化 arm_side
        if isinstance(arm_side, str):
            arm_side = ArmSide(arm_side.lower())
        
        self.robot_ip: str = robot_ip
        self.host_ip: str = host_ip
        self.joint_num: Literal[6, 7] = joint_num
        self.arm_side: ArmSide = arm_side
        self.toolset: Toolset = Toolset(
            load=Load(
                mass=self.TOOL_MASS, 
                cog=self.TOOL_CENTER_OF_MASS.tolist(),
                inertia=self.TOOL_INERTIA_TENSOR.tolist()),
            end=Frame(frame=self.TOOL_END_POS.tolist()),
            ref=Frame(trans=self.TOOL_REF_POS.tolist())
        )
        
        # 基坐标系变换矩阵（旋转矩阵的逆等于转置）
        matrix = R.from_euler("XYZ", [-90, 0, 0], degrees=True).as_matrix()
        self.M_base_in_ref: np.ndarray = matrix.T
        
        # 私有状态变量（SDK 类型）
        self.__joint_pos_feedback: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * self.joint_num) # 关节角度, 弧度
        self.__joint_pos_cmd: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * self.joint_num) # 关节角度命令, 弧度
        self.__joint_vel_feedback: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * self.joint_num) # 关节速度, 弧度/秒
        self.__cart_pos_feedback: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * 6) # 末端相对于基座标系
        self.__cart_vel_feedback: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * 6) # 末端相对于基座标系的速度
        self.__ext_wrench: PyTypeVectorDouble = PyTypeVectorDouble([0.0] * 6) # 末端受到的力和力矩, 相对于基座标系
        self.__psi: Optional[PyTypeDouble] = (
            PyTypeDouble(0.0) if self.joint_num == 7 else None
        )
        
        # 公共状态变量（numpy array）
        self.joint_pos_feedback: np.ndarray = np.zeros(self.joint_num, dtype=np.float64) # 关节角度, 弧度
        self.joint_pos_cmd: np.ndarray = np.zeros(self.joint_num, dtype=np.float64) # 关节角度命令, 弧度
        self.joint_vel_feedback: np.ndarray = np.zeros(self.joint_num, dtype=np.float64) # 关节速度, 弧度/秒
        self.cart_pos_cmd: np.ndarray = np.zeros(6, dtype=np.float64) # 笛卡尔位置命令, 末端相对于基座标系
        self.cart_pos_feedback: np.ndarray = np.zeros(6, dtype=np.float64) # 末端相对于基座标系
        self.cart_vel_feedback: np.ndarray = np.zeros(6, dtype=np.float64) # 末端相对于基座标系的速度
        self.ext_wrench: np.ndarray = np.zeros(6, dtype=np.float64) # 末端受到的力和力矩, 相对于基座标系
        self.psi: Optional[float] = 0.0 if self.joint_num == 7 else None # 臂角, 弧度, 当前版本不对臂角控制，将反馈值下发，当作指令值
        
        # 控制命令
        self.target_cart_pos: np.ndarray = np.zeros(6, dtype=np.float64)
        self.target_psi: float = 0.0
        self.target_joint_pos: np.ndarray = np.zeros(self.joint_num, dtype=np.float64)
        self.target_cart_vel: np.ndarray = np.zeros(6, dtype=np.float64)
        
        # 控制状态
        self.__terminate_realtime_loop: bool = False
        self.__in_realtime_loop: bool = False
        self.__last_vel_cmd_time: float = 0.0
        self.rt_control_mode: Optional[RtControllerMode] = None
        self.callback_mode: Optional[str] = None
        
        # 线程锁
        self.lock: Lock = Lock()
        
        # 错误码字典
        self.ec: Dict[str, int] = {}
        
        # 初始化机器人
        self._initialize_robot()
    
    def _initialize_robot(self) -> None:
        """初始化机器人连接和配置
        
        Raises:
            RuntimeError: 如果初始化失败
        """
        try:
            logger.info(f"开始连接机械臂 {self.robot_ip} -> {self.host_ip}")
            
            # 创建机器人实例
            if self.joint_num == 6:
                self.robot = rokae_sdk_api.xMateRobot(self.robot_ip, self.host_ip)
            else:  # joint_num == 7
                self.robot = rokae_sdk_api.xMateErProRobot(self.robot_ip, self.host_ip)
            
            logger.info("连接机械臂成功")
            
            # 切换到自动模式并上电
            logger.info("切换到自动模式...")
            self.robot.setOperateMode(rokae_sdk_api.OperateMode.automatic, self.ec)
            if self.ec.get("ec", 0) != 0:
                logger.warning(f"切换到自动模式失败，错误代码: {self.ec}")
            
            # 设置实时控制参数
            logger.info("设置实时控制参数...")
            self.robot.setRtNetworkTolerance(80, self.ec)
            self.robot.setMotionControlMode(
                rokae_sdk_api.MotionControlMode.RtCommandMode, self.ec
            )
            self.robot.setPowerState(True, self.ec)
            self.rtCon = self.robot.getRtMotionController()
            logger.info("实时控制器获取成功")
            
            # 移动到拖拽位姿
            self._move_to_drag_pose()
            
            # 设置阻抗参数
            self._setup_impedance()
            
            # 打开 RS485 通信并输出 24V 供电
            self.robot.setxPanelRS485(opt=3, if_rs485=True, ec=self.ec)

            # 更新笛卡尔和关节位置
            self.update_cart_and_joint_pos()
            
            logger.info("RokaeServer 初始化完成")
            
        except Exception as e:
            logger.error(f"RokaeServer 初始化失败: {e}")
            raise RuntimeError(f"机器人初始化失败: {e}") from e
    
    def _move_to_drag_pose(self) -> None:
        """移动到拖拽位姿"""
        logger.info("移动到拖拽位姿...")
        
        if self.joint_num == 6:
            self.q_drag: np.ndarray = self.DRAG_POSE_6DOF.copy()
        else:  # joint_num == 7
            if self.arm_side == ArmSide.LEFT:
                self.q_drag = self.DRAG_POSE_7DOF_LEFT.copy()
            else:  # arm_side == ArmSide.RIGHT
                self.q_drag = self.DRAG_POSE_7DOF_RIGHT.copy()
        
        self.rtCon.MoveJ(0.2, self.robot.jointPos(self.ec), self.q_drag.tolist())
        logger.info("拖拽位姿设置完成")
    
    def _setup_impedance(self) -> None:
        """设置阻抗参数"""
        logger.info("设置阻抗模式...")
        
        if self.joint_num == 6:
            self.joint_impedance_factor: List[float] = self.JOINT_IMPEDANCE_6DOF.copy()
        else:  # joint_num == 7
            self.joint_impedance_factor = self.JOINT_IMPEDANCE_7DOF.copy()
        
        self.cart_impedance_factor: List[float] = self.CART_IMPEDANCE.copy()
    
    @property
    def is_in_realtime_loop(self) -> bool:
        """检查是否在实时控制循环中
        
        Returns:
            如果在实时控制循环中返回 True，否则返回 False
        """
        return self.__in_realtime_loop
    
    def start_realtime_loop(
        self,
        rt_control_mode: RtControllerMode,
        callback_mode: Union[str, CallbackMode],
    ) -> None:
        """启动实时控制循环
        
        Args:
            rt_control_mode: 实时控制模式
            callback_mode: 回调模式，"cart_pos"、"cart_vel" 或 "joint_pos"
        
        Raises:
            ValueError: 如果回调模式无效
        """
        if self.is_in_realtime_loop:
            logger.warning("实时控制循环已在运行中")
            return
        
        # 标准化 callback_mode
        if isinstance(callback_mode, CallbackMode):
            callback_mode = callback_mode.value
        elif callback_mode not in ["cart_pos", "cart_vel", "joint_pos"]:
            raise ValueError(
                f"无效的回调模式: {callback_mode}, "
                f"必须是 'cart_pos'、'cart_vel' 或 'joint_pos'"
            )
        
        self.rt_control_mode = rt_control_mode
        self.callback_mode = callback_mode
        
        # 如果之前有终止标志，先停止
        if self.__terminate_realtime_loop:
            logger.info("停止之前的实时控制循环")
            self.__terminate_realtime_loop = False
            self.rtCon.stopLoop()
            self.robot.stopReceiveRobotState()
        
        # 初始化位置
        self.update_cart_and_joint_pos()
        
        # 启动接收机器人状态
        read_data_names: List[str] = [
            "q_m", "q_c", "dq_m", "pos_abc_m", "pos_vel_m", "tau_ext_base"
        ]
        if self.joint_num == 7:
            read_data_names.append("psi_m")
        
        # 设置阻抗模式
        if rt_control_mode == RtControllerMode.cartesianImpedance:
            self.rtCon.setCartesianImpedance(self.cart_impedance_factor, self.ec)
        elif rt_control_mode == RtControllerMode.jointImpedance:
            self.rtCon.setJointImpedance(self.joint_impedance_factor, self.ec)
        
        # 启动运动控制
        self.robot.startReceiveRobotState(
            timedelta(milliseconds=1), read_data_names
        )
        self.rtCon.startMove(rt_control_mode)
        if self.control_mode == RtControllerMode.cartesianPosition or self.control_mode == RtControllerMode.cartesianImpedance:
            self.rtCon.setControlLoopCar(self._create_callback(callback_mode))
        elif self.control_mode == RtControllerMode.jointPosition or self.control_mode == RtControllerMode.jointImpedance:
            self.rtCon.setControlLoopJoi(self._create_callback(callback_mode))
        
        self.rtCon.startLoop(False)
        self.__in_realtime_loop = True
        
        logger.info(
            f"实时控制循环已启动: 控制模式={rt_control_mode}, "
            f"回调模式={callback_mode}"
        )
    
    def update_cart_and_joint_pos(self) -> None:
        """初始化笛卡尔和关节位置, 不要在回调函数中调用"""
        cart_pos_feedback_flan_in_base = np.array(
            self.robot.posture(
                rokae_sdk_api.CoordinateType.flangeInBase, self.ec
            ),
            dtype=np.float64
        )
        cart_pos_feedback_end_in_base:np.ndarray = (pose_to_T(cart_pos_feedback_flan_in_base) @ pose_to_T(self.toolset.end.pos))
        self.cart_pos_feedback = T_to_pose(cart_pos_feedback_end_in_base)
        self.cart_pos_cmd = self.cart_pos_feedback.copy()
        self.target_cart_pos = self.cart_pos_feedback.copy()
        logger.debug(f"初始笛卡尔位置: {self.cart_pos_cmd}")
        
        self.joint_pos_feedback = np.array(self.robot.jointPos(self.ec), dtype=np.float64)
        self.joint_pos_cmd = self.joint_pos_feedback.copy()
        self.target_joint_pos = self.joint_pos_feedback.copy()

        self.target_cart_vel = np.zeros(6, dtype=np.float64)
        
        if self.joint_num == 7:
            cart_posture = self.robot.cartPosture(
                rokae_sdk_api.CoordinateType.flangeInBase, self.ec
            )
            self.target_psi = float(cart_posture.elbow)
    
    def stop_realtime_loop(self) -> None:
        """停止实时控制循环"""
        if not self.__in_realtime_loop:
            logger.warning("实时控制循环未在运行")
            return
        if self.__terminate_realtime_loop:
            logger.warning("实时控制循环正在终止")
            return
        
        self.__terminate_realtime_loop = True
        self.__in_realtime_loop = False
        time.sleep(0.1)
        self.rtCon.stopLoop()
        self.robot.stopReceiveRobotState()
        logger.info("实时控制循环已停止")
    
    def reset_position(self) -> None:
        """重置机器人到拖拽位姿"""
        logger.info("重置机器人位置...")
        self.stop_realtime_loop()

        self.rtCon.MoveJ(0.5, self.robot.jointPos(self.ec), self.q_drag.tolist())
        logger.info(f"MoveJ 完成，错误码: {self.ec}")
        
        if self.rt_control_mode is not None and self.callback_mode is not None:
            logger.info("重新启动实时控制循环")
            self.start_realtime_loop(self.rt_control_mode, self.callback_mode)
    
    def _update_proprioceptive_data(self) -> None:
        """更新本体感知数据（关节角度、位置、速度等）"""
        # self.robot.updateRobotState(timedelta(milliseconds=1)) # todo: 确定是否需要删除
         
        # 获取状态数据
        self.robot.getStateData("q_m", self.__joint_pos_feedback, self.joint_num)  # rad
        self.robot.getStateData("q_c", self.__joint_pos_cmd, self.joint_num)  # rad
        self.robot.getStateData("dq_m", self.__joint_vel_feedback, self.joint_num)  # rad/s
        self.robot.getStateData("pos_abc_m", self.__cart_pos_feedback, 6)  # m
        self.robot.getStateData("pos_vel_m", self.__cart_vel_feedback, 6)  # m/s
        self.robot.getStateData("tau_ext_base", self.__ext_wrench, 6)  # Nm
        
        if self.joint_num == 7:
            self.robot.getStateData("psi_m", self.__psi)
        
        # 转换为 numpy array
        self.joint_pos_feedback = np.array(self.__joint_pos_feedback.content(), dtype=np.float64)
        self.joint_vel_feedback = np.array(self.__joint_vel_feedback.content(), dtype=np.float64)
        self.cart_pos_feedback = np.array(self.__cart_pos_feedback.content(), dtype=np.float64)
        self.cart_vel_feedback = np.array(self.__cart_vel_feedback.content(), dtype=np.float64)
        self.ext_wrench = np.array(self.__ext_wrench.content(), dtype=np.float64)

        # 当回调不是关节位置回调时，更新关节位置命令
        if self.callback_mode != CallbackMode.JOINT_POS:
            self.joint_pos_cmd = np.array(self.__joint_pos_cmd.content(), dtype=np.float64)  # todo:测试实机上的值是否正确

        if self.joint_num == 7 and self.__psi is not None:
            self.psi = float(self.__psi.content())
    
    def _create_callback(
        self, callback_mode: str
    ) -> Union[Callable[[], CartesianPosition], Callable[[], JointPosition]]:
        """创建回调函数
        
        Args:
            callback_mode: 回调模式，"cart_pos"、"cart_vel" 或 "joint_pos"
        
        Returns:
            回调函数
        
        Raises:
            ValueError: 如果回调模式无效
        """
        if callback_mode not in ["cart_pos", "cart_vel", "joint_pos"]:
            raise ValueError(
                f"无效的回调模式: {callback_mode}, "
                f"必须是 'cart_pos'、'cart_vel' 或 'joint_pos'"
            )
        
        def cart_pos_callback() -> CartesianPosition:
            """笛卡尔位置回调函数, 返回值为末端相对于基座标系的位姿"""
            self._update_proprioceptive_data()
            
            self.cart_pos_cmd = self._cart_vel_saturate(
                current_cart_pos=self.cart_pos_cmd,
                target_cart_pos=self.target_cart_pos,
                period=self.CONTROL_PERIOD,
                v_lin_max=self.MAX_LINEAR_VEL,
                v_ang_max=self.MAX_ANGULAR_VEL,
            )
            
            # 更新笛卡尔位置命令的SDK类型
            cart_pos_cmd_rokae = CartesianPosition(pose_to_T(self.cart_pos_cmd).flatten().tolist())
            
            if self.joint_num == 7:
                cart_pos_cmd_rokae.hasElbow = True
                cart_pos_cmd_rokae.elbow = self.target_psi
            
            if self.__terminate_realtime_loop:
                cart_pos_cmd_rokae.setFinished()
                self.__terminate_realtime_loop = False
                logger.debug("实时控制循环已终止")
            
            return cart_pos_cmd_rokae
        
        def cart_vel_callback() -> CartesianPosition:
            """笛卡尔速度回调函数, 返回值为末端相对于基座标系的位姿"""
            self._update_proprioceptive_data()
            
            # 速度命令超时处理
            self.__last_vel_cmd_time += self.CONTROL_PERIOD
            if self.__last_vel_cmd_time > self.CART_VEL_TIMEOUT:
                self.target_cart_vel = np.zeros(6, dtype=np.float64)
            
            # 更新笛卡尔位置命令
            self.cart_pos_cmd[0:3] += self.target_cart_vel * self.CONTROL_PERIOD
            
            # 计算新的姿态
            R_cur = R.from_euler("xyz", self.cart_pos_cmd[3:6])
            R_rot = R.from_rotvec(self.target_cart_vel[3:6] * self.CONTROL_PERIOD)
            R_cmd = R_rot * R_cur
            self.cart_pos_cmd[3:6] = R_cmd.as_euler("xyz", degrees=False)

            # 更新笛卡尔位置命令的SDK类型
            cart_pos_cmd_rokae = CartesianPosition(pose_to_T(self.cart_pos_cmd).flatten().tolist())
            
            if self.joint_num == 7:
                cart_pos_cmd_rokae.hasElbow = True
                cart_pos_cmd_rokae.elbow = self.target_psi
            
            if self.__terminate_realtime_loop:
                cart_pos_cmd_rokae.setFinished()
                self.__terminate_realtime_loop = False
                logger.debug("实时控制循环已终止")
            
            return cart_pos_cmd_rokae
        
        def joint_pos_callback() -> JointPosition:
            """关节位置回调函数"""
            self._update_proprioceptive_data()

            self.joint_pos_cmd = self._joint_vel_saturate(
                last_joint_pos=self.joint_pos_cmd,
                target_joint_pos=self.target_joint_pos,
                period=self.CONTROL_PERIOD,
                max_joint_vel=np.full(self.joint_num, self.MAX_JOINT_VEL, dtype=np.float64),
            )
            
            joint_pos_cmd_rokae: JointPosition = JointPosition(joints=self.joint_pos_cmd.tolist())
            
            if self.__terminate_realtime_loop:
                joint_pos_cmd_rokae.setFinished()
                self.__terminate_realtime_loop = False
                logger.debug("实时控制循环已终止")
            
            return joint_pos_cmd_rokae
        
        # 返回对应的回调函数
        callback_map: Dict[str, Callable] = {
            "cart_pos": cart_pos_callback,
            "cart_vel": cart_vel_callback,
            "joint_pos": joint_pos_callback,
        }
        
        return callback_map[callback_mode]
    
    def set_target_joint_pos(self, target_joint_pos: np.ndarray) -> None:
        """设置目标关节位置
        
        Args:
            target_joint_pos: 目标关节位置数组（弧度）
        """
        if target_joint_pos.shape != (self.joint_num,):
            raise ValueError(
                f"关节位置数组长度应为 {self.joint_num}, "
                f"当前长度: {target_joint_pos.shape[0]}"
            )
        
        with self.lock:
            self.target_joint_pos = np.array(target_joint_pos, dtype=np.float64)
    
    def set_target_cart_pos(self, target_cart_pos: np.ndarray) -> None:
        """设置目标笛卡尔位置
        
        Args:
            target_cart_pos: 目标笛卡尔位置 [x, y, z, rx, ry, rz]（米，弧度）, 末端相对于基座标系
        """
        if target_cart_pos.shape != (6,):
            raise ValueError(
                f"笛卡尔位置数组长度应为 6，当前长度: {target_cart_pos.shape[0]}"
            )
        
        with self.lock:
            self.target_cart_pos = np.array(target_cart_pos, dtype=np.float64)
    
    def set_target_cart_vel(self, target_cart_vel: np.ndarray) -> None:
        """设置目标笛卡尔速度
        
        Args:
            target_cart_vel: 目标笛卡尔速度 [vx, vy, vz, wx, wy, wz]（米/秒，弧度/秒）, 末端相对于基座标系
        """
        if target_cart_vel.shape != (6,):
            raise ValueError(
                f"笛卡尔速度数组长度应为 6，当前长度: {target_cart_vel.shape[0]}"
            )
        
        with self.lock:
            self.__last_vel_cmd_time = 0.0
            self.target_cart_vel = np.array(target_cart_vel, dtype=np.float64)
    
    def _joint_vel_saturate(
        self,
        last_joint_pos: np.ndarray,
        target_joint_pos: np.ndarray,
        period: float,
        max_joint_vel: np.ndarray,
    ) -> np.ndarray:
        """关节速度饱和函数
        
        Args:
            last_joint_pos: 上一时刻关节位置（弧度）
            target_joint_pos: 目标关节位置（弧度）
            period: 控制周期（秒）
            max_joint_vel: 最大关节速度（弧度/秒）
        
        Returns:
            饱和后的关节位置列表（弧度）
        """
        joint_vel = (target_joint_pos - last_joint_pos) / period
        joint_vel = np.clip(joint_vel, -max_joint_vel, max_joint_vel)
        joint_pos = last_joint_pos + joint_vel * period
        return joint_pos
    
    def _cart_vel_saturate(
        self,
        current_cart_pos: np.ndarray,
        target_cart_pos: np.ndarray,
        period: float,
        v_lin_max: float,
        v_ang_max: float,
    ) -> np.ndarray:
        """Cartesian velocity saturation (SE(3) style)

        Args:
            current_cart_pos: current Cartesian pose [x, y, z, rx, ry, rz]
            target_cart_pos: target Cartesian pose  [x, y, z, rx, ry, rz]
            period: control period [s]
            v_lin_max: max linear velocity [m/s]
            v_ang_max: max angular velocity [rad/s]

        Returns:
            Saturated Cartesian pose [x, y, z, rx, ry, rz]
        """

        # =====================
        # Linear part
        # =====================
        delta_pos = target_cart_pos[:3] - current_cart_pos[:3]
        v_lin = delta_pos / period

        lin_speed = np.linalg.norm(v_lin)
        if lin_speed > v_lin_max:
            v_lin = v_lin / lin_speed * v_lin_max

        pos_sat = current_cart_pos[:3] + v_lin * period

        # =====================
        # Angular part (SO(3))
        # =====================
        R_curr = R.from_euler("xyz", current_cart_pos[3:], degrees=False)
        R_target = R.from_euler("xyz", target_cart_pos[3:], degrees=False)

        # Relative rotation: from current to target (body frame)
        R_rel = R_curr.inv() * R_target

        # Rotation vector (axis * angle), norm = rotation angle [rad]
        rotvec = R_rel.as_rotvec()
        v_ang = rotvec / period

        ang_speed = np.linalg.norm(v_ang)
        if ang_speed > v_ang_max:
            v_ang = v_ang / ang_speed * v_ang_max

        # Integrate one step in body frame
        R_step = R.from_rotvec(v_ang * period)
        R_sat = R_curr * R_step

        rpy_sat = R_sat.as_euler("xyz", degrees=False)

        # =====================
        # Output
        # =====================
        cart_pose_sat = np.zeros(6, dtype=np.float64)
        cart_pose_sat[:3] = pos_sat
        cart_pose_sat[3:] = rpy_sat

        # Normalize angles to [-pi, pi)
        cart_pose_sat[3:] = (cart_pose_sat[3:] + np.pi) % (2 * np.pi) - np.pi

        return cart_pose_sat


def create_flask_app(
    rokae_server: RokaeServer,
    gripper_server: Union[LinkerhandV10Server, DahuanGripperServer],
) -> Flask:
    """创建 Flask 应用
    
    Args:
        rokae_server: RokaeServer 实例
        gripper_server: 夹爪服务器实例
    
    Returns:
        Flask 应用实例
    """
    app = Flask(__name__)
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    
    @app.route("/start_realtime_loop", methods=["POST"])
    def start_realtime_loop() -> Tuple[Dict, int]:
        """启动实时控制循环"""
        data = request.json
        logger.debug(f"收到启动请求: {data}")
        
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if "rt_control_mode" not in data or "callback_mode" not in data:
            return (
                jsonify({"error": "Missing 'rt_control_mode' or 'callback_mode'"}),
                400,
            )
        
        mode_map: Dict[str, RtControllerMode] = {
            "cartesian_impedance": RtControllerMode.cartesianImpedance,
            "cartesian_position": RtControllerMode.cartesianPosition,
            "joint_impedance": RtControllerMode.jointImpedance,
            "joint_position": RtControllerMode.jointPosition,
        }
        
        if data["rt_control_mode"] not in mode_map:
            return (
                jsonify({
                    "error": f"Invalid rt_control_mode: {data['rt_control_mode']}",
                    "valid": list(mode_map.keys()),
                }),
                400,
            )
        
        try:
            rokae_server.start_realtime_loop(
                mode_map[data["rt_control_mode"]], data["callback_mode"]
            )
            return jsonify({"success": True}), 200
        except Exception as e:
            logger.error(f"启动实时控制循环失败: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/stop_realtime_loop", methods=["POST"])
    def stop_realtime_loop() -> Tuple[Dict, int]:
        """停止实时控制循环"""
        rokae_server.stop_realtime_loop()
        return jsonify({"success": True}), 200
    
    @app.route("/is_in_realtime_loop", methods=["GET"])
    def is_in_realtime_loop() -> Dict:
        """检查是否在实时控制循环中"""
        return jsonify({"is_in_realtime_loop": rokae_server.is_in_realtime_loop})
    
    @app.route("/reset_position", methods=["POST"])
    def reset_position() -> Tuple[Dict, int]:
        """重置机器人位置"""
        try:
            rokae_server.reset_position()
            if gripper_server is not None:
                gripper_server.open()
            return jsonify({"success": True}), 200
        except Exception as e:
            logger.error(f"重置位置失败: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/set_target_joint_pos", methods=["POST"])
    def set_target_joint_pos() -> Tuple[Dict, int]:
        """设置目标关节位置"""
        data = request.json
        if data is None or "joint_pos" not in data:
            return jsonify({"error": "Missing 'joint_pos'"}), 400
        
        try:
            rokae_server.set_target_joint_pos(np.array(data["joint_pos"]))
            return jsonify({"success": True}), 200
        except Exception as e:
            logger.error(f"设置关节位置失败: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/set_target_cart_pos", methods=["POST"])
    def set_target_cart_pos() -> Tuple[Dict, int]:
        """设置目标笛卡尔位置"""
        data = request.json
        if data is None or "cart_pos" not in data:
            return jsonify({"error": "Missing 'cart_pos'"}), 400
        
        try:
            rokae_server.set_target_cart_pos(np.array(data["cart_pos"]))
            return jsonify({"success": True}), 200
        except Exception as e:
            logger.error(f"设置笛卡尔位置失败: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/set_target_cart_vel", methods=["POST"])
    def set_target_cart_vel() -> Tuple[Dict, int]:
        """设置目标笛卡尔速度"""
        data = request.json
        if data is None or "cart_vel" not in data:
            return jsonify({"error": "Missing 'cart_vel'"}), 400
        
        try:
            rokae_server.set_target_cart_vel(np.array(data["cart_vel"]))
            return jsonify({"success": True}), 200
        except Exception as e:
            logger.error(f"设置笛卡尔速度失败: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/open_gripper", methods=["POST"])
    def open_gripper() -> Tuple[Dict, int]:
        """打开夹爪"""
        if gripper_server is not None:
            gripper_server.open()
        return jsonify({"success": True}), 200
    
    @app.route("/close_gripper", methods=["POST"])
    def close_gripper() -> Tuple[Dict, int]:
        """关闭夹爪"""
        if gripper_server is not None:
            gripper_server.close()
        return jsonify({"success": True}), 200
    
    @app.route("/get_state", methods=["GET"])
    def get_state() -> Tuple[Dict, int]:
        """获取机器人状态"""
        quantities = request.args.get("quantities")
        if quantities is None:
            return jsonify({"error": "Missing 'quantities'"}), 400
        
        requested = quantities.split(",")
        
        quantity_map: Dict[str, List[float]] = {
            "joint_pos_cmd": rokae_server.q.tolist(),
            "cart_pos_cmd": rokae_server.pos.tolist(),
            "gripper_pos": [0.0 if gripper_server is None else gripper_server.binary_gripper_pose],
        }
        
        result: Dict[str, List[float]] = {}
        for q in requested:
            if q not in quantity_map:
                return (
                    jsonify({
                        "error": f"Invalid quantity '{q}'",
                        "valid": list(quantity_map.keys()),
                    }),
                    400,
                )
            result[q] = quantity_map[q]
        
        return jsonify(result), 200
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rokae Robot Server")
    parser.add_argument(
        "--robot_ip",
        type=str,
        default="192.168.21.10",
        help="Robot IP address",
    )
    parser.add_argument(
        "--host_ip",
        type=str,
        default="192.168.21.1",
        help="Host IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Flask server port",
    )
    parser.add_argument(
        "--joint_num",
        type=int,
        default=7,
        choices=[6, 7],
        help="Number of joints (6 or 7)",
    )
    parser.add_argument(
        "--end_effector",
        type=str,
        default="none",
        choices=["linkerhand_v10", "dahuan_gripper", "none"],
        help="End effector type",
    )
    
    args = parser.parse_args()
    
    # 根据端口判断左右手：5000=左臂，5001=右臂
    arm_side = ArmSide.LEFT if args.port == 5000 else ArmSide.RIGHT
    
    logger.info(
        f"启动Rokae服务器: 机器人IP={args.robot_ip}, "
        f"主机IP={args.host_ip}, 端口={args.port}, "
        f"关节数={args.joint_num}, 手臂侧={arm_side.value}"
    )
    
    rokae_server = RokaeServer(
        args.robot_ip, args.host_ip, args.joint_num, arm_side=arm_side
    )
    
    if args.end_effector == "linkerhand_v10":
        gripper_server = LinkerhandV10Server(
            rokae_server.robot, rokae_server.ec, arm_side=arm_side.value
        )
    elif args.end_effector == "dahuan_gripper":
        gripper_server = DahuanGripperServer(rokae_server.robot, rokae_server.ec)
    elif args.end_effector == "none":
        gripper_server = None
    else:
        raise ValueError(f"无效的末端执行器类型: {args.end_effector}")
    
    app = create_flask_app(rokae_server, gripper_server)
    
    logger.info(f"RokaeServer Flask running on http://127.0.0.1:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
