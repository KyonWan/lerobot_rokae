import time
import numpy as np
from .gripper_server import GripperServer
from .rokae_sdk.xCoreSDK_python import PyTypeVectorInt
import threading

class LinkerhandV10Server(GripperServer):
    """灵心巧手V10控制服务器
    上层接口仍然是0/1状态（0=闭合，1=张开），底层自动转换为10个关节角度并下发
    """
    def __init__(self, rokae_robot, ec, arm_side="right"):
        """
        初始化灵巧手服务器

        Args:
            rokae_robot: Rokae机器人对象
            ec: 错误码字典
            arm_side: 手臂侧，"left"或"right"，默认为"right"
        """
        super().__init__()
        self.ec = ec
        self.robot = rokae_robot
        self.arm_side = arm_side

        # 灵巧手配置：张开和闭合状态的关节角度（10个关节，每个值0-255）
        # 参考 cr_record_linkerhand_teleprocess.py 中的配置
        self.open_angles = [170, 120, 230, 230, 230, 230, 128, 128, 128, 60]  # 张开状态
        self.close_angles = [120, 120, 90, 90, 90, 230, 128, 128, 128, 60]  # 闭合状态

        # 灵巧手Modbus配置：根据左右手设置不同的从机地址
        if arm_side == "left":
            self.slave_addr = 0x28  # 左手从机地址
        else:
            self.slave_addr = 0x27  # 右手从机地址（默认）
        self.start_addr = 0x0000       # 起始地址
        self.kMaxChunk = 3             # 每次最多写入3个寄存器

        # 当前灵巧手状态（上层接口：0=闭合，1=张开）
        self.binary_gripper_pose = 0

    def activate_gripper(self):
        """激活灵巧手"""
        # 灵心巧手可能不需要特殊的激活指令，这里可以留空或实现具体激活逻辑
        self.binary_gripper_pose = 0
        print("灵巧手已激活")

    def reset_gripper(self):
        """重置灵巧手"""
        self.activate_gripper()

    def _write_angles(self, angles):
        """分块写入关节角度到灵巧手
        
        Args:
            angles: 10个关节角度的列表，每个值范围0-255
            
        Returns:
            bool: 写入是否成功
        """
        offset = 0
        write_success = True

        while offset < len(angles):
            remaining = len(angles) - offset
            chunk_size = min(remaining, self.kMaxChunk)

            # 准备当前块的数据
            buffer = angles[offset:offset + chunk_size]
            buffer_vector = PyTypeVectorInt(buffer)

            # 写入寄存器：单个用0x06，多个用0x10
            if chunk_size == 1:
                self.robot.XPRWModbusRTUReg(
                    self.slave_addr, 
                    0x06, 
                    self.start_addr + offset, 
                    "uint16", 
                    1, 
                    buffer_vector, 
                    False, 
                    self.ec
                )
            else:
                self.robot.XPRWModbusRTUReg(
                    self.slave_addr, 
                    0x10, 
                    self.start_addr + offset, 
                    "uint16", 
                    chunk_size, 
                    buffer_vector, 
                    False, 
                    self.ec
                )

            if self.ec["ec"]:
                print(f"灵巧手写入失败 (offset={offset}): {self.ec}")
                write_success = False
                break

            time.sleep(0.005)  # 每次写入后等待5ms
            offset += chunk_size

        return write_success

    def open(self):
        """打开灵巧手（上层接口：1）"""
        if self.binary_gripper_pose == 1:
            return

        success = self._write_angles(self.open_angles)

        if success:
            self.binary_gripper_pose = 1
            time.sleep(0.01)  # 等待灵巧手执行
        else:
            print("灵巧手打开失败")

    def close(self):
        """闭合灵巧手（上层接口：0）"""
        if self.binary_gripper_pose == 0:
            return

        success = self._write_angles(self.close_angles)

        if success:
            self.binary_gripper_pose = 0
            time.sleep(0.01)  # 等待灵巧手执行
        else:
            print("灵巧手闭合失败")

    def close_slow(self):
        """缓慢闭合灵巧手（与close相同实现）"""
        self.close()

    def move(self, position: float):
        """移动灵巧手到指定位置
        
        Args:
            position: 位置值，范围[0, 1]，0表示闭合，1表示张开
            注意：灵心巧手是离散控制，暂时只支持完全张开或完全闭合
        """
        if position >= 0.5:
            self.open()
        else:
            self.close()
