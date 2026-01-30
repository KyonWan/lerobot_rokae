import time
import numpy as np
from .gripper_server import GripperServer
from .rokae_sdk.xCoreSDK_python import PyTypeVectorInt
import threading

class DahuanGripperServer(GripperServer):
    def __init__(self,rokae_robot,ec):
        super().__init__()
        self.ec=ec
        self.robot=rokae_robot
        self.activate = PyTypeVectorInt([0xA5])
        self.open_cmd  = PyTypeVectorInt([1000])   # fully open
        self.close_cmd = PyTypeVectorInt([0])   # fully close
        # self.activate_gripper()
        self.gripper_pos = PyTypeVectorInt([1])
        # 启动更新夹爪位置的线程
        self._gripper_thread = threading.Thread(target=self._update_gripper, daemon=True)
        self._gripper_thread.start()
        self.binary_gripper_pose = int(self.gripper_pos.content()[0])
    
    def activate_gripper(self):
        # 向机器人发送激活夹爪的指令
        self.robot.XPRWModbusRTUReg(0x01, 0x06, 0x0100, "int16", 1, self.activate, False, self.ec)
        time.sleep(1)
        self.binary_gripper_pose = 1
    
    def reset_gripper(self):
        self.activate_gripper()
        
    def open(self):
        if self.binary_gripper_pose == 1:
            return
        # 从机地址 0x01，功能码 0x06，寄存器地址 0x0103，数据类型 int16，数据长度 1，数据值 0，是否需要改变CRC校验高低位
        #（一般为false，个别设备的CRC校验高低位需要反转，否则无法正常通信），默认值false即可
        self.robot.XPRWModbusRTUReg(0x01, 0x06, 0x0103, "int16", 1, self.open_cmd, False, self.ec)
        self.binary_gripper_pose = 1

    def close(self):
        if self.binary_gripper_pose == 0:
            return
        self.robot.XPRWModbusRTUReg(0x01, 0x06, 0x0103, "int16", 1, self.close_cmd, False, self.ec)
        self.binary_gripper_pose = 0

    def close_slow(self):
        if self.binary_gripper_pose == 0:
            return
        self.robot.XPRWModbusRTUReg(0x01, 0x06, 0x0103, "int16", 1, self.close_cmd, False, self.ec)
        self.binary_gripper_pose = 0

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 1000]
            position的值范围是[0,1]
            用鼠标控制夹爪是离散的，好像暂时用不到这个功能
        """
        width = float(position * 1000) 
        if width > 1000:
            width = 1000
        if width < 0:
            width = 0
        self.robot.XPRWModbusRTUReg(0x01, 0x06, 0x0103, "int16", 1, PyTypeVectorInt([int(width)]), False, self.ec)
        

    def _update_gripper(self):
        """internal callback to get the latest gripper position."""
        while True:
            self.robot.XPRWModbusRTUReg(0x01, 0x03, 0x0202, "int16", 1, self.gripper_pos, False, self.ec)
            time.sleep(0.1)
              
