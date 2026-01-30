"""Rokae 机器人 HTTP API 客户端

封装所有与 rokae_server 的 HTTP 通信逻辑，
可以被任何库使用，不依赖 lerobot。
"""

import requests
import numpy as np
from typing import List, Dict, Any, Optional


class RokaeClientError(Exception):
    """Rokae 客户端错误基类"""
    pass


class RokaeClient:
    """Rokae 机器人 HTTP API 客户端
    
    封装所有与 rokae_server 的 HTTP 通信逻辑。
    提供统一的接口访问 Rokae 机器人的控制功能。
    
    Args:
        base_url: 服务器基础 URL，例如 "http://127.0.0.1:5000"
        timeout: 请求超时时间（秒），默认 5.0
    
    Example:
        >>> client = RokaeClient(base_url="http://127.0.0.1:5000")
        >>> client.start_realtime_loop("joint_position", "joint_pos")
        >>> client.set_target_joint_pos(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        >>> state = client.get_state(["joint_pos_cmd", "gripper_pos"])
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 5.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发送 GET 请求
        
        Args:
            endpoint: API 端点路径（不含基础 URL）
            params: 查询参数
        
        Returns:
            响应的 JSON 数据
        
        Raises:
            RokaeClientError: 请求失败时抛出
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            raise RokaeClientError(f"GET {url} failed: {e}") from e
    
    def _post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发送 POST 请求
        
        Args:
            endpoint: API 端点路径（不含基础 URL）
            json: JSON 请求体
        
        Returns:
            响应的 JSON 数据
        
        Raises:
            RokaeClientError: 请求失败时抛出
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            r = requests.post(url, json=json, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.JSONDecodeError as e:
            raise RokaeClientError(f"POST {url} returned invalid JSON: {r.text}") from e
        except requests.exceptions.RequestException as e:
            raise RokaeClientError(f"POST {url} failed: {e}") from e
    
    def is_in_realtime_loop(self) -> bool:
        """检查是否在实时控制循环中
        
        Returns:
            如果在实时控制循环中返回 True，否则返回 False
        """
        response = self._get("is_in_realtime_loop")
        return response.get("is_in_realtime_loop", False)
    
    def start_realtime_loop(self, rt_control_mode: str, callback_mode: str) -> Dict[str, Any]:
        """启动实时控制循环
        
        Args:
            rt_control_mode: 实时控制模式，例如 "joint_position", "cartesian_position"
            callback_mode: 回调模式，例如 "joint_pos", "cart_pos", "cart_vel"
        
        Returns:
            服务器响应数据
        """
        payload = {
            "rt_control_mode": rt_control_mode,
            "callback_mode": callback_mode
        }
        return self._post("start_realtime_loop", json=payload)
    
    def stop_realtime_loop(self) -> Dict[str, Any]:
        """停止实时控制循环
        
        Returns:
            服务器响应数据
        """
        return self._post("stop_realtime_loop")
    
    def set_target_joint_pos(self, joint_pos: np.ndarray) -> Dict[str, Any]:
        """设置目标关节位置
        
        Args:
            joint_pos: 关节位置数组（numpy array）
        
        Returns:
            服务器响应数据
        """
        payload = {"joint_pos": joint_pos.tolist()}
        return self._post("set_target_joint_pos", json=payload)
    
    def set_target_cart_pos(self, cart_pos: np.ndarray) -> Dict[str, Any]:
        """设置目标笛卡尔位置
        
        Args:
            cart_pos: 笛卡尔位置数组（numpy array），通常为 6 维 [x, y, z, rx, ry, rz]
        
        Returns:
            服务器响应数据
        """
        payload = {"cart_pos": cart_pos.tolist()}
        return self._post("set_target_cart_pos", json=payload)
    
    def set_target_cart_vel(self, cart_vel: np.ndarray) -> Dict[str, Any]:
        """设置目标笛卡尔速度
        
        Args:
            cart_vel: 笛卡尔速度数组（numpy array），通常为 6 维 [vx, vy, vz, wx, wy, wz]
        
        Returns:
            服务器响应数据
        """
        payload = {"cart_vel": cart_vel.tolist()}
        return self._post("set_target_cart_vel", json=payload)
    
    def open_gripper(self) -> Dict[str, Any]:
        """打开夹爪
        
        Returns:
            服务器响应数据
        """
        return self._post("open_gripper")
    
    def close_gripper(self) -> Dict[str, Any]:
        """关闭夹爪
        
        Returns:
            服务器响应数据
        """
        return self._post("close_gripper")
    
    def get_state(self, quantities: List[str]) -> Dict[str, Any]:
        """获取机器人状态
        
        Args:
            quantities: 要获取的状态量列表，例如 ["joint_pos_cmd", "cart_pos_cmd", "gripper_pos"]
        
        Returns:
            包含请求状态量的字典
        
        Raises:
            RokaeClientError: 如果请求的状态量无效
        """
        params = {"quantities": ",".join(quantities)}
        response = self._get("get_state", params=params)
        
        # 检查是否有错误信息
        if "error" in response:
            raise RokaeClientError(f"Get state failed: {response['error']}")
        
        return response
    
    def reset_position(self) -> Dict[str, Any]:
        """重置机器人到拖拽位姿
        
        Returns:
            服务器响应数据
        """
        return self._post("reset_position")
