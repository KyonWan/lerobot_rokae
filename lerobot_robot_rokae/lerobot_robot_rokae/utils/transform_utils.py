"""
坐标转换工具函数。

提供统一的坐标转换 API，简化齐次变换矩阵和位姿转换操作。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def make_homogeneous(R_mat: np.ndarray, trans_vec: np.ndarray) -> np.ndarray:
    """将旋转矩阵和平移向量组成齐次变换矩阵"""
    H = np.eye(4)
    H[:3, :3] = R_mat
    H[:3, 3] = trans_vec
    return H


def inv_homogeneous(T: np.ndarray) -> np.ndarray:
    """求4x4齐次变换矩阵的逆"""
    R_part = T[:3, :3]
    t_part = T[:3, 3]
    T_inv = np.eye(4)
    R_inv = R_part.T
    t_inv = -R_inv @ t_part
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """
    将位姿 [x, y, z, rx, ry, rz] 转换为齐次变换矩阵。
    
    Args:
        pose: 位姿 [x, y, z, rx, ry, rz]，旋转使用欧拉角 (xyz顺序)
    
    Returns:
        4x4 齐次变换矩阵
    """
    R_mat = R.from_euler("xyz", pose[3:], degrees=False).as_matrix()
    return make_homogeneous(R_mat, pose[:3])


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """
    将齐次变换矩阵转换为位姿 [x, y, z, rx, ry, rz]。
    
    Args:
        T: 4x4 齐次变换矩阵
    
    Returns:
        位姿 [x, y, z, rx, ry, rz]，旋转使用欧拉角 (xyz顺序)
    """
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    R_mat = R.from_matrix(T[:3, :3])
    pose[3:] = R_mat.as_euler("xyz", degrees=False)
    return pose


def transform_pose(pose: np.ndarray, T_transform: np.ndarray) -> np.ndarray:
    """
    使用变换矩阵转换位姿。
    
    Args:
        pose: 输入位姿 [x, y, z, rx, ry, rz]
        T_transform: 变换矩阵（例如 T_ref_in_base）
    
    Returns:
        转换后的位姿 [x, y, z, rx, ry, rz]
    """
    T_pose = pose_to_transform(pose)
    T_result = T_transform @ T_pose
    return transform_to_pose(T_result)


def transform_velocity(vel: np.ndarray, R_transform: np.ndarray) -> np.ndarray:
    """
    使用旋转矩阵转换速度（平移速度和角速度）。
    
    Args:
        vel: 速度 [vx, vy, vz, wx, wy, wz]
        R_transform: 旋转矩阵（例如 R_ref_in_base）
    
    Returns:
        转换后的速度 [vx, vy, vz, wx, wy, wz]
    """
    trans_vel = R_transform @ vel[:3]
    rot_vel = R_transform @ vel[3:]
    return np.concatenate([trans_vel, rot_vel])


def compute_base_ref_transform(
    tool_ref_pos: np.ndarray, base_frame_in_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ref 相对于 base 的变换矩阵和旋转矩阵。
    
    Args:
        tool_ref_pos: ref 相对于 world 的位姿 [x, y, z, rx, ry, rz]
        base_frame_in_world: base 相对于 world 的位姿 [x, y, z, rx, ry, rz]
    
    Returns:
        (T_ref_in_base, R_ref_in_base): ref 相对于 base 的齐次变换矩阵和旋转矩阵
    """
    T_base_in_world = pose_to_transform(base_frame_in_world)
    T_ref_in_world = pose_to_transform(tool_ref_pos)
    T_ref_in_base = inv_homogeneous(T_base_in_world) @ T_ref_in_world
    R_ref_in_base = T_ref_in_base[:3, :3]
    return T_ref_in_base, R_ref_in_base
