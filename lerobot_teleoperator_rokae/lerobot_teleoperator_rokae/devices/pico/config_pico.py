from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from lerobot.teleoperators.config import TeleoperatorConfig

if TYPE_CHECKING:
    from xrobotoolkit_teleop.common.xr_client import XrClient  # pyright: ignore[reportMissingImports]


@TeleoperatorConfig.register_subclass("pico")
@dataclass
class PicoConfig(TeleoperatorConfig):
    left_arm_controller: str = "right"
    right_arm_controller: str = "left"
    fps: float = 90.0                       # pico 数据更新频率   
    xr_client: Optional[Any] = None
    trigger_reverse: bool = True            # 是否反转原始 trigger 值
    trigger_threshold: float = 0.5          # trigger 二值化阈值
    close_position: float = 0.0
    open_position: float = 1.0
    xyz_scale_factor: float = 1           # 位置增量缩放因子
    rot_scale_factor: float = 0.25           # 姿态增量缩放因子
    R_headset_world: list[float] = field(default_factory=lambda: [90.0, 0.0, 90.0]) # pico 头显到世界坐标系的旋转矩阵，xyz Euler angles in degrees
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
    filter_enabled: bool = True             # 是否对 Pico 原始 delta pose 做滤波
    filter_window_sizes: list[int] | None = None  # None 时按 fps 从 rokae_server 的 [50, 50] 折算

    def __post_init__(self) -> None:
        valid_controllers = {"left", "right"}
        if self.left_arm_controller not in valid_controllers:
            raise ValueError(
                "PicoConfig.left_arm_controller must be 'left' or 'right', "
                f"got {self.left_arm_controller!r}"
            )
        if self.right_arm_controller not in valid_controllers:
            raise ValueError(
                "PicoConfig.right_arm_controller must be 'left' or 'right', "
                f"got {self.right_arm_controller!r}"
            )
        if self.left_arm_controller == self.right_arm_controller:
            raise ValueError(
                "PicoConfig does not allow both arms to use the same controller: "
                f"left_arm_controller={self.left_arm_controller!r}, "
                f"right_arm_controller={self.right_arm_controller!r}"
            )
