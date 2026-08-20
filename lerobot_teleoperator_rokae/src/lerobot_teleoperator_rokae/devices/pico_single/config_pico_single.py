from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from lerobot.teleoperators.config import TeleoperatorConfig

if TYPE_CHECKING:
    from xrobotoolkit_teleop.common.xr_client import XrClient  # pyright: ignore[reportMissingImports]


@TeleoperatorConfig.register_subclass("pico_single")
@dataclass
class PicoSingleConfig(TeleoperatorConfig):
    # 使用 str 而非 Literal：draccus CLI 解码不支持 Literal[...]
    side: str = "left"

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(f"PicoSingleConfig.side must be 'left' or 'right', got {self.side!r}")
    fps: float = 60.0                       # pico 数据更新频率   
    xr_client: Optional[Any] = None
    trigger_reverse: bool = True            # 是否反转原始 trigger 值
    trigger_threshold: float = 0.5          # trigger 二值化阈值
    close_position: float = 0.0
    open_position: float = 1.0
    xyz_scale_factor: float = 1           # 位置增量缩放因子比例
    rot_scale_factor: float = 0.5         # 姿态增量缩放因子比例
    R_headset_world: list[float] = field(default_factory=lambda: [90.0, 0.0, 90.0]) # pico 头显到世界坐标系的旋转矩阵，xyz Euler angles in degrees
    trans_max_vel: float = 0.1
    rot_max_vel: float = 0.2
    filter_enabled: bool = True             # 是否对 Pico 原始 delta pose 做滤波
    filter_window_sizes: list[int] | None = None  # None 时按 fps 从 rokae_server 的 [50, 50] 折算
