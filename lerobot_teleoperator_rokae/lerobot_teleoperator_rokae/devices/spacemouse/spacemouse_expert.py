import threading
import pyspacemouse
import numpy as np
from typing import Tuple, Optional


class SpaceMouseExpert:
    """
    Interface for one SpaceMouse device.
    Each instance controls exactly ONE SpaceMouse.
    """

    def __init__(self, device_index: int):
        """
        Args:
            device_index: device_index for pyspacemouse (0, 1, ...)
        """
        self.device_index = device_index

        # ⚠️ 关键：保存 device 对象
        self.device = pyspacemouse.open(
            device_index=device_index,
            nonblocking=True,
        )

        if self.device is None:
            raise RuntimeError(f"Failed to open SpaceMouse {device_index}")

        self.state_lock = threading.Lock()
        self.latest_data = {
            "action": np.zeros(6),
            "buttons": [0, 0],
        }

        self.thread = threading.Thread(
            target=self._read_spacemouse,
            daemon=True,
        )
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = self.device.read()   # ✅ 设备级 read
            if state is None:
                continue

            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [
                        -state.y,
                        state.x,
                        state.z,
                        -state.roll,
                        -state.pitch,
                        -state.yaw,
                    ],
                    dtype=np.float32,
                )
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self.state_lock:
            return (
                self.latest_data["action"].copy(),
                list(self.latest_data["buttons"]),
            )
