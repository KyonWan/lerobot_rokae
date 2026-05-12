import threading
import pyspacemouse
import numpy as np
from typing import Tuple
from easyhid import Enumeration


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
        self.device_index = int(device_index)

        supported_devices = self._list_supported_hid_devices()
        if not supported_devices:
            raise RuntimeError("No connected or supported SpaceMouse devices found.")

        if self.device_index < 0 or self.device_index >= len(supported_devices):
            raise ValueError(
                f"Invalid SpaceMouse index {self.device_index}. "
                f"Available indices are 0..{len(supported_devices) - 1}. "
                "This index maps to the supported HID device list order."
            )
        selected = supported_devices[self.device_index]
        print(
            f"SpaceMouse index={self.device_index} -> path={selected['path']} "
            f"product={selected['product']}"
        )

        # Open by hidraw path to avoid pyspacemouse same-model index fallback behavior.
        self.device = pyspacemouse.open_by_path(
            selected["path"],
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

    @staticmethod
    def _list_supported_hid_devices() -> list[dict]:
        hid = Enumeration()
        specs = pyspacemouse.get_supported_devices()
        supported_vid_pid = {(vendor_id, product_id) for _, vendor_id, product_id in specs}

        devices = []
        seen_paths = set()
        for hid_dev in hid.find():
            key = (hid_dev.vendor_id, hid_dev.product_id)
            if key not in supported_vid_pid:
                continue
            path = str(hid_dev.path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            devices.append(
                {
                    "path": path,
                    "product": hid_dev.product_string,
                    "vendor_id": hid_dev.vendor_id,
                    "product_id": hid_dev.product_id,
                }
            )
        return devices

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
