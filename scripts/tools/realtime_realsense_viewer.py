#!/usr/bin/env python3
"""Display all connected Intel RealSense color streams in one OpenCV window."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyrealsense2. Install it with:\n"
        '  pip install -e "./lerobot[intelrealsense]"\n'
        "or:\n"
        "  pip install pyrealsense2"
    ) from exc


@dataclass
class CameraStream:
    serial: str
    name: str
    pipeline: rs.pipeline
    last_frame: np.ndarray | None = None
    last_seen_s: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and display multiple Intel RealSense camera color streams."
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument("--window-name", default="RealSense Viewer", help="OpenCV window name.")
    parser.add_argument(
        "--rescan-interval",
        type=float,
        default=3.0,
        help="Seconds between device rescans.",
    )
    parser.add_argument(
        "--no-rescan",
        action="store_true",
        help="Search cameras only once at startup.",
    )
    parser.add_argument(
        "--poll-timeout-ms",
        type=int,
        default=1,
        help="Per-camera frame polling timeout in milliseconds.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=640,
        help="Displayed width for each camera tile.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=480,
        help="Displayed height for each camera tile.",
    )
    return parser.parse_args()


def list_realsense_devices() -> dict[str, str]:
    context = rs.context()
    devices: dict[str, str] = {}
    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices[serial] = name
    return devices


def start_camera(serial: str, name: str, width: int, height: int, fps: int) -> CameraStream | None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        pipeline.start(config)
    except RuntimeError as exc:
        print(f"[WARN] Failed to start {name} ({serial}): {exc}")
        return None

    print(f"[INFO] Started {name} ({serial}) at {width}x{height}@{fps}")
    return CameraStream(serial=serial, name=name, pipeline=pipeline, last_seen_s=time.time())


def stop_camera(camera: CameraStream) -> None:
    try:
        camera.pipeline.stop()
    except RuntimeError as exc:
        print(f"[WARN] Failed to stop {camera.name} ({camera.serial}): {exc}")
    else:
        print(f"[INFO] Stopped {camera.name} ({camera.serial})")


def sync_cameras(
    cameras: dict[str, CameraStream],
    width: int,
    height: int,
    fps: int,
) -> None:
    detected = list_realsense_devices()

    for serial in sorted(set(cameras) - set(detected)):
        camera = cameras.pop(serial)
        print(f"[INFO] Device removed: {camera.name} ({serial})")
        stop_camera(camera)

    for serial, name in sorted(detected.items()):
        if serial in cameras:
            continue
        camera = start_camera(serial, name, width, height, fps)
        if camera is not None:
            cameras[serial] = camera

    if not detected:
        print("[INFO] No RealSense cameras detected.")


def read_frames(cameras: dict[str, CameraStream], timeout_ms: int) -> None:
    for serial, camera in list(cameras.items()):
        try:
            frames = camera.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        camera.last_frame = np.asanyarray(color_frame.get_data())
        camera.last_seen_s = time.time()


def make_placeholder(text: str, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(image, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
    return image


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def build_grid(cameras: dict[str, CameraStream], tile_width: int, tile_height: int) -> np.ndarray:
    if not cameras:
        return make_placeholder("Searching for RealSense cameras...", tile_width, tile_height)

    tiles = []
    for index, camera in enumerate(cameras.values(), start=1):
        if camera.last_frame is None:
            tile = make_placeholder("Waiting for frames...", tile_width, tile_height)
        else:
            tile = cv2.resize(camera.last_frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        label = f"{index}: {camera.name} | {camera.serial}"
        tiles.append(label_frame(tile, label))

    cols = max(1, math.ceil(math.sqrt(len(tiles))))
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_images = []
    for row in range(rows):
        start = row * cols
        row_images.append(np.hstack(tiles[start : start + cols]))
    return np.vstack(row_images)


def main() -> int:
    args = parse_args()
    cameras: dict[str, CameraStream] = {}
    next_rescan_s = 0.0

    print("[INFO] Press 'q' or Esc in the OpenCV window to quit.")

    try:
        while True:
            now_s = time.time()
            should_rescan = not args.no_rescan and now_s >= next_rescan_s
            should_initial_scan = args.no_rescan and not cameras and next_rescan_s == 0.0

            if should_rescan or should_initial_scan:
                sync_cameras(cameras, args.width, args.height, args.fps)
                next_rescan_s = now_s + max(args.rescan_interval, 0.5)

            read_frames(cameras, max(args.poll_timeout_ms, 1))
            grid = build_grid(cameras, args.tile_width, args.tile_height)
            cv2.imshow(args.window_name, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        for camera in list(cameras.values()):
            stop_camera(camera)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
