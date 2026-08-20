#!/bin/bash
# Data-collect container entrypoint.
# - Audio: /usr/local/bin/spd-say is a no-op stub (no sound card needed).
# - Keyboard / camera preview: need host X11 at `docker run` time, e.g.
#     -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
#   Host: xhost +local:docker   (or xhost +local:)
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[lerobot-rokae] DISPLAY unset: recording works, but pynput keyboard/preview are headless." >&2
  echo "[lerobot-rokae] For hotkeys, re-run with: -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix" >&2
elif [[ ! -S "/tmp/.X11-unix/X${DISPLAY#*:}" ]] && [[ ! -e "/tmp/.X11-unix/X${DISPLAY#.}" ]]; then
  # Best-effort hint only; DISPLAY may be hostname:0 etc.
  echo "[lerobot-rokae] DISPLAY=${DISPLAY}; ensure /tmp/.X11-unix is mounted from the host." >&2
fi

# Pico USB reverse (idempotent; no-op if adb/device missing)
if [[ -x /usr/local/bin/setup_pico_adb_reverse.sh ]]; then
  /usr/local/bin/setup_pico_adb_reverse.sh || true
fi

exec "$@"
