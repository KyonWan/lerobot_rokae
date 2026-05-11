#!/usr/bin/env bash
# Install / refresh udev rules so that any 3Dconnexion SpaceMouse
# (USB or Bluetooth Wireless) is accessible to the current user
# without sudo. Fixes "easyhid HIDException: Failed to open device".
#
# Usage:
#   ./scripts/setup/install_spacemouse_udev.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_RULE="${SCRIPT_DIR}/99-spacemouse.rules"
DST_RULE="/etc/udev/rules.d/99-spacemouse.rules"

if [[ ! -f "${SRC_RULE}" ]]; then
    echo "[ERROR] Rule file not found: ${SRC_RULE}" >&2
    exit 1
fi

echo "[1/4] Installing ${DST_RULE}"
sudo install -m 0644 "${SRC_RULE}" "${DST_RULE}"

echo "[2/4] Reloading udev rules"
sudo udevadm control --reload-rules

echo "[3/4] Triggering udev for hidraw subsystem"
sudo udevadm trigger --subsystem-match=hidraw --action=change

echo "[4/4] Fixing permissions on currently connected SpaceMouse hidraw nodes"
fixed=0
for dev in /dev/hidraw*; do
    [[ -e "${dev}" ]] || continue
    sysdev="/sys/class/hidraw/$(basename "${dev}")/device"
    if [[ -L "${sysdev}" ]]; then
        # The parent directory name looks like "0005:256F:C63A.000C"
        parent="$(basename "$(readlink -f "${sysdev}")")"
        if [[ "${parent}" =~ :256F: ]]; then
            sudo chmod 0666 "${dev}"
            echo "    -> chmod 0666 ${dev}  (${parent})"
            fixed=$((fixed + 1))
        fi
    fi
done

if [[ "${fixed}" -eq 0 ]]; then
    echo "    (no SpaceMouse currently connected)"
fi

echo
echo "Done. From now on, whenever a SpaceMouse is plugged in or paired,"
echo "the corresponding /dev/hidraw* node will be created with mode 0666."
