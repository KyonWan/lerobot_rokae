#!/usr/bin/env bash
# 数据采集容器启动脚本（仓库根目录或任意路径执行均可）。
# 挂载宿主机 X11，使 pynput 键盘热键可用。
# 容器固定名称：lerobot_data_collect
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lerobot-rokae:data-collect}"
CONTAINER_NAME="${CONTAINER_NAME:-lerobot_data_collect}"

# 允许本机容器连 X（若已放行可忽略报错）
xhost +local: >/dev/null 2>&1 || true

XAUTH_SRC="${XAUTHORITY:-$HOME/.Xauthority}"
if [[ ! -f "${XAUTH_SRC}" ]]; then
  echo "警告: 找不到 XAUTHORITY (${XAUTH_SRC})，键盘热键可能不可用。" >&2
  XAUTH_ARGS=()
else
  XAUTH_ARGS=(
    -e "XAUTHORITY=/tmp/.docker.xauth"
    -v "${XAUTH_SRC}:/tmp/.docker.xauth:ro"
  )
fi

# 固定名字：若已有同名容器（含已退出），先删再建，便于反复 ./docker/run.sh
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[lerobot-rokae] 移除已有容器 ${CONTAINER_NAME}" >&2
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --network host \
  --privileged \
  -e DISPLAY \
  "${XAUTH_ARGS[@]}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  -v "${ROOT}:/workspace/lerobot_rokae" \
  -w /workspace/lerobot_rokae \
  "${IMAGE}" \
  "$@"
