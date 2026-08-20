# LeRobot Rokae 数据采集镜像（Ubuntu 22.04, CPU-only）
#
# 用途：Pico / SpaceMouse 遥操作 + RealSense 采集 + rerun 可视化。
# 不含：CUDA toolkit、CUDA PyTorch、policy 训练依赖（diffusers/transformers/jax 等）。
#
# 无声卡 / 无桌面说明：
#   - 语音：镜像内 /usr/local/bin/spd-say 为空实现，无需挂载声卡。
#   - 键盘停录 / 相机预览（pynput）：无法单靠镜像“装出”图形界面，必须把宿主机
#     X11 挂进容器（见下方 docker run）。不挂也能录，只是 headless，靠 episode_time_s
#     自动结束或 Ctrl+C。
#
# 本地包用 --no-deps -e 安装（仅注册元数据，供插件发现）；第三方依赖见
# docker/requirements-data-collect.txt。运行时把仓库挂到 /workspace/lerobot_rokae。
#
# 构建（在仓库根目录，需含子模块与 LFS 的 .so）:
#   docker build -f docker/Dockerfile -t lerobot-rokae:data-collect .
#
# 官方 PyPI（海外）:
#   docker build -f docker/Dockerfile --build-arg NO_MIRROR=1 -t lerobot-rokae:data-collect .
#
# 运行（推荐，容器名固定 lerobot_data_collect）:
#   ./docker/run.sh
# 说明见 docker/README.md
#
# Pico PC Service（容器内）:
#   bash /opt/apps/roboticsservice/runService.sh
#   # entrypoint / bashrc 会尝试 adb reverse tcp:63901

ARG TARGETARCH=amd64
ARG UBUNTU_VER=22.04
ARG UV_VERSION=0.8.4
ARG NO_MIRROR=
ARG PICO_PC_SERVICE_RELEASE=v1.0.0
ARG BASE_IMAGE=ubuntu:${UBUNTU_VER}

FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG UV_VERSION
ARG NO_MIRROR
ARG PICO_PC_SERVICE_RELEASE
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/opt/lerobot \
    PATH=/opt/lerobot/bin:/usr/local/bin:/root/.local/bin:${PATH} \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_TORCH_BACKEND=cpu \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XRT_SRC=/opt/xrobotoolkit/src \
    LD_LIBRARY_PATH=/opt/xrobotoolkit/lib:/usr/local/lib \
    PICO_PC_SERVICE_APP_PATH=/opt/apps/roboticsservice \
    PICO_ADB_REVERSE_PORT=63901 \
    NO_MIRROR=${NO_MIRROR} \
    QT_X11_NO_MITSHM=1

# ---------------------------------------------------------------------------
# apt 源
# ---------------------------------------------------------------------------
RUN if [ "$NO_MIRROR" != "1" ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
            /etc/apt/sources.list; \
    fi

# ---------------------------------------------------------------------------
# 系统包（无 CUDA；含 X11 客户端库，供挂载宿主机 DISPLAY 时 pynput 使用）
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        unzip \
        udev \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libhidapi-dev \
        libhidapi-hidraw0 \
        libusb-1.0-0 \
        libusb-1.0-0-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        android-tools-adb \
        xdg-utils \
        desktop-file-utils \
        libgtk-3-bin \
        iproute2 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxi6 \
        libxtst6 \
        libxkbcommon0 \
        xauth \
    && rm -rf /var/lib/apt/lists/*

# 无声卡：空 spd-say（PATH 优先 /usr/local/bin），避免 play_sounds 调真 TTS 失败
RUN printf '%s\n' \
        '#!/bin/sh' \
        '# No audio device in typical data-collect containers.' \
        'exit 0' \
        > /usr/local/bin/spd-say \
    && chmod +x /usr/local/bin/spd-say

# SpaceMouse udev
RUN printf '%s\n' \
        '# 3Dconnexion SpaceMouse udev rules' \
        'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"' \
        'KERNEL=="hidraw*", KERNELS=="0005:256F:*", MODE="0666"' \
        'KERNEL=="hidraw*", KERNELS=="????:256F:*", MODE="0666"' \
        > /etc/udev/rules.d/99-spacemouse.rules

# GitHub 代理前缀（留空则直连）
RUN if [ "$NO_MIRROR" != "1" ]; then \
        echo -n "https://ghfast.top/" > /etc/github_prefix; \
    else \
        : > /etc/github_prefix; \
    fi

ENV PIP_INDEX_URL=${NO_MIRROR:+https://pypi.org/simple}
ENV PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
ENV PIP_TRUSTED_HOST=${NO_MIRROR:+pypi.org}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}
ENV UV_DEFAULT_INDEX=${PIP_INDEX_URL}
ENV TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# uv + venv
# ---------------------------------------------------------------------------
RUN GITHUB_PREFIX="$(cat /etc/github_prefix)" \
    && curl -fsSL \
        "${GITHUB_PREFIX}https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        -o /tmp/uv.tar.gz \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu \
    && uv --version

RUN uv venv "${VIRTUAL_ENV}" --python 3.10 \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" -U pip setuptools wheel

# ---------------------------------------------------------------------------
# XRoboToolkit PC Service（.deb；参考 RLinf，仅 amd64）
# ---------------------------------------------------------------------------
RUN GITHUB_PREFIX="$(cat /etc/github_prefix)" \
    && set -eux \
    && if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "Skip XRoboToolkit PC Service deb on ${TARGETARCH} (amd64-only)." >&2; \
    else \
        VER="${PICO_PC_SERVICE_RELEASE#v}"; \
        downloaded=0; \
        for DEB in \
            "XRoboToolkit_PC_Service_${VER}_ubuntu_22.04_amd64.deb" \
            "XRoboToolkit-PC-Service-${VER}.deb"; do \
            DIRECT_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/${PICO_PC_SERVICE_RELEASE}/${DEB}"; \
            MIRROR_URL="${GITHUB_PREFIX}${DIRECT_URL}"; \
            if curl -fL --retry 3 --retry-delay 2 "${DIRECT_URL}" -o "/tmp/${DEB}"; then \
                downloaded=1; break; \
            fi; \
            if [ -n "${GITHUB_PREFIX}" ] && curl -fL --retry 3 --retry-delay 2 "${MIRROR_URL}" -o "/tmp/${DEB}"; then \
                downloaded=1; break; \
            fi; \
        done; \
        if [ "${downloaded}" -ne 1 ]; then \
            echo "Failed to download XRoboToolkit PC Service deb ${PICO_PC_SERVICE_RELEASE}" >&2; \
            exit 1; \
        fi; \
        apt-get update; \
        apt-get install -y --no-install-recommends "/tmp/${DEB}"; \
        rm -f "/tmp/${DEB}"; \
        rm -rf /var/lib/apt/lists/*; \
        test -x "${PICO_PC_SERVICE_APP_PATH}/runService.sh"; \
    fi

# ---------------------------------------------------------------------------
# Pico SDK（xrobotoolkit_sdk）+ teleop 包（--no-deps）
# 不做完整 Teleop editable 的传递依赖（跳过 mujoco 等）
# ---------------------------------------------------------------------------
RUN mkdir -p "${XRT_SRC}" /opt/xrobotoolkit/lib

# 克隆 Pybind（带重试）
RUN GITHUB_PREFIX="$(cat /etc/github_prefix)" \
    && set -eux \
    && for attempt in 1 2 3 4 5; do \
        rm -rf "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind"; \
        if git clone --depth 1 \
            "${GITHUB_PREFIX}https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git" \
            "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind"; then \
            break; \
        fi; \
        if [ -n "${GITHUB_PREFIX}" ] && git clone --depth 1 \
            "https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git" \
            "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind"; then \
            break; \
        fi; \
        sleep $((attempt * 3)); \
        if [ "${attempt}" -eq 5 ]; then exit 1; fi; \
    done

# 用 release tarball 拉 PC-Service 源码（比 git clone 稳）
RUN GITHUB_PREFIX="$(cat /etc/github_prefix)" \
    && set -eux \
    && mkdir -p "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service" \
    && DIRECT_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service/archive/refs/tags/${PICO_PC_SERVICE_RELEASE}.tar.gz" \
    && MIRROR_URL="${GITHUB_PREFIX}${DIRECT_URL}" \
    && if ! curl -fL --retry 5 --retry-delay 3 "${MIRROR_URL}" \
            | tar -xz --strip-components=1 \
                -C "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service"; then \
        curl -fL --retry 5 --retry-delay 3 "${DIRECT_URL}" \
            | tar -xz --strip-components=1 \
                -C "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service"; \
    fi \
    && test -f "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build.sh"

WORKDIR ${XRT_SRC}/XRoboToolkit-PC-Service-Pybind

RUN bash tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build.sh

RUN mkdir -p lib include \
    && cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/ \
    && cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/ \
    && cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/ \
    && cp lib/libPXREARobotSDK.so /opt/xrobotoolkit/lib/ \
    && cp lib/libPXREARobotSDK.so /usr/local/lib/ \
    && ldconfig

# Pico 编译期依赖 + 安装 SDK
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" "pybind11[global]" numpy \
    && export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}" \
    && export pybind11_DIR="$("${VIRTUAL_ENV}/bin/python" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    && "${VIRTUAL_ENV}/bin/python" -m pip uninstall -y xrobotoolkit_sdk || true \
    && "${VIRTUAL_ENV}/bin/python" setup.py install

# teleop 包仅装包本身（无 torch/mujoco 传递依赖）
WORKDIR ${XRT_SRC}
RUN GITHUB_PREFIX="$(cat /etc/github_prefix)" \
    && git clone --depth 1 \
        "${GITHUB_PREFIX}https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git" \
    || git clone --depth 1 \
        "https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git"

WORKDIR ${XRT_SRC}/XRoboToolkit-Teleop-Sample-Python
# 上游 pyproject 只写了 packages=["xrobotoolkit_teleop"]，非 editable 安装会丢掉 common/ 等子包。
# 用 -e 指向源码树，保证 import xrobotoolkit_teleop.common 可用；--no-deps 跳过 mujoco/torch 等。
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps -e .

# 清理编译中间产物与 PC-Service 源码（保留 Teleop 源码，供 editable 引用）
WORKDIR /
RUN rm -rf \
        "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/tmp" \
        "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind/build" \
        "${XRT_SRC}/XRoboToolkit-PC-Service-Pybind"/*.egg-info \
    && "${VIRTUAL_ENV}/bin/python" -c \
        "import xrobotoolkit_sdk; from xrobotoolkit_teleop.common.xr_client import XrClient; print('pico python deps ok', XrClient)"

# Pico USB：交互 shell 自动 adb reverse
RUN cat <<'EOF' > /usr/local/bin/setup_pico_adb_reverse.sh
#!/bin/bash
port="${PICO_ADB_REVERSE_PORT:-63901}"
command -v adb >/dev/null 2>&1 || exit 0
adb start-server >/dev/null 2>&1 || true
if adb reverse "tcp:${port}" "tcp:${port}" >/dev/null 2>&1; then
    echo "[pico-adb] reverse tcp:${port} tcp:${port}"
else
    echo "[pico-adb] skip reverse tcp:${port} (no device / enable USB debugging)" >&2
fi
EOF
RUN chmod +x /usr/local/bin/setup_pico_adb_reverse.sh \
    && echo '/usr/local/bin/setup_pico_adb_reverse.sh' >> /root/.bashrc

# ---------------------------------------------------------------------------
# CPU torch + 采集最小 pip 依赖
# ---------------------------------------------------------------------------
COPY docker/requirements-data-collect.txt /tmp/requirements-data-collect.txt

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url "${TORCH_INDEX}" \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        -r /tmp/requirements-data-collect.txt \
    && rm -f /tmp/requirements-data-collect.txt \
    && uv cache prune \
    && rm -rf /root/.cache/uv /root/.cache/pip

# ---------------------------------------------------------------------------
# 本仓库：--no-deps -e（插件元数据 + egg-link；依赖已在上一层装好）
# ---------------------------------------------------------------------------
ENV REPO_ROOT=/workspace/lerobot_rokae

COPY lerobot ${REPO_ROOT}/lerobot
COPY rokae_python_wrapper ${REPO_ROOT}/rokae_python_wrapper
COPY lerobot_robot_rokae ${REPO_ROOT}/lerobot_robot_rokae
COPY lerobot_teleoperator_rokae ${REPO_ROOT}/lerobot_teleoperator_rokae

WORKDIR ${REPO_ROOT}

RUN SDK_SO="rokae_python_wrapper/src/rokae_python_wrapper/rokae_sdk/Release/linux/xCoreSDK_python.cpython-310-x86_64-linux-gnu.so" \
    && test -f "${SDK_SO}" \
    && ! grep -q 'git-lfs.github.com/spec/v1' "${SDK_SO}" \
    && echo "rokae sdk .so ok"

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps \
        -e ./lerobot \
        -e ./rokae_python_wrapper \
        -e ./lerobot_robot_rokae \
        -e ./lerobot_teleoperator_rokae \
    && uv cache prune \
    && rm -rf /root/.cache/uv

RUN "${VIRTUAL_ENV}/bin/python" -c "\
import lerobot; import lerobot.datasets; \
import rokae_python_wrapper; import lerobot_robot_rokae; import lerobot_teleoperator_rokae; \
import torch; import pyrealsense2; import pyspacemouse; import rerun; \
import xrobotoolkit_sdk; from xrobotoolkit_teleop.common.xr_client import XrClient; \
print('data-collect smoke ok', torch.__version__, 'cuda=', torch.cuda.is_available(), XrClient)"

# 删除构建期源码；运行时挂载覆盖同一路径
RUN rm -rf "${REPO_ROOT}" && mkdir -p "${REPO_ROOT}"

COPY docker/entrypoint.sh /usr/local/bin/lerobot-rokae-entrypoint.sh
RUN chmod +x /usr/local/bin/lerobot-rokae-entrypoint.sh

WORKDIR /workspace/lerobot_rokae
ENTRYPOINT ["/usr/local/bin/lerobot-rokae-entrypoint.sh"]
CMD ["/bin/bash"]
