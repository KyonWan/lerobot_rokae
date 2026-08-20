# Docker 数据采集镜像

精简镜像，仅用于 **Pico / SpaceMouse 遥操作 + RealSense 采集 + Rerun 可视化**。  
不含 CUDA、CUDA PyTorch，也不含 policy 训练依赖。

| 项 | 值 |
| --- | --- |
| 镜像名 | `lerobot-rokae:data-collect` |
| 容器名（固定） | `lerobot_data_collect` |
| 工作目录（容器内） | `/workspace/lerobot_rokae`（挂载本仓库） |
| Python 环境 | `/opt/lerobot`（已注入 `PATH`，无需 `activate`） |

## 构建

在仓库根目录执行（需已 `git submodule update` + `rokae_python_wrapper` 的 Git LFS）：

```bash
docker build -f docker/Dockerfile -t lerobot-rokae:data-collect .
```

默认使用国内镜像加速。海外环境可关掉镜像：

```bash
docker build -f docker/Dockerfile --build-arg NO_MIRROR=1 -t lerobot-rokae:data-collect .
```

未改动的层会自动复用 Docker cache；不要随意加 `--no-cache`。

## 启动容器

推荐用脚本（已配置 X11、设备、仓库挂载，容器名固定为 `lerobot_data_collect`）：

```bash
./docker/run.sh
# 或直接进 bash：
./docker/run.sh bash
```

若同名容器已存在（含上次未 `--rm` 残留），脚本会先 `docker rm -f` 再启动。

常用运维：

```bash
docker exec -it lerobot_data_collect bash   # 另开 shell
docker stop lerobot_data_collect            # 停止
```

可选环境变量：

- `IMAGE`：镜像 tag，默认 `lerobot-rokae:data-collect`
- `CONTAINER_NAME`：容器名，默认 `lerobot_data_collect`

### 说明

- `--network host` + `--privileged` + `-v /dev:/dev`：RealSense / HID / USB / adb
- 挂载宿主机 `DISPLAY` + `XAUTHORITY`：`pynput` 键盘热键、相机预览
- 无声卡：镜像内 `spd-say` 为空 stub；录制可加 `--play_sounds=false`
- Headless 也能录：不挂 X11 时靠 `episode_time_s` 或 Ctrl+C 结束

## 容器内：启动机器人 Server

在**另一个终端**进容器，或容器内开 tmux/screen：

```bash
# 单臂
cd /workspace/lerobot_rokae/rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/single.example.yaml

# 双臂
./scripts/rokae_run.sh --config config/server/dual.example.yaml
```

## SpaceMouse 录制

插好 SpaceMouse 后：

```bash
cd /workspace/lerobot_rokae
./scripts/record/rokae_record.sh --config_path=config/record/single_spacemouse_example.yaml
# 双臂：
./scripts/record/rokae_record.sh --config_path=config/record/dual_spacemouse_example.yaml
```

## Pico 录制

### 1. 启动 PC Service

```bash
pkill -f RoboticsServiceProcess || true
ss -ltnp | grep -E '63901|60061'
bash /opt/apps/roboticsservice/runService.sh
```

### 2. USB 连接与 adb reverse

用 USB-C 连 Pico。entrypoint / bashrc 会尽量自动执行 `adb reverse`；也可手动：

```bash
adb devices
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

头显上打开 XRoboToolkit，Enter 填 `127.0.0.1`，勾选 `head`、`controller`、`send`。

### 3. 录制

```bash
cd /workspace/lerobot_rokae
./scripts/record/rokae_record.sh --config_path=config/record/single_pico_example.yaml
# 双臂：
./scripts/record/rokae_record.sh --config_path=config/record/dual_pico_example.yaml
```

## 可视化（Rerun）

`display_data: true` 时走 Rerun。Docker + host 网络下，浏览器打开 Rerun Web Viewer（若脚本/环境启用了 Web 模式）或按终端提示连接。详细录制参数见仓库 [`RECORDING.md`](../RECORDING.md)。

## 目录说明

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 镜像定义 |
| `requirements-data-collect.txt` | 采集最小 pip 依赖 |
| `entrypoint.sh` | 启动提示 + 尝试 Pico adb reverse |
| `run.sh` | 宿主机一键启动固定名容器 |
| `.dockerignore` | 在**仓库根目录**（构建上下文是 `.`） |

## 相关文档

- [仓库 README](../README.md) — 安装与整体说明
- [RECORDING.md](../RECORDING.md) — 录制流程与排错
- [config/record/README.md](../config/record/README.md) — 示例 YAML
