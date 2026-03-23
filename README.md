# LeRobot with Rokae

LeRobot 框架的 Rokae 机器人集成，支持数据采集、训练和部署。

## 项目结构

- `lerobot/` - LeRobot 核心框架
- `lerobot_robot_rokae/` - Rokae 机器人设备集成
- `lerobot_teleoperator_rokae/` - SpaceMouse 和 Pico 遥操作设备集成
- `rokae_python_wrapper/` - Rokae Python SDK 封装（Git 子模块）

## 安装

### 0. 克隆仓库（如果尚未克隆）

如果尚未克隆仓库，可以使用 `--recursive` 选项一次性克隆所有子模块：

```bash
git clone --recursive git@gitlab.i.rokae.com:embodied_ai_group/lerobot_rokae.git
cd lerobot_rokae
```

如果已经克隆了仓库但没有子模块，需要初始化子模块：

```bash
git submodule init
git submodule update
```

### 1. 创建 Conda 环境

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

### 2. 安装 ffmpeg（用于视频处理）

```bash
conda install ffmpeg -c conda-forge
```

### 3. 安装 LeRobot

```bash
cd lerobot
pip install -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
cd ..
```

### 4. 安装 Rokae 机器人集成

```bash
pip install -e lerobot_robot_rokae
pip install -e lerobot_teleoperator_rokae
```

### 5. 安装 Rokae Python Wrapper

```bash
cd rokae_python_wrapper
pip install -e .
cd ..
```
### 6. 安装 [XRoboToolkit](https://github.com/XR-Robotics) （仅当使用 Pico 遥操作时需要）


#### 安装 XRoboToolkit PC 服务
- 下载适用于 [Ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb)/ [Ubuntu 24.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb) 的 .deb 安装包，或从[源码仓库](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)自行构建。
- 安装命令：
  Ubuntu 22.04:
  ```
  sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
  ```
  Ubuntu 24.04:
  ```
  sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_24.04_amd64.deb
  ```
- 随后在 lerobot_rokae 之外的文件夹完成以下安装：
  ```bash
  git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
  cd XRoboToolkit-Teleop-Sample-Python
  bash setup_conda.sh --install
  ```

#### 在Pico 4U头显设备上安装XR app
- 打开Pico 4U的[开发者模式](https://developer.picoxr.com/ja/document/unreal/test-and-build/)，确保电脑安装了[adb](https://developer.android.com/tools/adb)。
- 在装有adb的电脑上下载apk文件[XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)。
- 使用adb命令安装apk：
  ```
  adb install -g XRoboToolkit-PICO-1.1.1.apk
  ```

#### 使用Pico采集数据前的必要操作
- 确保控制机器人的电脑和Pico头显处于同一网络下。
- 在控制机器人的电脑端，双击应用XRoboToolkit-PC-Service的图标或通过以下命令打开服务：
  ```
  /opt/apps/roboticsservice/runService.sh
  ```
- 在Pico头显上打开应用XRoboToolkit，在应用界面的Enter处输入控制机器人的电脑的IP，勾选以下方框：head，controller和send。


## 数据采集

### 前置准备

1. **配置机器人参数**：编辑 `rokae_python_wrapper/scripts/start_rokae_left_server.sh`，设置正确的 `--robot_ip`、`--host_ip`、`--q_drag` 等参数。

2. **设置工具信息**：如果使用自定义工具（如夹爪），需要在 `rokae_python_wrapper/rokae_server.py` 中修改 `RokaeServer` 类的工具信息常量。详见 `rokae_python_wrapper/README.md`。

### 采集步骤

#### 1. 激活环境

```bash
conda activate lerobot
```

#### 2. 启动 Rokae 服务器

```bash
# 使用启动脚本（推荐）
cd rokae_python_wrapper
./scripts/start_rokae_left_server.sh

# 或直接使用命令行（无夹爪时省略 --gripper_address；有夹爪时需先启动 rokae_gripper_server）
python -m rokae_python_wrapper.rokae_zmq_server \
  --robot_ip=<你的机器人IP> \
  --host_ip=<你的主机IP> \
  --zmq_port=5555 \
  --zmq_transport=ipc \
  --joint_num=7 \
  --q_drag="-70,34,-64,105,50,0,-10" \
  --gripper_address=ipc:///tmp/rokae_gripper_5557
```

#### 3. 启动数据采集

**使用 SpaceMouse 进行单臂数据采集：**

使用 `scripts/rokae_record.sh` 脚本进行数据采集：

```bash
chmod +x scripts/rokae_record.sh
./scripts/rokae_record.sh
```

或者直接使用命令行：

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=rokae_robot \
  --teleop.type=spacemouse \
  --dataset.repo_id=Rokae/lerobot_test_1 \
  --dataset.root="./datasets" \
  --dataset.num_episodes=2 \
  --dataset.single_task="Grab the cube" \
  --dataset.push_to_hub=False \
  --display_data=true
```

**常用可选参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--display_data` | 是否在 Rerun 中显示采集数据 | `false` |
| `--log_slow_loop_periodically` | 是否每秒打印一次控制循环耗时（用于监控帧率稳定性） | `false` |
**使用 Pico 进行单臂数据采集：**

使用 `scripts/pico_single_rokae_record.sh` 脚本进行数据采集：

```bash
chmod +x scripts/pico_single_rokae_record.sh
./scripts/pico_single_rokae_record.sh
```

或者直接使用命令行：

```bash
python -m lerobot.scripts.lerobot_record \
    --robot.type=rokae_robot \
    --robot.zmq_port=5555 \
    --robot.joint_num=7 \
    --robot.control_mode=joint_impedance \
    --robot.callback_mode=joint_pos \
    --robot.rbv="$RBV_M" \
    --robot.min_joint="$MIN_JOINT_RAD" \
    --robot.max_joint="$MAX_JOINT_RAD" \
    --teleop.type=pico_single \
    --teleop.side='right' \
    --teleop.R_headset_world='[90.0, 0.0, 180.0]' \
    --dataset.repo_id=test_2025/rokae_record \
    --dataset.root="/home/rx78/dataset/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=False \
    --display_data=False
```

**常用可选参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--teleop.side` | 使用哪个手柄控制单个机械臂 | `right` |
| `--teleop.R_headset_world` | 头显设备到世界坐标系的旋转矩阵（xyz内旋欧拉角表示），根据佩戴头显设备的操作者的站位和世界坐标系的设定自行修改配置 | `[90.0, 0.0, 180.0]` |

**添加相机支持**（可选）：

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=rokae_robot \
  --teleop.type=spacemouse \
  --robot.cameras="{laptop: {type: intelrealsense, serial_number_or_name: 838212074037, width: 640, height: 480, fps: 60}}" \
  --dataset.repo_id=Rokae/lerobot_test_1 \
  --dataset.root="./datasets" \
  --dataset.num_episodes=2 \
  --dataset.single_task="Grab the cube" \
  --dataset.push_to_hub=False \
  --display_data=true
```

#### 4. 数据回放

```bash
lerobot-dataset-viz \
  --repo-id Rokae/lerobot_test_1 \
  --root "./datasets" \
  --episode-index 0
```

## 注意事项

- ⚠️ 服务器启动后会自动运动到 `--q_drag` 指定的关节角度，务必确保设置正确，无碰撞风险。
- ⚠️ 所有参数（`--robot_ip`、`--host_ip`、`--q_drag`、`--gripper_address` 等）都需要与你的硬件实际配置匹配。
- ⚠️ 使用自定义工具时，必须在代码中设置正确的工具信息（质量、质心、惯性张量等），否则可能导致位置控制偏差。

## 双臂机器人支持

本项目支持使用两个 SpaceMouse 和 Pico 同时控制两个 Rokae 单臂机器人，实现双臂遥操作数据采集。

详见：[双臂机器人使用说明](BI_ROKAE_README.md)

## 相关文档

- [双臂机器人使用说明](BI_ROKAE_README.md) - 双 SpaceMouse + 双 Rokae 配置
- [Rokae Python Wrapper 文档](rokae_python_wrapper/README.md)
- [LeRobot 官方文档](https://github.com/huggingface/lerobot)