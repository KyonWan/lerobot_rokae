# LeRobot with Rokae

LeRobot 框架的 Rokae 机器人集成，支持数据采集、训练和部署。

## 项目结构

- `lerobot/` - LeRobot 核心框架
- `lerobot_robot_rokae/` - Rokae 机器人设备集成
- `lerobot_teleoperator_rokae/` - SpaceMouse 遥操作设备集成
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

# 或直接使用命令行
python -m rokae_python_wrapper.rokae_zmq_server \
  --robot_ip=<你的机器人IP> \
  --host_ip=<你的主机IP> \
  --zmq_port=5555 \
  --zmq_transport=ipc \
  --joint_num=7 \
  --q_drag="-70,34,-64,105,50,0,-10" \
  --end_effector=linkerhand_v10 \
  --gripper_slave_addr=0x28
```

#### 3. 启动数据采集

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
- ⚠️ 所有参数（`--robot_ip`、`--host_ip`、`--q_drag`、`--end_effector` 等）都需要与你的硬件实际配置匹配。
- ⚠️ 使用自定义工具时，必须在代码中设置正确的工具信息（质量、质心、惯性张量等），否则可能导致位置控制偏差。

## 双臂机器人支持

本项目支持使用两个 SpaceMouse 同时控制两个 Rokae 单臂机器人，实现双臂遥操作数据采集。

详见：[双臂机器人使用说明](BI_ROKAE_README.md)

## 相关文档

- [双臂机器人使用说明](BI_ROKAE_README.md) - 双 SpaceMouse + 双 Rokae 配置
- [Rokae Python Wrapper 文档](rokae_python_wrapper/README.md)
- [LeRobot 官方文档](https://github.com/huggingface/lerobot)