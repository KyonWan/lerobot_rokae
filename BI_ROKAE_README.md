# 双臂 Rokae 遥操作数据采集说明（SpaceMouse / Pico）

本项目已支持以下遥操作数采模式：
- 双 SpaceMouse 控制双 Rokae 单臂机器人（双臂）
- Pico 控制双 Rokae 单臂机器人（双臂）

> 📖 **单臂机器人使用说明**：如需单臂机器人配置，请参考 [README.md](README.md)

## 主要修改内容

### 1. SpaceMouse支持多设备
- 修改了 `SpaceMouseExpert` 以支持设备索引参数
- 修改了 `SpacemouseConfig` 添加 `device_index` 参数

### 2. 双SpaceMouse遥操作器
- 创建了 `BiSpacemouse` 类（`bi_spacemouse`）
- 支持两个SpaceMouse设备，分别控制左臂和右臂
- 配置文件：`BiSpacemouseConfig`
  - `left_device_index`: 左臂SpaceMouse设备索引（默认0）
  - `right_device_index`: 右臂SpaceMouse设备索引（默认1）

### 3. Pico 遥操作支持
- 创建了双臂 Pico 遥操作器：`Pico`（`teleop.type=pico`）
- 配置文件：`PicoConfig`，主要配置变量：
  - `fps`： pico 端数据更新频率
  - `xyz_scale_factor`：位置增量缩放因子
  - `rot_scale_factor`：姿态增量缩放因子
  - `R_headset_world`：pico 头显到世界坐标系的旋转矩阵，顺序为ZYX的内旋欧拉角，pico 头显坐标系详见：https://github.com/XR-Robotics/XRoboToolkit-PC-Service

### 4. 双Rokae单臂机器人
- 创建了 `BiRokaeRobot` 类（`bi_rokae_robot`）
- 包含两个独立的Rokae单臂机器人实例
- 配置文件：`BiRokaeRobotConfig`
  - `left_server_port`, `right_server_port`: 左右臂服务器端口（用于推断ZMQ端口：5000->5555, 5001->5556）
  - `left_joint_num`, `right_joint_num`: 左右臂关节数
  - `left_control_mode`, `right_control_mode`: 左右臂控制模式
  - `left_callback_mode`, `right_callback_mode`: 左右臂回调模式

### 5. 数据处理器
- 创建了 `ExtractBiCartVelAndGripper` 处理器，用于处理使用双 spacemouse 遥操作时双臂的笛卡尔速度和夹爪状态
- 创建了`PicoBiInverseKinematicsProcessor`处理器，用于处理 pico 遥操作时返回的笛卡尔坐标增量
- 创建了 `GenerateBiJointPosCmd` 处理器，用于生成双臂关节位置命令

### 6. lerobot_record 支持
- 修改了 `lerobot_record.py` 以自动检测双臂系统并使用相应的处理器（ bi_spacemouse 或 pico ）

## 使用前准备

### 1. 启动Rokae服务器
每个单臂机器人需要运行一个独立的服务器实例，使用不同的端口和机器人IP。

**方法一：使用提供的启动脚本（推荐）**

**同时启动两个服务器：**
```bash
chmod +x rokae_python_wrapper/scripts/start_bi_rokae_servers.sh
./rokae_python_wrapper/scripts/start_bi_rokae_servers.sh
```
这会在后台启动两个服务器，日志输出到 `rokae_left_server.log` 和 `rokae_right_server.log`。

**分别启动：**
```bash
# 左臂服务器（ZMQ端口5555，机器人IP: 192.168.71.161）
chmod +x rokae_python_wrapper/scripts/start_rokae_left_server.sh
./rokae_python_wrapper/scripts/start_rokae_left_server.sh

# 右臂服务器（端口5001，机器人IP: 192.168.71.160）
chmod +x rokae_python_wrapper/scripts/start_rokae_right_server.sh
./rokae_python_wrapper/scripts/start_rokae_right_server.sh
```

**方法二：使用命令行参数**

`rokae_server.py` 现在支持命令行参数，可以灵活配置：

**左臂服务器：**
```bash
python -m rokae_python_wrapper.rokae_server \
    --robot_ip=192.168.71.161 \
    --host_ip=192.168.71.230 \
    --zmq_port=5555 \
    --zmq_transport=ipc \
    --joint_num=7 \
    --end_effector=linkerhand_v10
```

**右臂服务器：**
```bash
python -m rokae_python_wrapper.rokae_server \
    --robot_ip=192.168.71.160 \
    --host_ip=192.168.71.230 \
    --zmq_port=5556 \
    --zmq_transport=ipc \
    --joint_num=7 \
    --end_effector=linkerhand_v10
```

**命令行参数说明：**
- `--robot_ip`: 机器人IP地址（必需）
- `--host_ip`: 主机IP地址（必需）
- `--zmq_port`: ZMQ服务器端口（左臂5555，右臂5556）
- `--zmq_transport`: ZMQ传输协议（`tcp` 跨网络，`ipc` 本地更快）
- `--joint_num`: 关节数量（6或7，根据实际机器人配置）
- `--end_effector`: 末端执行器类型（`linkerhand_v10`、`dahuan_gripper` 或 `none`）

**注意：** 
- 确保两个服务器的ZMQ端口不同（左臂5555，右臂5556）
- IPC模式仅支持Unix/Linux，Windows会自动使用TCP模式

### 2. 连接 SpaceMouse 设备（双 SpaceMouse 模式）
确保两个SpaceMouse设备已正确连接到计算机。系统会自动检测设备索引0和1。

**注意：** 如果 `pyspacemouse` 库不支持通过索引打开多个设备，您可能需要：
1. 修改 `pyspacemouse` 库以支持多设备
2. 或者使用其他方法区分设备（如设备序列号）

### 3. 安装 XRoboToolkit PC 并连接 Pico 设备（Pico 模式）
- 准备工作：参照[Lerobot with Rokae](README.md)完成对 XRoboToolkit 的安装。
- 使用说明：完成准备工作后，在头显上打开 app XRoboToolkit，在enter处连接控制机器人电脑的IP，勾选 head，hand，controller，send（注意：控制机器人的电脑需要和pico 4 U 头显在同一个网络环境下）。
- 默认控制映射：
  - 双臂模式 `teleop.type=pico`：左手柄控制左臂，右手柄控制右臂。

## 使用方法

### 数据采集示例

#### 1) 双 SpaceMouse + 双臂数采

使用 `scripts/bi_rokae_record.sh` 脚本进行数据采集：

```bash
chmod +x scripts/bi_rokae_record.sh
./scripts/bi_rokae_record.sh
```

或者直接使用命令行：

```bash
python -m lerobot.scripts.lerobot_record \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_joint_num=7 \
    --robot.right_joint_num=7 \
    --robot.left_control_mode=joint_position \
    --robot.left_callback_mode=joint_pos \
    --robot.right_control_mode=joint_position \
    --robot.right_callback_mode=joint_pos \
    --teleop.type=bi_spacemouse \
    --teleop.left_device_index=0 \
    --teleop.right_device_index=1 \
    --dataset.repo_id=test_2025/bi_rokae_record \
    --dataset.root="/home/rokae/Projects/datasets" \
    --dataset.num_episodes=2 \
    --dataset.single_task="Bimanual manipulation task" \
    --dataset.push_to_hub=False \
    --display_data=true \
    --robot.cameras="{external: {type: intelrealsense, serial_number_or_name: 809512060572, width: 640, height: 480, fps: 60}, wrist: {type: intelrealsense, serial_number_or_name: 125322062165, width: 640, height: 480, fps: 60}}"
```

#### 2) Pico + 双臂数采

使用 `scripts/pico_rokae_record.sh` 脚本：

```bash
chmod +x scripts/pico_rokae_record.sh
./scripts/pico_rokae_record.sh
```

或者直接使用命令行：

```bash
python -m lerobot.scripts.lerobot_record \
    --resume=False \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_joint_num=7 \
    --robot.right_joint_num=7 \
    --robot.left_control_mode=joint_position \
    --robot.left_callback_mode=joint_pos \
    --robot.right_control_mode=joint_position \
    --robot.right_callback_mode=joint_pos \
    --robot.left_rbv="$LEFT_RBV_M" \
    --robot.left_min_joint="$LEFT_MIN_JOINT_RAD" \
    --robot.left_max_joint="$LEFT_MAX_JOINT_RAD" \
    --robot.right_rbv="$RIGHT_RBV_M" \
    --robot.right_min_joint="$RIGHT_MIN_JOINT_RAD" \
    --robot.right_max_joint="$RIGHT_MAX_JOINT_RAD" \
    --teleop.type=pico \
    --dataset.repo_id=test_2026/bi_rokae_record \
    --dataset.root="/home/rokae/Projects/datasets" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Use the left arm to place the two small joint modules into the two left blue boxes, and use the right arm to place the two large joint modules into the two right blue boxes." \
    --dataset.push_to_hub=False \
    --display_data=False
```

### 关键参数说明

- `--robot.type=bi_rokae_robot`: 使用双臂Rokae机器人
- `--teleop.type=bi_spacemouse`: 使用双SpaceMouse遥操作器
- `--robot.left_zmq_port=5555`: 左臂ZMQ端口
- `--robot.right_zmq_port=5556`: 右臂ZMQ端口
- `--teleop.type=bi_spacemouse`: 使用双 SpaceMouse双臂遥操作
- `--teleop.left_device_index=0`: 左臂SpaceMouse设备索引
- `--teleop.right_device_index=1`: 右臂SpaceMouse设备索引
- `--teleop.type=pico`: 使用 Pico 双臂遥操作
- `--teleop.xyz_scale_factor`: Pico 位置增量缩放比例
- `--teleop.rot_scale_factor`: Pico 姿态增量缩放比例

## 数据结构

### 处理前动作（Action）特征
**双 SpaceMouse 模式（`teleop.type=bi_spacemouse`）**
- `left_cart_vel0` 到 `left_cart_vel5`: 左臂笛卡尔速度（3个平移 + 3个旋转）
- `left_buttons`: 左臂SpaceMouse按钮状态
- `right_cart_vel0` 到 `right_cart_vel5`: 右臂笛卡尔速度
- `right_buttons`: 右臂SpaceMouse按钮状态

**Pico 双臂模式（`teleop.type=pico`）**
- `left_target_x/y/z`、`left_target_wx/wy/wz`、`left_gripper_pos`：左臂位置、旋转增量，左臂夹爪位置
- `right_target_x/y/z`、`right_target_wx/wy/wz`、`right_gripper_pos`：右臂位置、旋转增量，右臂夹爪位置

### 处理后动作（Action）特征
- `left_joint_pos0` 到 `left_joint_pos5`: 左臂关节位置
- `left_gripper_pos`: 左臂夹爪位置
- `right_joint_pos0` 到 `right_joint_pos5`: 右臂关节位置
- `right_gripper_pos`: 右臂夹爪位置
- `left_cart_vel0` 到 `left_cart_vel5`: 左臂笛卡尔速度（仅双 SpaceMouse 收集到的数据集中包含）
- `right_cart_vel0` 到 `right_cart_vel5`: 右臂笛卡尔速度（仅双 SpaceMouse 收集到的数据集中包含）

### 观测（Observation）特征
- `left_joint_pos0` 到 `left_joint_pos6`: 左臂关节位置
- `left_cart_pos0` 到 `left_cart_pos5`: 左臂笛卡尔位置（末端相对于工件坐标系）
- `left_gripper_pos`: 左臂夹爪位置
- `right_joint_pos0` 到 `right_joint_pos6`: 右臂关节位置
- `right_cart_pos0` 到 `right_cart_pos5`: 右臂笛卡尔位置（末端相对于工件坐标系）
- `right_gripper_pos`: 右臂夹爪位置
- 相机数据（如果配置）

## 注意事项

1. **ZMQ端口**: 确保两个Rokae服务器运行在不同的ZMQ端口上（默认5555和5556），并在配置中正确设置
2. **SpaceMouse设备**: 仅双 SpaceMouse 模式需要，确保两个设备正确连接
3. **控制模式**: 确保左右臂的控制模式和回调模式配置正确
4. **Pico 控制映射**: 默认 grip 用于激活位姿跟踪，trigger 用于夹爪开合切换，控制机械臂需要按下 grip 键
5. **pyspacemouse多设备支持**: 如果遇到多设备问题，可能需要修改 `pyspacemouse` 库或使用其他方法区分设备

## 故障排除

1. **无法连接SpaceMouse**: 检查设备是否正确连接，尝试交换 `left_device_index` 和 `right_device_index`
2. **机械臂未响应 Pico 发送的数据且无任何报错**: 检查 Pico 和机械臂的通讯是否正常，尝试在头显的app XRoboToolkit 中点击  reconnect 按钮
3. **无法连接机器人**: 检查服务器是否在正确的端口运行
4. **动作不响应**: 检查控制模式和回调模式配置是否正确

## 文件结构

```
lerobot_teleoperator_rokae/
  └── lerobot_teleoperator_rokae/
      └── devices/
          ├── spacemouse/          # 单SpaceMouse（已修改支持设备索引）
          ├── pico_single/         # Pico 单臂遥操作（新增）
          ├── bi_spacemouse/       # 双SpaceMouse（新增）
              ├── __init__.py
              ├── config_bi_spacemouse.py
              ├── bi_spacemouse_processor.py
              └── bi_spacemouse.py
          └── pico/                # Pico 双臂遥操作（新增）
              ├── __init__.py
              ├── config_pico.py
              ├── pico_processor.py
              └── pico.py

lerobot_robot_rokae/
  └── lerobot_robot_rokae/
      └── devices/
          ├── rokae_robot/    # 单臂机器人（已修改支持端口配置）
          └── bi_rokae_robot/      # 双臂机器人（新增）
              ├── __init__.py
              ├── config_bi_rokae_robot.py
              ├── bi_rokae_robot.py
              └── bi_rokae_processor.py

rokae_python_wrapper/
  └── scripts/                     # 启动脚本
      ├── start_bi_rokae_servers.sh    # 启动双臂服务器（Linux/macOS）
      ├── start_bi_rokae_servers.bat   # 启动双臂服务器（Windows）
      ├── start_rokae_left_server.sh   # 启动左臂服务器（Linux/macOS）
      ├── start_rokae_left_server.bat   # 启动左臂服务器（Windows）
      ├── start_rokae_right_server.sh  # 启动右臂服务器（Linux/macOS）
      └── start_rokae_right_server.bat # 启动右臂服务器（Windows）

scripts/                           # 项目特定脚本
  ├── bi_rokae_record.sh          # 双臂数据采集脚本
  ├── bi_rokae_record.bat         # 双臂数据采集脚本（Windows）
  ├── bi_rokae_record_resume.sh   # 恢复录制脚本
  ├── pico_rokae_record.sh        # Pico 双臂数据采集脚本
  ├── rokae_record.sh             # 单臂数据采集脚本
  └── rokae_record.bat            # 单臂数据采集脚本（Windows）
```
