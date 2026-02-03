# 双SpaceMouse控制双Rokae单臂机器人使用说明

本项目已扩展支持使用两个SpaceMouse同时控制两个Rokae单臂机器人，实现双臂遥操作数据采集。

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

### 3. 双Rokae单臂机器人
- 创建了 `BiRokaeRobot` 类（`bi_rokae_robot`）
- 包含两个独立的Rokae单臂机器人实例
- 配置文件：`BiRokaeRobotConfig`
  - `left_server_port`, `right_server_port`: 左右臂服务器端口（默认5000和5001）
  - `left_joint_num`, `right_joint_num`: 左右臂关节数
  - `left_control_mode`, `right_control_mode`: 左右臂控制模式
  - `left_callback_mode`, `right_callback_mode`: 左右臂回调模式

### 4. 数据处理器
- 创建了 `ExtractBiCartVelAndGripper` 处理器，用于处理双臂的笛卡尔速度和夹爪状态
- 创建了 `GenerateBiJointPosCmd` 处理器，用于生成双臂关节位置命令

### 5. lerobot_record支持
- 修改了 `lerobot_record.py` 以自动检测双臂系统并使用相应的处理器

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
# 左臂服务器（端口5000，机器人IP: 192.168.71.161）
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
    --port=5000 \
    --joint_num=7 \
    --end_effector=linkerhand_v10
```

**右臂服务器：**
```bash
python -m rokae_python_wrapper.rokae_server \
    --port=5001 \
    --joint_num=7 \
    --end_effector=linkerhand_v10
```

**命令行参数说明：**
- `--port`: Flask服务器端口（必须不同，默认5000和5001）
- `--joint_num`: 关节数量（6或7，根据实际机器人配置）
- `--end_effector`: 末端执行器类型（`linkerhand_v10` 或 `dahuan_gripper`）

**注意：** 
- 确保两个服务器的端口不同（左臂5000，右臂5001）

### 2. 连接SpaceMouse设备
确保两个SpaceMouse设备已正确连接到计算机。系统会自动检测设备索引0和1。

**注意：** 如果 `pyspacemouse` 库不支持通过索引打开多个设备，您可能需要：
1. 修改 `pyspacemouse` 库以支持多设备
2. 或者使用其他方法区分设备（如设备序列号）

## 使用方法

### 数据采集示例

使用 `scripts/bi_rokae_record.sh` 脚本进行数据采集：

```bash
chmod +x scripts/bi_rokae_record.sh
./scripts/bi_rokae_record.sh
```

或者直接使用命令行：

```bash
python -m lerobot.scripts.lerobot_record \
    --robot.type=bi_rokae_robot \
    --robot.left_server_port=5000 \
    --robot.right_server_port=5001 \
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

### 关键参数说明

- `--robot.type=bi_rokae_robot`: 使用双臂Rokae机器人
- `--teleop.type=bi_spacemouse`: 使用双SpaceMouse遥操作器
- `--robot.left_server_port=5000`: 左臂服务器端口
- `--robot.right_server_port=5001`: 右臂服务器端口
- `--teleop.left_device_index=0`: 左臂SpaceMouse设备索引
- `--teleop.right_device_index=1`: 右臂SpaceMouse设备索引

## 数据结构

### 动作（Action）特征
- `left_cart_vel0` 到 `left_cart_vel5`: 左臂笛卡尔速度（3个平移 + 3个旋转）
- `left_buttons`: 左臂SpaceMouse按钮状态
- `right_cart_vel0` 到 `right_cart_vel5`: 右臂笛卡尔速度
- `right_buttons`: 右臂SpaceMouse按钮状态

### 观测（Observation）特征
- `left_joint_pos0` 到 `left_joint_pos5`: 左臂关节位置
- `left_gripper_pos`: 左臂夹爪位置
- `right_joint_pos0` 到 `right_joint_pos5`: 右臂关节位置
- `right_gripper_pos`: 右臂夹爪位置
- 相机数据（如果配置）

## 注意事项

1. **服务器端口**: 确保两个Rokae服务器运行在不同的端口上（默认5000和5001）
2. **SpaceMouse设备**: 确保两个SpaceMouse设备正确连接，系统会自动分配设备索引
3. **控制模式**: 确保左右臂的控制模式和回调模式配置正确
5. **pyspacemouse多设备支持**: 如果遇到多设备问题，可能需要修改 `pyspacemouse` 库或使用其他方法区分设备

## 故障排除

1. **无法连接SpaceMouse**: 检查设备是否正确连接，尝试交换 `left_device_index` 和 `right_device_index`
2. **无法连接机器人**: 检查服务器是否在正确的端口运行
3. **动作不响应**: 检查控制模式和回调模式配置是否正确

## 文件结构

```
lerobot_teleoperator_rokae/
  └── lerobot_teleoperator_rokae/
      └── devices/
          ├── spacemouse/          # 单SpaceMouse（已修改支持设备索引）
          └── bi_spacemouse/       # 双SpaceMouse（新增）
              ├── __init__.py
              ├── config_bi_spacemouse.py
              └── bi_spacemouse.py

lerobot_robot_rokae/
  └── lerobot_robot_rokae/
      └── devices/
          ├── rokae_single_arm/    # 单臂机器人（已修改支持端口配置）
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
  ├── rokae_record.sh             # 单臂数据采集脚本
  └── rokae_record.bat            # 单臂数据采集脚本（Windows）
```
