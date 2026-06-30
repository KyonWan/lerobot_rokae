# Rokae 录制说明

本文档聚焦数据采集流程（单臂 / 双臂，SpaceMouse / Pico）。安装与环境准备请先看仓库根 [README.md](README.md)。

## 前置准备

1. 激活环境

```bash
conda activate lerobot
```

2. 启动 Rokae 服务器（推荐 YAML 方式）

- 单臂：

```bash
cd rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/single.example.yaml
```

- 双臂：

```bash
cd rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/dual.example.yaml
```

3. 录制配置文件

- 单臂 SpaceMouse：`config/record/single_spacemouse_example.yaml`
- 双臂 SpaceMouse：`config/record/dual_spacemouse_example.yaml`
- 单臂 Pico：`config/record/single_pico_example.yaml`
- 双臂 Pico：`config/record/dual_pico_example.yaml`

> 录制侧无需重复配置 `joint_num`/`robot_ip`，由 ZMQ server 的 `get_robot_info()` 提供。

## 通用录制入口

统一使用：

```bash
./scripts/record/rokae_record.sh --config_path=<你的yaml>
```

例如：

```bash
./scripts/record/rokae_record.sh --config_path=config/record/single_spacemouse_example.yaml
./scripts/record/rokae_record.sh --config_path=config/record/dual_spacemouse_example.yaml
./scripts/record/rokae_record.sh --config_path=config/record/single_pico_example.yaml
./scripts/record/rokae_record.sh --config_path=config/record/dual_pico_example.yaml
```

## 单臂录制

### SpaceMouse（推荐）

```bash
./scripts/record/rokae_record.sh --config_path=config/record/single_spacemouse_example.yaml
```

常用覆盖参数示例：

```bash
./scripts/record/rokae_record.sh \
  --config_path=config/record/single_spacemouse_example.yaml \
  --dataset.root="./datasets/run_$(date +%Y%m%d_%H%M%S)" \
  --dataset.num_episodes=10
```

### Pico 单臂

使用 `teleop.type=pico_single`，推荐从示例 YAML 开始：

```bash
./scripts/record/rokae_record.sh --config_path=config/record/single_pico_example.yaml
```

`teleop.side` 选择左/右 Pico 手柄输入；`R_headset_world` 可按操作者站位配置，详见 [`config/record/README.md`](config/record/README.md)。

## 双臂录制

### 双 SpaceMouse

```bash
./scripts/record/rokae_record.sh --config_path=config/record/dual_spacemouse_example.yaml
```

关键参数：

- `--robot.type=bi_rokae_robot`
- `--robot.left_zmq_port=5555`
- `--robot.right_zmq_port=5556`
- `--teleop.type=bi_spacemouse`
- `--teleop.left_device_index=0`
- `--teleop.right_device_index=1`

### Pico 双臂

使用 `teleop.type=pico`，推荐从示例 YAML 开始：

```bash
./scripts/record/rokae_record.sh --config_path=config/record/dual_pico_example.yaml
```

Pico 双臂使用 `left_arm_controller` / `right_arm_controller` 显式配置“机械臂 -> Pico 手柄”的映射；面对机器人和同向站位的推荐配置见 [`config/record/README.md`](config/record/README.md)。

## 数据回放

```bash
lerobot-dataset-viz --repo-id <repo_id> --root <dataset_root> --episode-index 0
```

## 常见问题

- **双臂无响应**：确认左右臂 server 均已启动，且端口与录制配置一致（默认 5555/5556）。
- **SpaceMouse 设备混淆**：尝试交换 `left_device_index` 与 `right_device_index`。
- **Pico 无动作**：检查 XRoboToolkit PC 服务状态、头显与控制机网络是否同网段，并确认已勾选 `send`。
- **控制不稳定**：优先检查 `reset_joint_pos`、工具参数、控制模式是否与现场配置一致（`callback_mode` 已按 `control_mode` 自动推导）。
