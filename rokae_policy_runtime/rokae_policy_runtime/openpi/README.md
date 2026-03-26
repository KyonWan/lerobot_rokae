# Rokae 单臂 OpenPI 推理桥接

将 Rokae 单臂机器人与 [OpenPI](https://github.com/Physical-Intelligence/openpi) 策略服务器对接。实现位于本目录的 `bridge.py`（`OpenPIPolicyBridge`）与 `cli.py`（命令行入口）。

---

## 系统架构

```
┌─────────────────────────────────────┐       WebSocket        ┌──────────────────────────┐
│  rokae-openpi / python -m …openpi.cli │ ─────────────────────► │  OpenPI Policy Server    │
│                                     │ ◄─────────────────────  │  (pi0 / pi0-fast 等)     │
│  • 采集关节位置 & 图像               │   actions (chunk×dim)   └──────────────────────────┘
│  • 组装 observation                 │
│  • 执行 action                      │        ZMQ IPC/TCP
│                                     │ ◄─────────────────────  ┌──────────────────────────┐
└─────────────────────────────────────┘                         │  rokae_zmq_server        │
                                                                  │  （须提前单独启动）       │
                                                                  └──────────────────────────┘
```

---

## 前置依赖

### 1. 安装 Python 包

在仓库根目录执行（需已能 `import lerobot`）：

```bash
pip install -e rokae_policy_runtime
pip install -e lerobot_robot_rokae
```

`rokae_policy_runtime` 已声明依赖 `openpi-client`（WebSocket 客户端）；若环境未装齐，请按报错补装。

### 2. 启动 ZMQ 服务器（另开终端）

与 Rokae SDK 通信的 ZMQ 服务需**先于**桥接脚本启动，例如：

```bash
python -m rokae_python_wrapper.rokae_zmq_server \
    --robot_ip 192.168.2.180 \
    --host_ip 192.168.2.200 \
    --zmq_port 5555 \
    --zmq_transport ipc \
    --joint_num 6 \
    --q_drag 71,10,-114,0,-55,180
```

参数以你现场机器人与 `rokae_python_wrapper` 文档为准。

### 3. 启动 OpenPI 策略服务器

参考 [OpenPI 仓库说明](https://github.com/Physical-Intelligence/openpi) 启动策略服务（默认端口常见为 `8000`）。

---

## 快速开始

命令行入口（任选其一）：

```bash
rokae-openpi --help
python -m rokae_policy_runtime.openpi.cli --help
```

### 测试模式（不发送真实运动指令）

```bash
python -m rokae_policy_runtime.openpi.cli \
    --mode test \
    --task_prompt "Pick and place the part." \
    --cam_high_serial 809512060572 \
    --cam_wrist_serial 125322062165
```

### 自主执行模式

```bash
python -m rokae_policy_runtime.openpi.cli \
    --mode autonomous \
    --task_prompt "Pick and place the part." \
    --cam_high_serial 809512060572 \
    --cam_wrist_serial 125322062165 \
    --control_freq 30 \
    --max_steps 500
```

至少需要 `--cam_high_serial` 或 `--cam_wrist_serial` 之一（与代码中校验一致）。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy_host` | `localhost` | 策略服务器主机 |
| `--policy_port` | `8000` | 策略服务器端口 |
| `--mode` | `autonomous` | `autonomous` 真实执行，`test` 仅日志不运动 |
| `--task_prompt` | `move the arm to the left` | 传给策略的自然语言任务 |
| `--max_steps` | `10000` | 单次 Episode 最大步数 |
| `--control_freq` | `30` | 控制频率（Hz） |
| `--zmq_port` | `5555` | 本机 ZMQ 端口（与 ZMQ 服务一致） |
| `--zmq_address` | `None` | 若指定完整 ZMQ 地址，则优先于 `--zmq_port` |
| `--cam_high_serial` | `None` | 顶部 RealSense 序列号 → 观测里键名为 `external` |
| `--cam_wrist_serial` | `None` | 腕部 RealSense 序列号 → 观测里键名为 `wrist` |

---

## 观测 / 动作格式（与实现一致）

桥接发往 `infer()` 的 `observation` 为：

```python
{
    "state": np.ndarray,   # shape: (joint_num + 1,)，关节位置 + gripper
    "images": {
        "external": np.ndarray,  # 若配置了 cam_high_serial；shape (3, 224, 224) uint8 RGB
        "wrist": np.ndarray,     # 若配置了 cam_wrist_serial；同上
    },
    "prompt": str,
}
```

关节键来自 `RokaeRobot` 观测：`joint_pos0` … `joint_pos{n-1}` 与 `gripper_pos`（见 `OpenPIPolicyBridge._obs_to_state`）。

策略返回中使用：

```python
response["actions"]  # 通常为 (chunk_size, action_dim)；action_dim = joint_num + 1
```

首步会将预测位姿通过 `move_to_start_position` 平滑过渡；`test` 模式下 `execute_action` 不驱动真机。

**时序集成**：`OpenPIPolicyBridge` 内预留 `temporal_ensemble_coefficient`，默认为 `None`（关闭）；当前 CLI 未暴露该参数，需在代码中设置后使用。

---

## 与典型 Trossen（双臂）示例的差异（概念对照）

| 对比项 | 常见双臂示例 | 本仓库 Rokae 单臂 |
|--------|----------------|-------------------|
| 机器人配置 | 如 BiWidowX 等 | `RokaeRobotConfig` / `RokaeRobot` |
| 通信 | 依硬件而定 | ZMQ → `rokae_zmq_server` → SDK |
| 状态 / 动作维度 | 依机型 | `joint_num + 1`（含夹爪） |
| 图像键名 | 依示例 | `external` / `wrist`（非 `cam_high` 字面键） |

若策略侧对键名有固定约定，请在 OpenPI 侧 checkpoint / 数据配置与上述键名对齐。
