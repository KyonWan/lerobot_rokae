# Rokae OpenPI 异步推理与 RTC 运行说明

本文说明 `/home/wintersun/LeRobot/lerobot_rokae` 如何连接 `/home/wintersun/code/openpi` 的策略服务运行单臂和双臂 RTC。文中只写运行方式和参数关系，不写当前配置里的具体参数值。

## 1. 进程关系

完整运行至少有三类进程：

```text
Rokae ZMQ 服务  <----ZMQ----  lerobot_rokae bridge  <----WebSocket----  OpenPI 策略服务
```

单臂启动一个 Rokae ZMQ 服务，双臂启动左右两个 Rokae ZMQ 服务。OpenPI 策略服务负责模型推理和 RTC action prefix 缓存；`lerobot_rokae` 负责采集机器人状态/相机、发送 observation、接收 action chunk、按 RTC 时序切换 chunk。

## 2. 启动顺序

建议固定按这个顺序启动：

1. 启动 Rokae ZMQ 服务。
2. 启动 OpenPI 策略服务。
3. 启动单臂或双臂 bridge。


## 3. 启动 Rokae ZMQ 服务

先复制示例配置为本地配置，并按现场修改机器人 IP、控制机 IP、关节数、复位位姿、夹爪和端口。

单臂：

```bash
cd /home/wintersun/LeRobot/lerobot_rokae/rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/single.local.yaml
```

双臂：

```bash
cd /home/wintersun/LeRobot/lerobot_rokae/rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/dual.local.yaml
```

## 4. 启动 OpenPI 策略服务

OpenPI 的配置名必须和 checkpoint 的训练配置一致：单臂 checkpoint 用单臂配置，双臂 checkpoint 用双臂配置，Pi0.5 checkpoint 用 Pi0.5 配置。

```bash
cd /home/wintersun/code/openpi

OPENPI_CONFIG="<训练时使用的 OpenPI 配置名>"
CHECKPOINT_DIR="<checkpoint 目录>"
POLICY_PORT="<OpenPI 端口>"

uv run scripts/serve_policy.py \
  --port="${POLICY_PORT}" \
  policy:checkpoint \
  --policy.config="${OPENPI_CONFIG}" \
  --policy.dir="${CHECKPOINT_DIR}"
```

## 5. 单臂 RTC 运行

单臂 Python 入口是 `python3 -m rokae_policy_runtime.openpi.cli`，默认使用 `async_bridge`。如果 OpenPI metadata 声明 RTC 开启，bridge 会按 RTC 方式发送 `rtc_delay` 并按 delay 切换 action chunk。

```bash
cd /home/lerobot_rokae

POLICY_HOST="<OpenPI 机器 IP 或 hostname>"
POLICY_PORT="<OpenPI 端口>"
TASK_PROMPT="<任务描述>"
CONTROL_FREQ="<控制频率>"
H="<OpenPI action_horizon>"
R="<OpenPI rtc_rate_of_inference>"
D="<运行时 RTC delay 或 delay 上限>"
ZMQ_PORT="<单臂 ZMQ 端口>"
CAM_EXTERNAL="<顶部相机序列号>"
CAM_WRIST="<腕部相机序列号>"

python3 -m rokae_policy_runtime.openpi.cli \
  --bridge async_bridge \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --mode autonomous \
  --task_prompt "${TASK_PROMPT}" \
  --control_freq "${CONTROL_FREQ}" \
  --action_chunk_size "${H}" \
  --rate_of_inference "${R}" \
  --rtc_delay "${D}" \
  --zmq_port "${ZMQ_PORT}" \
  --cam_high_serial "${CAM_EXTERNAL}" \
  --cam_wrist_serial "${CAM_WRIST}"
```


如果希望固定 delay，不让 bridge 根据推理耗时动态估计，加入：

```bash
--disable_dynamic_delay
```

## 6. 双臂 RTC 运行

双臂 Python 入口是 `python3 -m rokae_policy_runtime.openpi.bi_cli`。这份代码没有 `--rtc_enabled` 开关；是否走 RTC 由 OpenPI metadata 和 `--rtc_delay` 决定。

如果已经执行过 `pip install -e ./rokae_policy_runtime`，也可以用短命令 `rokae-openpi-bi`，它等价于上面的 Python 入口。

```bash
cd /home/wintersun/LeRobot/lerobot_rokae

POLICY_HOST="<OpenPI 机器 IP 或 hostname>"
POLICY_PORT="<OpenPI 端口>"
TASK_PROMPT="<任务描述>"
CONTROL_FREQ="<控制频率>"
H="<OpenPI action_horizon>"
R="<OpenPI rtc_rate_of_inference>"
D="<运行时 RTC delay 或 delay 上限>"
LEFT_ZMQ_PORT="<左臂 ZMQ 端口>"
RIGHT_ZMQ_PORT="<右臂 ZMQ 端口>"
CAM_EXTERNAL="<顶部相机序列号>"
CAM_LEFT="<左腕相机序列号>"
CAM_RIGHT="<右腕相机序列号>"

python3 -m rokae_policy_runtime.openpi.bi_cli \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --mode autonomous \
  --task_prompt "${TASK_PROMPT}" \
  --control_freq "${CONTROL_FREQ}" \
  --action_chunk_size "${H}" \
  --rate_of_inference "${R}" \
  --rtc_delay "${D}" \
  --left_zmq_port "${LEFT_ZMQ_PORT}" \
  --right_zmq_port "${RIGHT_ZMQ_PORT}" \
  --cam_high_serial "${CAM_EXTERNAL}" \
  --cam_left_wrist_serial "${CAM_LEFT}" \
  --cam_right_wrist_serial "${CAM_RIGHT}" \
  --log_level INFO
```

## 7. 必须对齐的参数

| 参数 | `lerobot_rokae` 位置 | OpenPI 对应项 | 要求 |
| --- | --- | --- | --- |
| OpenPI 配置名 | OpenPI 启动命令 `--policy.config` | checkpoint 训练配置 | 必须和 checkpoint 类型一致 |
| checkpoint | OpenPI 启动命令 `--policy.dir` | 训练输出目录 | 必须属于同一个配置和数据规格 |
| `H` | `--action_chunk_size` | `model.action_horizon` | 必须一致 |
| `R` | `--rate_of_inference` | `policy_metadata.rtc_rate_of_inference` / policy 内部 `rate_of_inference` | 必须一致；OpenPI 用它从上一包 `[R:H]` 对齐 prefix |
| `D` | `--rtc_delay` 或 metadata `max_delay` | OpenPI policy 的 RTC delay 上限 | 运行时 delay 不应超过模型训练/服务端支持范围 |
| 控制频率 | `--control_freq` | 机器人真实控制循环频率 | 用于把推理耗时换算成控制步数，必须接近真实频率 |
| 相机键 | `--cam_*_serial` | 训练数据 image keys | 单臂/双臂相机数量和语义必须和训练一致 |
| 状态/动作维度 | bridge 从机器人读 state，执行 action | OpenPI data config 和 checkpoint | 单臂用单臂维度，双臂用双臂维度，不能混用 |
| ZMQ 地址 | `--zmq_*` | Rokae ZMQ YAML | 端口、transport、左右臂顺序必须一致 |
