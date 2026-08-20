# rokae_policy_runtime

Runtime bridges between **Rokae** hardware and policy servers (e.g. **OpenPI** over WebSocket). Other policy adapters can live in this package later.

## Install

Requires Python >= 3.10. This package is experimental/debug-only for now. It is not installed by the default `requirements.txt` and is not recommended for regular recording or teleoperation users.

From the **repository root**:

```bash
pip install -e ./rokae_policy_runtime --no-deps
```

`pyproject.toml` pulls in `lerobot`, `lerobot_robot_rokae`, `openpi-client`, and common numeric/image deps. If you develop `lerobot` / `lerobot_robot_rokae` from local checkouts, install those in editable mode first so imports resolve.

## OpenPI bridge (single-arm Rokae)

- **CLI:** `rokae-openpi` or `python -m rokae_policy_runtime.openpi.cli`
- **Details (ZMQ server, cameras, observation format):** [src/rokae_policy_runtime/openpi/README.md](src/rokae_policy_runtime/openpi/README.md)

```bash
rokae-openpi --help
```

## DexGraspVLA bridge (双臂 Rokae)

DexGraspVLA 运行在自己的 conda 环境里，启动一个兼容 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 的 WebSocket policy server；`lerobot_rokae` 侧只采集 Rokae 双臂状态和三路相机，通过 WebSocket 请求动作，不在本进程加载 DexGraspVLA 模型。

- **CLI:** `rokae-dexgraspvla` 或 `python -m rokae_policy_runtime.dexgraspvla.cli`

DexGraspVLA 服务端：

```bash
cd ~/Projects/DexGraspVLA
conda activate dexgraspvla
python scripts/serve_dexgrasp_websocket_policy.py \
  --host 0.0.0.0 \
  --port 8008 \
  --device cuda:0
```

Rokae bridge 侧：

```bash
# 8 维左臂模型：只需 external + left_wrist
rokae-dexgraspvla \
  --mode test \
  --dexgrasp_host 127.0.0.1 \
  --dexgrasp_port 8008 \
  --task_prompt "grasp the target object" \
  --cam_high_serial CP2G8530004K \
  --cam_left_wrist_serial 260322271562 \
  --left_zmq_port 5555 \
  --right_zmq_port 5556 \
  --action_dim 8 \
  --control_arm left

# 16 维双臂模型：需要再加 --cam_right_wrist_serial
# rokae-dexgraspvla ... --action_dim 16 --cam_right_wrist_serial 352122272829
```

确认日志里的 action 正常后，再把 `--mode test` 改成 `--mode autonomous`。
