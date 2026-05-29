# lerobot_teleoperator_rokae

LeRobot 遥操作插件：SpaceMouse、双臂 SpaceMouse、Pico / Pico Single 等与 Rokae 流程的集成。

## 系统依赖（Linux / SpaceMouse）

`pyspacemouse` 依赖 **HIDAPI** 的 C 库与头文件，**不能**用 pip 替代，需在系统包管理器中安装，例如 Debian/Ubuntu：

```bash
sudo apt-get update
sudo apt-get install -y libhidapi-dev libhidapi-hidraw0
```

安装后若仍无权限访问 `/dev/hidraw*`，可参考仓库根目录 `rokae_python_wrapper/README.md` 中的 udev 规则说明。

## Python 依赖（Pink / 逆解）

本包只负责 **遥操作设备输入**（SpaceMouse / Pico 等）；笛卡尔积分与逆解在 **`rokae_python_wrapper.rokae_kinematics`**（`cartesian_ik_solver`、`urdf_presets`、`action_fields` 等）：7 轴用 [Pink](https://github.com/stephane-caron/pink)，6 轴用 xCore `model.getJointPos`，不重复实现 IK 逻辑。

注意：在 PyPI 上请安装 **`pin-pink`**（`pip install pin-pink`），**不要**安装名为 `pink` 的 PyPI 包——那是另一个无关项目；安装后仍使用 `import pink`。

上述依赖已在本包与 `rokae_python_wrapper` 的 `pyproject.toml` 中声明；安装本仓库时请同时 **editable 安装** `rokae_python_wrapper`，否则运行时会缺少 `rokae_python_wrapper` 本身：

```bash
# 在仓库根目录，示例顺序（与根 README 一致即可）
pip install -e lerobot
pip install -e lerobot_robot_rokae
pip install -e rokae_python_wrapper
pip install -e lerobot_teleoperator_rokae
```

若运行时提示没有可用 QP 求解器，请确认已安装：

```bash
pip install "qpsolvers[open_source_solvers]"
```

## 与仓库其他包的关系

- **必选**：`lerobot`、`lerobot_robot_rokae`、`rokae_python_wrapper`（`rokae_kinematics` 运动学与 IK；robot 包的 `transform_utils` 为其 re-export）。
- **仅 Pico 遥操作**：另需按仓库根 `README.md` 安装 XRoboToolkit PC 服务与头显端应用。
