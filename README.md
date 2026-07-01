# LeRobot Rokae

LeRobot 框架的 Rokae 机器人集成，默认支持 Rokae 机器人控制、SpaceMouse/Pico 遥操作和数据采集。Policy/runtime 相关代码仍处于调试/试验阶段，默认安装不会启用。

## 项目结构

- `lerobot/` - LeRobot 核心框架
- `lerobot_robot_rokae/` - Rokae 机器人设备集成
- `lerobot_teleoperator_rokae/` - SpaceMouse 和 Pico 遥操作设备集成
- `rokae_python_wrapper/` - Rokae Python SDK 封装（Git 子模块）
- `rokae_policy_runtime/` - policy runtime bridge（experimental/debug）
- `lerobot_policy_rokae/` - LeRobot policy 插件 scaffold（experimental/debug）
- `config/record/` - 录制示例配置

## 安装

### 0. 克隆仓库（如果尚未克隆）

如果尚未克隆仓库，可以使用 `--recursive` 选项一次性克隆所有子模块：

```bash
git clone --recursive <lerobot_rokae_repo_url>
cd lerobot_rokae
```

如果已经克隆了仓库但没有子模块，需要初始化子模块：

```bash
git submodule init
git submodule update
```

### 0.1 Git LFS（`rokae_python_wrapper` 子模块必需）

子模块 `rokae_python_wrapper` 中的 Rokae SDK 二进制（`*.so`）由 **Git LFS** 管理。未安装 LFS 或未拉取对象时，工作区里可能只有几行指针文本，运行会失败。

```bash
# 一次性（每台机器）
sudo apt-get install -y git-lfs    # 若系统尚未安装
git lfs install

# 克隆 / submodule update 之后，在子模块目录拉取大文件
cd rokae_python_wrapper
git lfs pull
cd ..
```

日常在子模块里 `git pull` 更新代码后，若 SDK `.so` 异常，同样在 `rokae_python_wrapper/` 下执行 `git lfs pull`。更多说明见 [rokae_python_wrapper/README.md](rokae_python_wrapper/README.md)。

### 1. 创建 Conda 环境

推荐显式使用 `conda-forge` 创建环境：

```bash
conda create -y -n lerobot -c conda-forge --override-channels --strict-channel-priority python=3.10 ffmpeg pip
conda activate lerobot
```

也可以使用仓库提供的 [`environment.yml`](environment.yml)：

```bash
conda env create -f environment.yml
```

本仓库的 Conda 配置只使用 `conda-forge`。

### 2. 一键安装（推荐）

在仓库根目录 `lerobot_rokae/` 执行（需 pip >= 21.2，建议先 `pip install -U pip`）：

```bash
pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

上述命令会通过 [`requirements.txt`](requirements.txt) 一次性 editable 安装默认公开组件：

- `lerobot[intelrealsense]`（含 RealSense 相机依赖）
- `rokae_python_wrapper`
- `lerobot_robot_rokae`
- `lerobot_teleoperator_rokae`

### 2.1 Experimental policy runtime（调试/试验阶段）

`rokae_policy_runtime` 和 `lerobot_policy_rokae` 当前仍处于调试/试验阶段，接口、依赖和使用方式可能变化；普通录制和遥操作用户不建议安装或依赖它们。

如果你正在开发 policy runtime，可额外执行：

```bash
pip install -r requirements-policy.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 3. 系统依赖（SpaceMouse / 视频）

使用 SpaceMouse（`pyspacemouse`）时，Linux 上需要系统级 **HIDAPI**（pip 无法提供），请先安装：

```bash
sudo apt-get update
sudo apt-get install -y libhidapi-dev libhidapi-hidraw0
```

更多录制与遥操作说明见 [RECORDING.md](RECORDING.md)。

### 4. 分步安装 / 仅 wrapper（高级，可跳过）

> 仅在**不使用第 2 步一键安装**时执行本节。  
> 如果你已经执行了第 2 步，请直接跳过，避免重复安装。

若不用默认 requirements，或只需单独开发某一子包，可分别 editable 安装：

```bash
pip install -e "./lerobot[intelrealsense]"
pip install -e ./rokae_python_wrapper
pip install -e ./lerobot_robot_rokae
pip install -e ./lerobot_teleoperator_rokae
```

policy runtime 调试/试验模块如需单独开发，可执行 `pip install -r requirements-policy.txt`。

`rokae_python_wrapper` 亦可单独 clone 后在项目根 `pip install -e .`（见 [rokae_python_wrapper/README.md](rokae_python_wrapper/README.md)）。

### 5. 安装 [XRoboToolkit](https://github.com/XR-Robotics) （仅当使用 Pico 遥操作时需要）

XRoboToolkit 组件不随本仓库分发；请从 XR-Robotics 官方仓库或 release 获取，并遵循其对应许可证和第三方声明。

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
- 在控制机器人的电脑端，双击应用XRoboToolkit-PC-Service的图标或通过以下命令打开服务：
  ```
  /opt/apps/roboticsservice/runService.sh
  ```
- 推荐使用 USB-C 有线连接，数据流更稳定：
  ```bash
  adb devices
  adb reverse tcp:63901 tcp:63901
  adb reverse --list
  ```
  在 Pico 头显上打开 XRoboToolkit，在应用界面的 Enter 处输入 `127.0.0.1`，勾选 `head`、`controller` 和 `send`。
- 也可以使用 Wi-Fi：确保控制机器人的电脑和 Pico 头显处于同一网络下，在 XRoboToolkit 中输入控制机器人的电脑 IP，并勾选 `head`、`controller` 和 `send`。


## 数据采集

录制相关说明已独立到 [`RECORDING.md`](RECORDING.md)，包含：

- 单臂 / 双臂启动与录制
- SpaceMouse / Pico 录制示例
- `config/record/*.yaml` 用法；示例按 teleop 类型拆分，见 [`config/record/README.md`](config/record/README.md)
- 常见问题与排错

快速开始（单臂 SpaceMouse）：

```bash
cd rokae_python_wrapper
./scripts/rokae_run.sh --config config/server/single.example.yaml
cd ..
./scripts/record/rokae_record.sh --config_path=config/record/single_spacemouse_example.yaml
```

## 相关文档

- [录制说明](RECORDING.md) - 单臂/双臂、SpaceMouse/Pico 录制流程
- [Rokae Python Wrapper 文档](rokae_python_wrapper/README.md)
- [LeRobot 官方文档](https://github.com/huggingface/lerobot)

## License

本仓库中由 ROKAE Robotics 开发的代码以 [Apache License 2.0](LICENSE) 授权。使用、修改和分发时请保留版权、许可证和 [NOTICE](NOTICE) 中的署名信息。

第三方组件或单独授权组件仍遵循各自许可证或供应商协议，包括 LeRobot、Rokae SDK 原生库、XRoboToolkit、Python 依赖、模型权重、数据集和系统包。

本项目可能调用用户环境中的 FFmpeg 以及 H.264/AVC、HEVC、AV1 等视频编码器。ROKAE Robotics 不随本仓库分发这些编码器二进制，也不代使用者取得相关 codec 专利或第三方许可证；商业分发、产品集成、公开发布视频数据或再分发运行环境时，请使用者自行确认 FFmpeg 许可证和 codec 专利合规。
