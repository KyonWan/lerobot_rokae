# 安装

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `conda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

```bash
cd LerobotWithRokae/lerobot
```

```bash
pip install -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

# 数据采集
0. 运行conda 环境
```bash
conda activate lerobot
```

1. 启动rokae server
```bash
python -m rokae_python_wrapper.rokae_server
```

2. 启动lerobot数据采集脚本
```bash
python -m lerobot.scripts.lerobot_record `
    --robot.type=rokae_robot `
    --teleop.type=spacemouse `
    --dataset.repo_id=Rokae/lerobot_test_1 `
    --dataset.root="D:/ayyzw/Documents/CodeLibrary/Embodied Intelligence/Projects/LeRobotWithRokae/datasets" `
    --dataset.num_episodes=2 `
    --dataset.single_task="Grab the cube" `
    --dataset.push_to_hub=False `
    --display_data=true
```
若要添加相机，追加
```bash
--robot.cameras="{laptop: {type: intelrealsense, serial_number_or_name: 838212074037, width: 640, height: 480, fps: 60}}" `
```
3. 数据回放
```bat
lerobot-dataset-viz `
    --repo-id Rokae/lerobot_test_1 `
    --root "D:/ayyzw/Documents/CodeLibrary/Embodied Intelligence/Projects/LeRobotWithRokae/datasets" `
    --episode-index 0
```