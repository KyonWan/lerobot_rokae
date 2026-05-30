#!/bin/bash
# 单SpaceMouse和单Rokae单臂机器人数据采集脚本
#
# 恢复录制说明：
# 如果录制中断，可以使用 rokae_record_resume.sh 恢复录制
# 或者修改此脚本：
#   1. 添加 --resume=true 参数
#   2. 将 --dataset.root 改为上次录制的数据集路径（不要使用日期时间戳）
#   3. 将 --dataset.num_episodes 改为要额外录制的episode数量（不是总数）
# 示例：如果已录制50个episode，想再录制50个，设置 num_episodes=50

# 相机配置
# CAMERAS_CONFIG="{external: {type: intelrealsense, serial_number_or_name: '809512060572', \
# width: 640, height: 480, fps: 60, use_depth: false}}"
DATASET_VCODEC="h264"

# 相机配置
# external 设为 640x480 时，rokae_record_plugin 会跳过 Python 裁切+resize（降低 obs_proc；需全幅画面请在相机端或改分辨率）
# cpu_core 要求录制进程允许的 CPU 集合包含这些核（勿再用 taskset -c 0 单核，否则线程无法绑到 5/6/7）
CAMERAS_CONFIG="{left_wrist: {type: intelrealsense, serial_number_or_name: '260322271849', \
width: 640, height: 480, fps: 60, use_depth: false}}"

# CAMERAS_CONFIG="{external: {type: orbbec, serial_number_or_index: 'CP2G85300022', \
# width: 1280, height: 720, fps: 60, use_depth: false, cpu_core: 5}, \
# left_wrist: {type: intelrealsense, serial_number_or_name: '260322274865', \
# width: 640, height: 480, fps: 60, use_depth: false, cpu_core: 6}}"

taskset -c 0 python -m lerobot.scripts.lerobot_record \
    --robot.type=rokae_robot \
    --robot.zmq_port=5555 \
    --robot.control_mode=joint_position \
    --robot.callback_mode=joint_pos \
    --teleop.type=spacemouse \
    --teleop.device_index=0 \
    --dataset.vcodec="$DATASET_VCODEC" \
    --dataset.repo_id=test_2025/rokae_record \
    --dataset.root="/home/wanhao/Documents/datasets/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=False \
    --log_slow_loop_periodically=True \
    --display_data=False \
    --robot.cameras="$CAMERAS_CONFIG"
