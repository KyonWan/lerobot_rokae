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
CAMERAS_CONFIG="{external: {type: intelrealsense, serial_number_or_name: '809512060572', \
width: 640, height: 480, fps: 60, use_depth: false}, \
wrist: {type: intelrealsense, serial_number_or_name: '125322062165', \
width: 640, height: 480, fps: 60, use_depth: false}}"

python -m lerobot.scripts.lerobot_record \
    --robot.type=rokae_robot \
    --robot.zmq_port=5555 \
    --robot.joint_num=7 \
    --robot.control_mode=cartesian_position \
    --robot.callback_mode=cart_vel \
    --teleop.type=spacemouse \
    --teleop.device_index=0 \
    --dataset.repo_id=test_2025/rokae_record \
    --dataset.root="/home/wanhao/Projects/lerobot_rokae/dataset/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=2 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=False \
    --display_data=True
    # --robot.cameras="$CAMERAS_CONFIG"
