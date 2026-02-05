#!/bin/bash
# 双SpaceMouse和双Rokae单臂机器人数据采集脚本
#
# 恢复录制说明：
# 如果录制中断，可以使用 bi_rokae_record_resume.sh 恢复录制
# 或者修改此脚本：
#   1. 添加 --resume=true 参数
#   2. 将 --dataset.root 改为上次录制的数据集路径（不要使用日期时间戳）
#   3. 将 --dataset.num_episodes 改为要额外录制的episode数量（不是总数）
# 示例：如果已录制50个episode，想再录制50个，设置 num_episodes=50

# 相机配置
CAMERAS_CONFIG="{external: {type: intelrealsense, serial_number_or_name: '125322062165', \
width: 640, height: 480, fps: 60, use_depth: false}, \
left_wrist: {type: intelrealsense, serial_number_or_name: '809512060572', \
width: 640, height: 480, fps: 60, use_depth: false}, \
right_wrist: {type: intelrealsense, serial_number_or_name: '036422060433', \
width: 640, height: 480, fps: 60, use_depth: false}}"

python -m lerobot.scripts.lerobot_record \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_joint_num=7 \
    --robot.right_joint_num=7 \
    --robot.left_control_mode=cartesian_position \
    --robot.left_callback_mode=cart_vel \
    --robot.right_control_mode=cartesian_position \
    --robot.right_callback_mode=cart_vel \
    --teleop.type=bi_spacemouse \
    --teleop.left_device_index=0 \
    --teleop.right_device_index=3 \
    --dataset.repo_id=test_2025/bi_rokae_record \
    --dataset.root="/home/wanhao/Projects/lerobot_rokae/dataset/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Use the left arm to place the two small joint modules into the two left blue boxes, and use the right arm to place the two large joint modules into the two right blue boxes." \
    --dataset.push_to_hub=False \
    --display_data=True
    # --robot.cameras="$CAMERAS_CONFIG"
