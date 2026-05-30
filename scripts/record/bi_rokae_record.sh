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
DATASET_VCODEC="h264"

# 相机配置
# external 设为 640x480 时，rokae_record_plugin 会跳过 Python 裁切+resize（降低 obs_proc；需全幅画面请在相机端或改分辨率）
# 单核 taskset 时写盘线程过多会与主循环抢 CPU，加重 Slow loop；每相机 1 个写线程通常更稳。
# （若录制进程可占多核且希望更快清空队列，可适当增大该值。）
CAMERAS_CONFIG="{external: {type: orbbec, serial_number_or_index: 'CP2G8530004K', \
width: 640, height: 480, fps: 60, use_depth: false}, \
left_wrist: {type: intelrealsense, serial_number_or_name: '260322271562', \
width: 640, height: 480, fps: 60, use_depth: false}, \
right_wrist: {type: intelrealsense, serial_number_or_name: '352122272829', \
width: 640, height: 480, fps: 60, use_depth: false}}"

# --- Teleop / IK（6 轴 xCore model；7 轴 Pink 自动解析 URDF）---
# rokae_record_plugin 在创建 pipeline 时对各臂调用 get_robot_info().type，
# 在 rokae_python_wrapper/rokae_kinematics/rokae_urdf/ 下查找 {type}.urdf，
# 并从 URDF 解析末端 link（优先 *_tcp，其次 *_flan_link）。找不到机型 URDF 会报错。
# 请保证 ZMQ server 已连接且 robotInfo.type 与 rokae_urdf 内文件名一致（如 AR5-5_07L-W4C4A2）。
# joint_num / robot_ip 由 ZMQ server 提供，无需在此脚本重复配置。
# ------------------------------------------------------------------------------------

taskset -c 9 python -m lerobot.scripts.lerobot_record \
    --resume=False \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_control_mode=joint_position \
    --robot.left_callback_mode=joint_pos \
    --robot.right_control_mode=joint_position \
    --robot.right_callback_mode=joint_pos \
    --teleop.type=bi_spacemouse \
    --teleop.left_device_index=0 \
    --teleop.right_device_index=1 \
    --dataset.repo_id=test_2025/bi_rokae_record \
    --dataset.root="/home/wanhao/Documents/datasets/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=300 \
    --dataset.single_task="Put the two white parts into the gray box." \
    --dataset.push_to_hub=False \
    --display_data=False 
    # --robot.cameras="$CAMERAS_CONFIG"
