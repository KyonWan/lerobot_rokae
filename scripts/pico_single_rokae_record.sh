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
width: 640, height: 480, fps: 60, use_depth: false}}"

# rokae_algo 运动学参数（7 轴 cross_wrist7）
# rbv: 机器人描述参数，接口为米(m)，此处已由 mm 换算为 m
RBV_M="[0.0,0.0,0.0,0.0,0.0,0.1745,0.0,0.0,0.314,0.01,0.0,0.0,-0.01,0.0,0.272,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.097]"
# 关节限位（弧度），由角度换算：JOINT_RANGE_MIN/MAX_CUSTOMIZE 度 -> 弧度
MIN_JOINT_RAD="[-3.106686,-2.094395,-3.106686,-1.047198,-3.106686,-1.047198,-1.047198]"
MAX_JOINT_RAD="[3.106686,2.094395,3.106686,2.530727,3.106686,1.047198,1.047198]"

python -m lerobot.scripts.lerobot_record \
    --robot.type=rokae_robot \
    --robot.zmq_port=5555 \
    --robot.joint_num=7 \
    --robot.control_mode=joint_impedance \
    --robot.callback_mode=joint_pos \
    --robot.rbv="$RBV_M" \
    --robot.min_joint="$MIN_JOINT_RAD" \
    --robot.max_joint="$MAX_JOINT_RAD" \
    --teleop.type=pico_single \
    --teleop.side='right' \
    --teleop.R_headset_world='[90.0, 0.0, 180.0]' \
    --dataset.repo_id=test_2025/rokae_record \
    --dataset.root="/home/rx78/dataset/test_$(date +"%Y%m%d_%H%M%S")" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=False \
    --display_data=False
    # --robot.cameras="$CAMERAS_CONFIG"
