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

# rokae_algo 运动学参数（7 轴 cross_wrist7）
# 左臂：直接沿用单臂 rokae_record.sh 中的配置（单位已是 m / rad）
LEFT_RBV_M="[0.0,0.0,0.0,0.0,0.0,0.1745,0.0,0.0,0.314,0.01,0.0,0.0,-0.01,0.0,0.272,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.097]"
LEFT_MIN_JOINT_RAD="[-3.106686,-2.094395,-3.106686,-1.047198,-3.106686,-1.047198,-1.047198]"
LEFT_MAX_JOINT_RAD="[3.106686,2.094395,3.106686,2.530727,3.106686,1.047198,1.047198]"

# 右臂：根据 ROBOT_DIMENSIONS（单位 mm）和 JOINT_RANGE_*_CUSTOMIZE（单位 deg）换算
# ROBOT_DIMENSIONS:
#   [0.0,0.0,0.0,0.0,0.0,174.5,0.0,0.0,314.0,10.0,0.0,0.0,-10.0,0.0,272.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,97.0]
#   将长度部分除以1000得到 m，与 LEFT_RBV_M 数值一致：
RIGHT_RBV_M="[0.0,0.0,0.0,0.0,0.0,0.1745,0.0,0.0,0.314,0.01,0.0,0.0,-0.01,0.0,0.272,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.097]"

# JOINT_RANGE_MIN_CUSTOMIZE: [-178, -120, -178, -60, -178, -60, -60]
# JOINT_RANGE_MAX_CUSTOMIZE: [178, 120, 178, 145, 178, 60, 60]
# 换算成弧度后同样与左臂一致：
RIGHT_MIN_JOINT_RAD="[-3.106686,-2.094395,-3.106686,-1.047198,-3.106686,-1.047198,-1.047198]"
RIGHT_MAX_JOINT_RAD="[3.106686,2.094395,3.106686,2.530727,3.106686,1.047198,1.047198]"

# --- Teleop / Pink 双臂 IK（kinematics_preset，rokae_record_plugin + pink_ik_helpers）---
# --teleop.kinematics_preset
#   wheeled_ar_dual    固定式双臂：默认 URDF 为包内 AR5-5_07L / 07R，末端 link 为 *_tcp。
#   fixed_ar_dual  轮式/整机双臂：默认末端为 08* 系列的 *_flan_link；须配合「整机」URDF。
# 环境变量（未写 --teleop.left/right_urdf_path 时由 default_rokae_urdf_path_* 读取）：
#   ROKAE_IK_URDF_PATH_LEFT / ROKAE_IK_URDF_PATH_RIGHT 覆盖默认 .urdf 路径。
# 可选 CLI 覆盖（写上则覆盖预设里解析出的路径/末端名；必须与对应 URDF 里 <link name> 一致）：
#   --teleop.left_urdf_path / right_urdf_path
#   --teleop.left_end_effector_frame / right_end_effector_frame
# 注意：仓库内 07L/07R 单机描述只有 *_tcp，没有 *_flan_link；用单机 URDF 时应用 fixed_ar_dual，
#       或像下面这样显式写 *_tcp 与 07 urdf（不要写 URDF 中不存在的 link 名）。
# ------------------------------------------------------------------------------------

taskset -c 9 python -m lerobot.scripts.lerobot_record \
    --resume=False \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_joint_num=7 \
    --robot.right_joint_num=7 \
    --robot.left_control_mode=joint_impedance \
    --robot.left_callback_mode=joint_pos \
    --robot.right_control_mode=joint_impedance \
    --robot.right_callback_mode=joint_pos \
    --robot.left_rbv="$LEFT_RBV_M" \
    --robot.left_min_joint="$LEFT_MIN_JOINT_RAD" \
    --robot.left_max_joint="$LEFT_MAX_JOINT_RAD" \
    --robot.right_rbv="$RIGHT_RBV_M" \
    --robot.right_min_joint="$RIGHT_MIN_JOINT_RAD" \
    --robot.right_max_joint="$RIGHT_MAX_JOINT_RAD" \
    --teleop.type=bi_spacemouse \
    --teleop.left_device_index=0 \
    --teleop.right_device_index=1 \
    --dataset.repo_id=test_2025/bi_rokae_record \
    --dataset.root="/home/rokae/dataset/gripper_parts_single_test" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=300 \
    --dataset.single_task="Put the two white parts into the gray box." \
    --dataset.push_to_hub=False \
    --display_data=False \
    --robot.cameras="$CAMERAS_CONFIG" \
    --teleop.kinematics_preset=wheeled_ar_dual \
    --teleop.left_end_effector_frame=AR5-5_07L-W4C4A2_tcp \
    --teleop.right_end_effector_frame=AR5-5_07R-W4C4A2_tcp \
    --teleop.left_urdf_path=/home/rokae/Projects/lerobot_rokae/rokae_python_wrapper/rokae_kinematics/rokae_urdf/AR5-5_07L-W4C4A2_description/urdf/AR5-5_07L-W4C4A2.urdf \
    --teleop.right_urdf_path=/home/rokae/Projects/lerobot_rokae/rokae_python_wrapper/rokae_kinematics/rokae_urdf/AR5-5_07R-W4C4A2_description/urdf/AR5-5_07R-W4C4A2.urdf
