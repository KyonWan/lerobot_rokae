#!/bin/bash
# 双SpaceMouse和双Rokae单臂机器人数据采集脚本（恢复录制版本）
#
# 使用方法：
#   ./bi_rokae_record_resume.sh <数据集路径> <要额外录制的episode数量>
#
# 使用示例：
#   1. 从固定数据集路径恢复，额外录制50个episode：
#      ./bi_rokae_record_resume.sh /home/rokae/Projects/datasets/double_arm_pickup1 50
#
#   2. 如果已录制了30个episode，想再录制70个达到100个：
#      ./bi_rokae_record_resume.sh /home/rokae/Projects/datasets/double_arm_pickup1 70
#
#   3. 查看当前数据集已录制的episode数量：
#      ls -la /home/rokae/Projects/datasets/double_arm_pickup1/meta/episodes/
#
# 注意事项：
#   - <要额外录制的episode数量> 是要新增的episode数，不是总数
#   - 数据集路径必须与上次录制时使用的路径完全一致
#   - 确保机器人服务器已启动（左臂5000端口，右臂5001端口）

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "错误: 缺少必需参数"
    echo ""
    echo "使用方法: $0 <数据集路径> <要额外录制的episode数量>"
    echo ""
    echo "示例:"
    echo "  $0 /home/rokae/Projects/datasets/double_arm_pickup1 50"
    echo "  (从指定路径恢复，额外录制50个episode)"
    echo ""
    echo "提示: 要查看已录制的episode数量，可以运行:"
    echo "  ls -la /home/rokae/Projects/datasets/double_arm_pickup1/meta/episodes/"
    exit 1
fi

DATASET_ROOT="$1"
ADDITIONAL_EPISODES="$2"

# 相机配置
CAMERAS_CONFIG="{external: {type: intelrealsense, serial_number_or_name: '125322062165', \
width: 640, height: 480, fps: 60, use_depth: false}, \
left_wrist: {type: intelrealsense, serial_number_or_name: '809512060572', \
width: 640, height: 480, fps: 60, use_depth: false}, \
right_wrist: {type: intelrealsense, serial_number_or_name: '036422060433', \
width: 640, height: 480, fps: 60, use_depth: false}}"

python -m lerobot.scripts.lerobot_record \
    --resume=true \
    --robot.type=bi_rokae_robot \
    --robot.left_zmq_port=5555 \
    --robot.right_zmq_port=5556 \
    --robot.left_joint_num=7 \
    --robot.right_joint_num=7 \
    --robot.left_control_mode=cartesian_impedance \
    --robot.left_callback_mode=cart_vel \
    --robot.right_control_mode=cartesian_impedance \
    --robot.right_callback_mode=cart_vel \
    --teleop.type=bi_spacemouse \
    --teleop.left_device_index=5 \
    --teleop.right_device_index=1 \
    --dataset.repo_id=test_2025/bi_rokae_record \
    --dataset.root="$DATASET_ROOT" \
    --dataset.num_episodes="$ADDITIONAL_EPISODES" \
    --dataset.episode_time_s=100 \
    --dataset.single_task="Use the left arm to place the two small joint modules into the two left blue boxes, and use the right arm to place the two large joint modules into the two right blue boxes." \
    --dataset.push_to_hub=False \
    --display_data=true \
    --robot.cameras="$CAMERAS_CONFIG"
