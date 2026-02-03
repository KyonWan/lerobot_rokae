python -m lerobot.scripts.lerobot_record ^
    --robot.type=bi_rokae_robot ^
    --robot.left_server_port=5000 ^
    --robot.right_server_port=5001 ^
    --robot.left_joint_num=7 ^
    --robot.right_joint_num=7 ^
    --robot.left_control_mode=cartesian_position ^
    --robot.left_callback_mode=cart_vel ^
    --robot.right_control_mode=cartesian_position ^
    --robot.right_callback_mode=cart_vel ^
    --teleop.type=bi_spacemouse ^
    --teleop.left_device_index=0 ^
    --teleop.right_device_index=1 ^
    --dataset.repo_id=test_2025/bi_rokae_record ^
    --dataset.root="D:\ayyzw\Documents\CodeLibrary\Embodied Intelligence\Projects\LeRobotWithRokae\dataset" ^
    --dataset.num_episodes=2 ^
    --dataset.episode_time_s=100 ^
    --dataset.single_task="Bimanual manipulation task" ^
    --dataset.push_to_hub=False ^
    --display_data=False
    
    @REM --robot.cameras="{external: {type: intelrealsense, serial_number_or_name: 809512060572, width: 640, height: 480, fps: 60}, wrist: {type: intelrealsense, serial_number_or_name: 125322062165, width: 640, height: 480, fps: 60}}"
