python -m lerobot.scripts.lerobot_record \
    --robot.type=rokae_robot \
    --teleop.type=spacemouse \
    --dataset.repo_id=test_2025/rokae_record \
    --dataset.root="/home/rokae/Projects/datasets" \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    --dataset.push_to_hub=False \
    --display_data=true
    --robot.cameras="{external: {type: intelrealsense, serial_number_or_name: 809512060572, width: 640, height: 480, fps: 60}, wrist: {type: intelrealsense, serial_number_or_name: 125322062165, width: 640, height: 480, fps: 60}}" \
