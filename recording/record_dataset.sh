export HF_USER=CursedRock17
export HF_DATASET=so101_remove_battery_full

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="/dev/ttyACM3" \
    --robot.id="Raven" \
    --robot.cameras="{ base: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.top: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port="/dev/ttyACM2" \
    --teleop.id="Odin" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/${HF_DATASET} \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=18 \
    --dataset.reset_time_s=3 \
    --dataset.single_task="Remove Green Battery, place into dropbox" \
