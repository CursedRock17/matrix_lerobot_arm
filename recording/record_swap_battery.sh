export HF_USER=CursedRock17
export HF_DATASET=so101_remove_battery_full

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="/dev/ttyACM1" \
    --robot.id="Raven" \
    --robot.cameras="{ base: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.top: {type: intelrealsense, serial_number_or_name: 353322271691, width: 640, height: 480, fps: 30, use_depth: false, color_mode: RGB}}" \
    --teleop.type=so101_leader \
    --teleop.port="/dev/ttyACM0" \
    --teleop.id="Odin" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/${HF_DATASET} \
    --dataset.num_episodes=70 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="Pick Up Battery, Insert Battery into Case" \
