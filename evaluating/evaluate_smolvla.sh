# export HF_USER=$(huggingface-cli whoami | head -n 1)
export HF_USER=CursedRock17
export HF_DATASET=tic_tac_mini
export HF_POLICY=tic_tac_mini_smolvla

# Robot Configuration
export ROBOT_PORT="/dev/ttyACM1"
export ROBOT_ID="Raven"

# User Configuration
export CAMERA_CONFIG="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.top: {type: intelrealsense, serial_number_or_name: 353322271691, width: 640, height: 480, fps: 30, use_depth: false}}" \

export CHECKPOINT_LATEST=""
export NUM_EPISODES=10
export TASK_DESCRIPTION="Place an O in square: (0, 0)"

# DERIVED VARIABLES
if [[ -z "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_PATH="outputs/train/act_${HF_USER}_test/checkpoints/last/pretrained_model"
fi

# Debug B4 Running
echo "=============================================="
echo "Port       : $ROBOT_PORT"
echo "Robot ID   : $ROBOT_ID"
echo "HF dataset : $HF_DATASET"
echo "Episodes   : $NUM_EPISODES"
echo "=============================================="

# Run Evaluation
lerobot-record \
	--robot.type=so101_follower \
	--robot.port="$ROBOT_PORT" \
	--robot.cameras="$CAMERA_CONFIG" \
	--robot.id="$ROBOT_ID" \
	--display_data=true \
	--dataset.repo_id=${HF_USER}/eval_${HF_DATASET} \
	--dataset.single_task="$TASK_DESCRIPTION" \
	--dataset.episode_time_s=999 \
	--dataset.reset_time_s=5 \
	--dataset.num_episodes="$NUM_EPISODES" \
	--policy.path=${HF_USER}/${HF_POLICY} \


echo "Evaluation done. Dataset pushed to https://huggingface.co/datasets/$HF_DATASET"
