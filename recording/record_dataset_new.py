# Cameras
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode

# Datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

# --- Configuration ---
EPISODES_PER_SQUARE = 10  # How many times to record each specific square
EPISODE_TIME_SEC = 10    # Increased slightly for precision placement
RESET_TIME_SEC = 3

HF_USERNAME = "CursedRock17"
HF_DATASET_NAME = "tic_tac_mini"

WRIST_CAMERA_FPS = 30
BASE_CAMERA_FPS = 30
FRAME_W = 640
FRAME_H = 480

# --- Robot Setup ---
follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Raven",
    cameras={
        "wrist.top": RealSenseCameraConfig(serial_number_or_name="353322271691", width=FRAME_W, height=FRAME_H, fps=WRIST_CAMERA_FPS, use_depth=False),
        "top": OpenCVCameraConfig(index_or_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0", width=FRAME_W, height=FRAME_H, fps=BASE_CAMERA_FPS, fourcc="MJPG")
   }
)

leader_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="Odin",
)

follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config)

# --- Dataset Setup ---
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=f"{HF_USERNAME}/{HF_DATASET_NAME}",
    fps=WRIST_CAMERA_FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)

_, events = init_keyboard_listener()
init_rerun(session_name="recording")

follower.connect()
leader.connect()

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# --- Main Recording Loop ---
# This iterates through the 3x3 grid: (0,0) to (2,2)
try:
    for m in range(3):
        for n in range(3):
            for rep in range(EPISODES_PER_SQUARE):
                if events["stop_recording"]:
                    break

                # Ensure the dataset recognizes this episode belongs to this task
                current_task = f"Place an O in square: ({m},{n})"
                
                log_say(current_task)
                print(f"\n--- TARGET: {current_task} (Attempt {rep + 1}/{EPISODES_PER_SQUARE}) ---")

                # Record the Action
                record_loop(
                    robot=follower,
                    events=events,
                    fps=WRIST_CAMERA_FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=leader,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=current_task,
                    display_data=True,
                )

                # Handle Re-recording or Reset
                if events["rerecord_episode"]:
                    log_say("Re-recording last square")
                    print("Re-recording last square")
                    # We decrement the 'rep' to redo this specific square iteration
                    rep = rep-1
                    events["rerecord_episode"] = False
                    dataset.clear_episode_buffer()
                    continue 

                dataset.save_episode()

                # Reset Phase
                if not events["stop_recording"]:
                    log_say("Reset the environment")
                    print("Reset the environment")
                    record_loop(
                        robot=follower,
                        events=events,
                        fps=WRIST_CAMERA_FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=leader,
                        control_time_s=RESET_TIME_SEC,
                        single_task="Reset to home position",
                        display_data=True,
                    )

except KeyboardInterrupt:
    log_say("Interrupted by user")
    print("Interrupted by user")

# --- Finalize ---
log_say("Pushing dataset to hub")
print("Pushing dataset to hub")
follower.disconnect()
leader.disconnect()
dataset.finalize()
dataset.push_to_hub()
