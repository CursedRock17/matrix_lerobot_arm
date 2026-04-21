from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

NUM_EPISODES = 2 
EPISODE_TIME_SEC = 5
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Grab a Block, place in bin"

HF_USERNAME="CursedRock17"
HF_DATASET_NAME="tic_tac_mini"

# Create Information for our Camera
WRIST_CAMERA_FPS = 30
BASE_CAMERA_FPS = 30
FRAME_W = 640
FRAME_H = 480

# Create Follower Arm Configuarion
follower_config = SO101FollowerConfig(
    port="/dev/ttyACM3",
    id="Raven",
    cameras={
        "base": OpenCVCameraConfig(index_or_path="/dev/video7", width=FRAME_W, height=FRAME_H, fps=WRIST_CAMERA_FPS, fourcc="MJPG"),
        "top": OpenCVCameraConfig(index_or_path="/dev/v4l/by-id/usb-icSpring_Web_Camera_2024120914230143508-video-index0", width=FRAME_W, height=FRAME_H, fps=BASE_CAMERA_FPS, fourcc="MJPG")
   }
)

# Create Leader Arm Configuration
leader_config = SO101LeaderConfig(
    port="/dev/ttyACM2",
    id="Odin",
)

# Initialize the robot and teleoperator
follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config)

# Configure the dataset features
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=(HF_USERNAME + "/" + HF_DATASET_NAME),
    fps=WRIST_CAMERA_FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=3,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
follower.connect()
leader.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
    print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

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
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=follower,
            events=events,
            fps=WRIST_CAMERA_FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=leader,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
follower.disconnect()
leader.disconnect()
dataset.finalize()
dataset.push_to_hub()
