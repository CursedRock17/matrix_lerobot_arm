"""Record a block-pick-up teleop dataset for ACT training.

Two 640x480 cameras:
  - top       : Logitech C920 (opencv, addressed by persistent /dev/v4l/by-id path)
  - wrist.top : Intel RealSense D405 (serial 353322271691)

Robot follower: SO101 "Raven" on /dev/ttyACM1
Teleop leader : SO101 "Odin"  on /dev/ttyACM0

ACT ignores the task string, but we still pass one so the dataset is
interchangeable with the SmolVLA/Pi0 runs.
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors


# Configuration
HF_USER = "CursedRock17"
DATASET_NAME = "so101_block_grab_act"
TASK_DESCRIPTION = "Grab a block, place it in the bin"

# Dataset Information
NUM_EPISODES = 50
EPISODE_TIME_SEC = 15
RESET_TIME_SEC = 5

# Camera Information
FPS = 30
FRAME_W, FRAME_H = 640, 480

# Leader/Follower SO101 Arm Setup
follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Raven",
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0",
            width=FRAME_W,
            height=FRAME_H,
            fps=FPS,
            fourcc="MJPG",
        ),
        "wrist.top": RealSenseCameraConfig(
            serial_number_or_name="353322271691",
            width=FRAME_W,
            height=FRAME_H,
            fps=FPS,
            use_depth=False,
            color_mode=ColorMode.RGB,
        ),
    },
)

leader_config = SO101LeaderConfig(port="/dev/ttyACM0", id="Odin")

follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config)

action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=f"{HF_USER}/{DATASET_NAME}",
    fps=FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)

_, events = init_keyboard_listener()
init_rerun(session_name="act_record")

follower.connect()
leader.connect()

teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

try:
    episode = 0
    while episode < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Episode {episode + 1} of {NUM_EPISODES}")

        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_proc,
            robot_action_processor=robot_action_proc,
            robot_observation_processor=robot_obs_proc,
            teleop=leader,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        if episode < NUM_EPISODES and not events["stop_recording"]:
            log_say("Reset the environment")
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_proc,
                robot_action_processor=robot_action_proc,
                robot_observation_processor=robot_obs_proc,
                teleop=leader,
                control_time_s=RESET_TIME_SEC,
                single_task="Reset to home position",
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording last episode")
            events["rerecord_episode"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode += 1

except KeyboardInterrupt:
    log_say("Interrupted by user")

follower.disconnect()
leader.disconnect()
dataset.finalize()

log_say("Pushing dataset to hub")
dataset.push_to_hub()
