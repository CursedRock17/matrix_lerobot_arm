#!/usr/bin/env python3
"""
Tic-Tac-Toe Dataset Recorder for SO-101 Arm

Records a multi-chunk dataset where each chunk contains teleoperated
demonstrations of the arm moving to a specific tic-tac-toe grid position.
Two cameras are used: a RealSense wrist camera and an OpenCV overhead camera.

Grid layout (row, col):
  (0,0) | (0,1) | (0,2)
  ------+-------+------
  (1,0) | (1,1) | (1,2)
  ------+-------+------
  (2,0) | (2,1) | (2,2)

Each chunk uses a unique task description so a trained policy can
differentiate between target positions at inference time.

Cameras:
  wrist.top  – Intel RealSense (RGB only, no depth)  640x480 @ 30fps
  top        – OpenCV camera                          640x480 @ 30fps

Usage:
  pipenv run python record_tictactoe.py
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

import time

# ── Hugging Face ─────────────────────────────────────────────────────
USERNAME = "CursedRock17"
REPO_NAME = "tic_tac_mini"

# ── Recording Parameters ─────────────────────────────────────────────
EPISODES_PER_POSITION = 20
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5

# ── Grid Positions ───────────────────────────────────────────────────
# All 9 tic-tac-toe cells: ((row, col), label)
GRID_POSITIONS = [
    ((0, 0), "top-left"),
    ((0, 1), "top-center"),
    ((0, 2), "top-right"),
    ((1, 0), "middle-left"),
    ((1, 1), "center"),
    ((1, 2), "middle-right"),
    ((2, 0), "bottom-left"),
    ((2, 1), "bottom-center"),
    ((2, 2), "bottom-right"),
]

# Set to a list of indices to record only specific positions, e.g. [0, 4, 8].
# None records all 9 positions.
POSITIONS_TO_RECORD = None

# ── Hardware ─────────────────────────────────────────────────────────
# Find ports with:    pipenv run lerobot-find-port
# Find cameras with:  pipenv run lerobot-find-cameras opencv
#                      pipenv run lerobot-find-cameras realsense
follower_port = "/dev/ttyACM0"  # Gripper Arm
leader_port = "/dev/ttyACM1"    # Controller Arm

camera_config = {
    # RealSense wrist camera (RGB only, no depth)
    "wrist.top": RealSenseCameraConfig(
        serial_number_or_name="",   # TODO: fill with `pipenv run lerobot-find-cameras realsense`
        fps=FPS,
        width=640,
        height=480,
        use_depth=False,
    ),
    # OpenCV overhead camera
    "top": OpenCVCameraConfig(
        index_or_path=0,            # TODO: verify with `pipenv run lerobot-find-cameras opencv`
        fps=FPS,
        width=640,
        height=480,
    ),
}

# ── Robot Setup ──────────────────────────────────────────────────────
follower_config = SO101FollowerConfig(
    port=follower_port,
    id="gripper_arm",
    cameras=camera_config,
)
leader_config = SO101LeaderConfig(
    port=leader_port,
    id="teleop_arm",
)

gripper = SO101Follower(follower_config)
teleop_arm = SO101Leader(leader_config)

# ── Dataset Setup ────────────────────────────────────────────────────
action_features = hw_to_dataset_features(gripper.action_features, "action")
obs_features = hw_to_dataset_features(gripper.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=f"{USERNAME}/{REPO_NAME}",
    fps=FPS,
    features=dataset_features,
    robot_type=gripper.name,
    use_videos=True,
    image_writer_threads=4,
)

# ── Initialisation ───────────────────────────────────────────────────
_, events = init_keyboard_listener()
init_rerun(session_name="tictactoe_recording")

gripper.connect()
teleop_arm.connect()
time.sleep(0.5)
print("Connected")

teleop_action_processor, robot_action_processor, robot_observation_processor = (
    make_default_processors()
)


def task_description(row: int, col: int, label: str) -> str:
    """Generate a consistent task description for a grid position."""
    return f"Move to tic-tac-toe position ({row},{col}) - {label}"


# ── Recording Loop ───────────────────────────────────────────────────
positions = (
    [GRID_POSITIONS[i] for i in POSITIONS_TO_RECORD]
    if POSITIONS_TO_RECORD is not None
    else GRID_POSITIONS
)

for (row, col), label in positions:
    task = task_description(row, col, label)

    print(f"\n{'=' * 60}")
    print(f"  CHUNK: Position ({row},{col}) - {label}")
    print(f"  Recording {EPISODES_PER_POSITION} episodes")
    print(f"  Task: {task}")
    print(f"{'=' * 60}\n")

    log_say(f"Recording position {row},{col} {label}")

    episode_idx = 0
    while episode_idx < EPISODES_PER_POSITION and not events["stop_recording"]:
        log_say(f"Episode {episode_idx + 1} of {EPISODES_PER_POSITION} for position {row},{col}")

        record_loop(
            robot=gripper,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop_arm,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=task,
            display_data=True,
        )

        # Reset the environment between episodes (not recorded to dataset)
        if not events["stop_recording"] and (
            episode_idx < EPISODES_PER_POSITION - 1 or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=gripper,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop_arm,
                control_time_s=RESET_TIME_SEC,
                single_task=task,
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

    if events["stop_recording"]:
        log_say("Recording stopped early")
        break

# ── Cleanup ──────────────────────────────────────────────────────────
log_say("Stop recording")
gripper.disconnect()
teleop_arm.disconnect()

print(f"\nDataset complete: {dataset.num_episodes} total episodes")
print("Pushing to hub...")
dataset.push_to_hub()
print("Done!")
