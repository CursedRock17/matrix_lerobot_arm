"""Evaluate a trained ACT checkpoint on the SO101 follower.

ACT does not support RTC: `select_action` uses a simple internal action
queue that refills every `n_action_steps`. We lean on `record_loop` with
`policy=...` so the same capture/processing path used during recording is
reused at inference.
"""

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
HF_USER = "CursedRock17"
DATASET_NAME = "so101_block_grab_act"
POLICY_PATH = f"{HF_USER}/{DATASET_NAME}_act_0"
EVAL_REPO_ID = f"{HF_USER}/eval_{DATASET_NAME}"
device = torch.device("cuda")

# Evaluation
TASK_DESCRIPTION = "Grab a block, place it in the bin"
NUM_EPISODES = 5
EPISODE_TIME_SEC = 20

# Camera + control rate. `record_loop` demands dataset.fps == fps, so we set a
# single value here. 10 Hz keeps the laptop GPU happy while still smooth enough
# for the block-grab task.
FPS = 10
FRAME_W, FRAME_H = 640, 480

# SO101 Follower Arm Setup
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
follower = SO101Follower(follower_config)

# Load the fine-tuned ACT checkpoint onto GPU and put it in inference mode.
policy = ACTPolicy.from_pretrained(POLICY_PATH)
policy.to(device)
policy.eval()

# Build the feature schema that the dataset + policy share (state, action, images).
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Save each eval rollout as a fresh dataset so we can review success rate later.
dataset = LeRobotDataset.create(
    repo_id=EVAL_REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)

# Pre/post processors are stored alongside the policy checkpoint — normalize in, denormalize out.
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=POLICY_PATH,
    dataset_stats=None,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Keyboard listener exposes an "events" dict with stop_recording / exit_early flags.
_, events = init_keyboard_listener()
init_rerun(session_name="act_eval")

# Identity processors for the teleop/robot side — nothing to transform here during eval.
teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

follower.connect()
log_say("Robot connected")

try:
    for episode_idx in range(NUM_EPISODES):
        if events["stop_recording"]:
            break
        log_say(f"Eval episode {episode_idx + 1}/{NUM_EPISODES}")

        # record_loop runs obs → policy.select_action → send_action at `fps`,
        # and appends each frame to the dataset buffer along the way.
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_proc,
            robot_action_processor=robot_action_proc,
            robot_observation_processor=robot_obs_proc,
        )

        dataset.save_episode()

except KeyboardInterrupt:
    log_say("Interrupted by user")

log_say("Evaluation finished")
follower.disconnect()
dataset.finalize()
