from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Config
HF_USER = "CursedRock17"
DATASET_NAME = "so101_arms_usmsm"
MODEL_NAME = "so101_arms_usmsm"
HF_DATASET_REPO_ID = f"{HF_USER}/eval_{DATASET_NAME}"
HF_MODEL_REPO_ID = f"{HF_USER}/{MODEL_NAME}"

EPISODE_TIME_S = 10
NUM_EPISODES = 3
WRIST_CAMERA_FPS = 30
TASK_DESCRIPTION = "Pick up an Object"

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "gripper_arm"

# Camera config
camera_config = {
    "wrist.top": OpenCVCameraConfig(
        index_or_path=2,
        width=640,
        height=480,
        fps=WRIST_CAMERA_FPS,
    )
}

# Follower config
follower_config = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id=FOLLOWER_ID,
    cameras=camera_config,
)

# Initialize our Gripper Arm
follower = SO101Follower(follower_config)

# Initialize the Policy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Dataset config
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_REPO_ID,
    fps=WRIST_CAMERA_FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)

# Init helpers
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

teleop_action_processor, robot_action_processor, robot_observation_processor = (
    make_default_processors()
)

# Process the Model
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_REPO_ID,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)

# Connect Follower Arm
follower.connect()
log_say("Robot connected")


# ======================
# Recording loop
# --dataset.num_episodes
# --dataset.episode_time_s
# ======================
for episode_idx in range (NUM_EPISODES):
    log_say(f"Recording episode {episode_idx + 1}/{NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=follower,
        events=events,
        fps=WRIST_CAMERA_FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_S,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    dataset.save_episode()

# ======================
# Cleanup
# ======================
log_say("Finished recording")
follower.disconnect()
dataset.push_to_hub()

