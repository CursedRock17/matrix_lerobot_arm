from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 5
EPISODE_TIME_SEC = 15
RESET_TIME_SEC = 5

TASK_DESCRIPTION = "Grab a Block, place in bin"

HF_USERNAME="CursedRock17"
HF_DATASET_NAME="eval_so101_arms_usmsm"
HF_MODEL_NAME="so101_arms_usmsm_act"

# Create Information for our Camera
WRIST_CAMERA_FPS = 30
BASE_CAMERA_FPS = 30
FRAME_W = 640
FRAME_H = 480

# Create Follower Arm Configuarion
follower_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Raven",
    cameras={
        "wrist.top": OpenCVCameraConfig(index_or_path="/dev/video0", width=FRAME_W, height=FRAME_H, fps=WRIST_CAMERA_FPS),
        "wrist.base": OpenCVCameraConfig(index_or_path="/dev/video2", width=FRAME_W, height=FRAME_H, fps=BASE_CAMERA_FPS)
   }
)

# Initialize the robot
follower = SO101Follower(follower_config)

# Initalize the Policy
act_policy = ACTPolicy.from_pretrained(HF_USERNAME + "/" + HF_MODEL_NAME)

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

# Create the required processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=act_policy,
    pretrained_path=(HF_USERNAME + "/" + HF_MODEL_NAME),
    dataset_stats=dataset.meta.stats,
)

# Test for x number of episodes
episode_idx = 0
while episode_idx < NUM_EPISODES:
    log_say(f"Running Evaluation, {episode_idx + 1} of {NUM_EPISODES}")
    print(f"Running Evaluation, {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(make_pre_post_
        robot=follower,
        events=events,
        fps=WRIST_CAMERA_FPS,
        policy=act_policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
follower.disconnect()
dataset.push_to_hub()
