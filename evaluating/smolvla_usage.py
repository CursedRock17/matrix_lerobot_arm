import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 999
USERNAME="CursedRock17"
DATASET="so101_block_grab"


def main():
    device = torch.device("cuda")  # or "cpu"
    model_id = USERNAME + "/" + DATASET + "_smolvla_1"

    model = SmolVLAPolicy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        # This overrides allows to run on CUDA (if available)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # find ports using lerobot-find-port
    follower_port = "/dev/ttyACM1"

    # the robot ids are used the load the right calibration files
    follower_id = "Raven"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "top": OpenCVCameraConfig(index_or_path="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0", width=640, height=480, fps=30),
        "wrist.top": RealSenseCameraConfig(serial_number_or_name=353322271691, 
                                           width=640, height=480, 
                                           fps=30, color_mode=ColorMode.BGR, use_depth=False),
    }

    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)
    robot.connect()

    task = ""  # something like "pick the red block"
    robot_type = ""  # something like "so101_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
