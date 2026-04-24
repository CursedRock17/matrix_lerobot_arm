"""Evaluate a fine-tuned Pi0 checkpoint with Real-Time Chunking (RTC).

Pi0, like SmolVLA, is a flow-matching policy that supports RTC. RTC runs
through `predict_action_chunk` + `ActionQueue`, not `select_action`, and
uses two threads:

  * get_actions : requests a new chunk whenever the queue drops below a
                  threshold, feeding RTC the leftover prefix of the old chunk
                  so the new chunk blends smoothly with actions already sent.
  * actor       : pops one action at a time from the queue at FPS and sends
                  it to the robot.

Toggle `RTC_ENABLED = False` to run the same dual-thread scaffolding but
without RTC (`ActionQueue` just appends new chunks instead of inpainting).
"""

import math
import threading
import time

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
HF_USER = "CursedRock17"
DATASET_NAME = "so101_block_grab_pi0"
POLICY_PATH = f"{HF_USER}/{DATASET_NAME}_pi0_0"
device = torch.device("cuda")

# Evaluation
TASK_DESCRIPTION = "Grab a block, place it in the bin"
NUM_EPISODES = 5
EPISODE_TIME_SEC = 20
ROBOT_TYPE = "so101_follower"

# RTC
RTC_ENABLED = True
EXECUTION_HORIZON = 10           # how many overlapping steps to blend
QUEUE_THRESHOLD = 15             # request new chunk when queue <= this many actions
MAX_GUIDANCE_WEIGHT = 10.0       # how strongly to enforce consistency

# Rates: cameras stream at CAMERA_FPS, the arm is driven at CONTROL_FPS.
# Pi0 inference is slower than SmolVLA; 10 Hz control is well within budget.
CAMERA_FPS = 30
CONTROL_FPS = 10
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
            fps=CAMERA_FPS,
            fourcc="MJPG",
        ),
        "wrist.top": RealSenseCameraConfig(
            serial_number_or_name="353322271691",
            width=FRAME_W,
            height=FRAME_H,
            fps=CAMERA_FPS,
            use_depth=False,
            color_mode=ColorMode.RGB,
        ),
    },
)
follower = SO101Follower(follower_config)

# Policy + RTC wiring
policy = PI0Policy.from_pretrained(POLICY_PATH)
policy.config.rtc_config = RTCConfig(
    enabled=RTC_ENABLED,
    execution_horizon=EXECUTION_HORIZON,
    max_guidance_weight=MAX_GUIDANCE_WEIGHT,
)
policy.init_rtc_processor()
policy.to(device)
policy.eval()

action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=POLICY_PATH,
    dataset_stats=None,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

_, events = init_keyboard_listener()
init_rerun(session_name="pi0_eval_rtc")

# Queue holds pending actions for the actor thread to consume.
action_queue = ActionQueue(policy.config.rtc_config)
# Rolling window of recent inference latencies — RTC uses max() to pick inference_delay.
latency_tracker = LatencyTracker(maxlen=20)
# Seconds per action at control rate — used to translate latency into a step count.
time_per_chunk = 1.0 / CONTROL_FPS
shutdown_event = threading.Event()


def get_actions_loop():
    """Background thread: keep the action queue filled via RTC."""
    # With RTC off, we only refill when the queue is fully empty.
    threshold = QUEUE_THRESHOLD if RTC_ENABLED else 0
    while not shutdown_event.is_set():
        # Skip if queue still has enough buffered actions.
        if action_queue.qsize() > threshold:
            time.sleep(0.01)
            continue

        # Snapshot queue state before inference so RTC can blend correctly.
        idx_before = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over()
        # Predict how many control steps inference will take, based on recent runs.
        inference_delay = math.ceil((latency_tracker.max() or 0.0) / time_per_chunk)

        t0 = time.perf_counter()
        # Grab latest camera frames + joint state from the follower.
        obs = follower.get_observation()
        # Assemble the dict Pi0 expects (tokenized task, normalized image tensors).
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=TASK_DESCRIPTION,
            robot_type=ROBOT_TYPE,
        )
        obs_frame = preprocessor(obs_frame)

        # Flow-matching denoise a new chunk, conditioned on the leftover prefix.
        actions = policy.predict_action_chunk(
            obs_frame,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_actions,
        )
        # Keep an unpost-processed copy for the RTC prefix next iteration.
        original_actions = actions.squeeze(0).clone()
        post_actions = postprocessor(actions).squeeze(0)

        # Record real latency and translate into steps for the next inference_delay.
        real_latency = time.perf_counter() - t0
        latency_tracker.add(real_latency)
        real_delay = math.ceil(real_latency / time_per_chunk)
        # RTC blends the new chunk into the queue at the correct offset.
        action_queue.merge(original_actions, post_actions, real_delay, idx_before)


def actor_loop():
    """Background thread: pop actions and send them to the robot at CONTROL_FPS."""
    # Target wall-clock budget per action (0.1s at 10Hz).
    interval = 1.0 / CONTROL_FPS
    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        # Pop one action from the queue; may be None if producer is still warming up.
        action = action_queue.get()
        if action is not None:
            # Package the 6-DoF action into the robot's expected dict schema.
            action_dict = make_robot_action(action.unsqueeze(0), dataset_features)
            follower.send_action(action_dict)
        # Sleep the remainder of the control period so we send at a steady rate.
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, interval - dt))


follower.connect()
log_say(f"Robot connected. RTC={'on' if RTC_ENABLED else 'off'}")

inference_thread = threading.Thread(target=get_actions_loop, daemon=True)
control_thread = threading.Thread(target=actor_loop, daemon=True)
inference_thread.start()
control_thread.start()

try:
    for episode_idx in range(NUM_EPISODES):
        if events["stop_recording"]:
            break
        log_say(f"Eval episode {episode_idx + 1}/{NUM_EPISODES}")

        episode_end = time.perf_counter() + EPISODE_TIME_SEC
        while time.perf_counter() < episode_end:
            if events["stop_recording"]:
                break
            time.sleep(0.05)

except KeyboardInterrupt:
    log_say("Interrupted by user")

finally:
    shutdown_event.set()
    inference_thread.join(timeout=5.0)
    control_thread.join(timeout=5.0)
    follower.disconnect()
    log_say("Evaluation finished")
