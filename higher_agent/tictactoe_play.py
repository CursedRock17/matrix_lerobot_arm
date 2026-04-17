#!/usr/bin/env python3
"""
Tic-Tac-Toe Player for SO-101 Arm

Uses OpenAI API to analyze the board state via camera and determine the
next move, then executes the movement using a trained LeRobot policy.

Flow:
  1. Show live camera feed of the board
  2. Press SPACEBAR to capture & analyze the board with OpenAI
  3. OpenAI returns the best move (e.g. B2 → center)
  4. The trained policy executes the arm movement to that grid cell
  5. Wait for opponent's move, press SPACEBAR again

Prerequisites:
  - Recorded dataset with record_tictactoe.py
  - Trained policy on that dataset (e.g. SmolVLA fine-tune)
  - OPENAI_API_KEY environment variable set

Usage:
  OPENAI_API_KEY=sk-... pipenv run python tictactoe_play.py
"""

import base64
import os
import re
import cv2
from openai import OpenAI

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

# ── Configuration ────────────────────────────────────────────────────
HF_USER = "CursedRock17"
DATASET_NAME = "so101_tictactoe"
MODEL_NAME = "so101_tictactoe"
HF_DATASET_REPO_ID = f"{HF_USER}/eval_{DATASET_NAME}"
HF_MODEL_REPO_ID = f"{HF_USER}/{MODEL_NAME}"

WRIST_CAMERA_FPS = 30
MOVE_TIME_S = 10
OPENAI_MODEL = "gpt-5.2"

FOLLOWER_PORT = "/dev/ttyACM0"
VISION_CAMERA = "/dev/video0"

# ── Grid Mapping ─────────────────────────────────────────────────────
# Chess-like notation → (row, col, label)
MOVE_MAP = {
    "A1": (0, 0, "top-left"),     "B1": (0, 1, "top-center"),     "C1": (0, 2, "top-right"),
    "A2": (1, 0, "middle-left"),  "B2": (1, 1, "center"),         "C2": (1, 2, "middle-right"),
    "A3": (2, 0, "bottom-left"),  "B3": (2, 1, "bottom-center"),  "C3": (2, 2, "bottom-right"),
}

LABEL_MAP = {
    (0, 0): "top-left",     (0, 1): "top-center",     (0, 2): "top-right",
    (1, 0): "middle-left",  (1, 1): "center",         (1, 2): "middle-right",
    (2, 0): "bottom-left",  (2, 1): "bottom-center",  (2, 2): "bottom-right",
}

REGION_MAP = {label: pos for pos, label in LABEL_MAP.items()}

# ── OpenAI System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a tic-tac-toe AI controlling a robot arm. Analyze the board image
and decide the best next move.

Grid notation: A-C (columns left to right), 1-3 (rows top to bottom).
  A1 | B1 | C1
  ---+----+---
  A2 | B2 | C2
  ---+----+---
  A3 | B3 | C3

Your response MUST include these exact fields on separate lines:
BOARD: (describe each cell as X, O, or Empty)
TURN: X or O
MOVE: <cell> (e.g., B2)
REASON: <one sentence>
POSITION: (<row>,<col>) (e.g., (1,1) for center)
"""


# ── OpenAI Helpers ───────────────────────────────────────────────────
def get_api_key():
    """Load OpenAI API key from OPENAI_API_KEY environment variable."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set the OPENAI_API_KEY environment variable")
    return key


def encode_frame(frame):
    """JPEG-encode an OpenCV frame and return a base64 string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.standard_b64encode(buf).decode("utf-8")


def analyze_board(client, frame):
    """Send a frame to OpenAI and return the raw analysis text."""
    b64_image = encode_frame(frame)

    print("\n--- Analyzing board with OpenAI... ---")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this tic-tac-toe board and recommend the next move.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
    )

    reply = response.choices[0].message.content
    print(f"\n=== OpenAI Analysis ===\n{reply}\n=======================\n")
    return reply


def parse_move(reply):
    """Extract (row, col) from the OpenAI response. Returns None on failure."""
    # Try POSITION: (row,col)
    pos_match = re.search(r"POSITION:\s*\((\d),\s*(\d)\)", reply)
    if pos_match:
        row, col = int(pos_match.group(1)), int(pos_match.group(2))
        if 0 <= row <= 2 and 0 <= col <= 2:
            return row, col

    # Try MOVE: <cell>
    move_match = re.search(r"MOVE:\s*([A-C][1-3])", reply)
    if move_match:
        cell = move_match.group(1).upper()
        if cell in MOVE_MAP:
            row, col, _ = MOVE_MAP[cell]
            return row, col

    # Try region keywords
    reply_lower = reply.lower()
    for region, (row, col) in REGION_MAP.items():
        if region in reply_lower:
            return row, col

    return None


def get_task_description(row, col):
    """Must match the task descriptions used in record_tictactoe.py."""
    label = LABEL_MAP[(row, col)]
    return f"Move to tic-tac-toe position ({row},{col}) - {label}"


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # OpenAI client
    client = OpenAI(api_key=get_api_key())

    # Vision camera (overhead / board view, separate from wrist cam)
    vision_cap = cv2.VideoCapture(VISION_CAMERA)
    if not vision_cap.isOpened():
        raise RuntimeError(f"Cannot open vision camera at {VISION_CAMERA}")
    vision_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vision_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Robot setup
    camera_config = {
        "camera1": OpenCVCameraConfig(
            index_or_path=2, width=640, height=480, fps=WRIST_CAMERA_FPS,
        )
    }
    follower_config = SO101FollowerConfig(
        port=FOLLOWER_PORT, id="gripper_arm", cameras=camera_config,
    )
    follower = SO101Follower(follower_config)

    # Load trained policy
    policy = SmolVLAPolicy.from_pretrained(HF_MODEL_REPO_ID)

    # Dataset for recording evaluation episodes
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
    init_rerun(session_name="tictactoe_play")

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=HF_MODEL_REPO_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
        },
    )

    # Connect
    follower.connect()
    log_say("Robot connected. Starting tic-tac-toe game.")

    print("\n" + "=" * 50)
    print("  TIC-TAC-TOE with SO-101")
    print("  SPACEBAR  - analyze board & execute move")
    print("  Q         - quit")
    print("=" * 50 + "\n")

    move_count = 0

    try:
        while not events["stop_recording"]:
            ret, frame = vision_cap.read()
            if not ret:
                continue

            cv2.imshow("Tic-Tac-Toe Board", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key != ord(" "):
                continue

            # ── Capture fresh frame, analyze, and move ──
            ret, frame = vision_cap.read()
            if not ret:
                print("Warning: failed to grab frame")
                continue

            reply = analyze_board(client, frame)
            position = parse_move(reply)

            if position is None:
                print("Could not parse move from OpenAI response. Try again.")
                continue

            row, col = position
            task = get_task_description(row, col)
            move_count += 1

            print(f"\n>>> Move #{move_count}: ({row},{col}) - {LABEL_MAP[(row, col)]}")
            print(f">>> Task: {task}")
            log_say(f"Moving to position {row},{col}")

            # Execute the move with the trained policy
            record_loop(
                robot=follower,
                events=events,
                fps=WRIST_CAMERA_FPS,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=MOVE_TIME_S,
                single_task=task,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            dataset.save_episode()
            log_say("Move complete. Your turn!")
            print("\n--- Waiting for opponent. Press SPACEBAR when ready. ---")

    finally:
        log_say("Game over")
        follower.disconnect()
        vision_cap.release()
        cv2.destroyAllWindows()

        if dataset.num_episodes > 0:
            print(f"\nRecorded {dataset.num_episodes} evaluation episodes")
            dataset.push_to_hub()

        print("Done!")


if __name__ == "__main__":
    main()
