#!/usr/bin/env python3
"""
Tic-Tac-Toe Vision Analyzer for SO-101 Arm

Press SPACEBAR to capture a frame from /dev/video0 and send it to
GPT-5.2 for tic-tac-toe board analysis.  The model returns the best
next move for the SO-101 arm (playing as X or O).

Press Q to quit.
"""

import base64
import os
import sys

import cv2
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────
CAMERA_DEVICE = "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405-video-index2"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL = "gpt-5.2"

# Load API key: env var first, fall back to local key file
API_KEY = ("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")
client = OpenAI(api_key=API_KEY)

# ── System prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a tic-tac-toe analysis assistant for a VLA-trained SO-101 robotic arm.
Remeber, you need to get 3 of your object in a row and cannot place it on an occupied square.
The grid is rotated 180 degrees, so for the robot the top left corner, is actually the camerasbottom right corner

When you receive an image of a tic-tac-toe board:

1. **Read the board** – Identify every cell (3x3 grid). Label columns
   left-to-right as A, B, C and rows top-to-bottom as 1, 2, 3.
   Report each cell as X, O, or empty.

2. **Determine whose turn it is** – Count Xs and Os.
   If counts are equal it is X's turn; otherwise O's turn.

3. **Find the optimal move** – Use minimax reasoning:
   - Prioritise: win now > block opponent win > fork > centre > corner > edge.

4. **Return a structured response** with:
   - The current board state (text grid).
   - Whose turn it is.
   - The best move (cell label, e.g. B2).
   - A brief explanation of why this move is best.
   - Physical guidance: describe the approximate board region
     (top-left, centre, bottom-right, etc.) so the SO-101 arm
     can position itself over the correct cell.
"""

def encode_frame(frame):
    """JPEG-encode an OpenCV frame and return a base64 string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.standard_b64encode(buf).decode("utf-8")


def analyze_board(frame):
    """Send a captured frame to GPT-5.2 and print the analysis."""
    b64_image = encode_frame(frame)

    print("\n--- Sending frame to GPT-5.2 for analysis... ---")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """
                            "Analyze this tic-tac-toe board image. "
                            "Determine the current state and recommend "
                            "the best next move for the SO-101 arm."
                            """
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    reply = response.choices[0].message.content
    print("\n=== GPT-5.2 Analysis ===")
    print(reply)
    print("========================\n")
    return reply


def main():
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        sys.exit(f"Error: could not open camera at {CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Tic-Tac-Toe Vision Analyzer")
    print("  SPACEBAR  – capture & analyze")
    print("  Q         – quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to grab frame, retrying...")
            continue

        cv2.imshow("Tic-Tac-Toe SO-101 Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            analyze_board(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
