"""Fine-tune SmolVLA on the block-pick-up dataset.

Uses `--policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base` so the
policy's `input_features` are re-derived from the dataset (keys
`observation.images.top` and `observation.images.wrist.top`) rather than
inherited from the base checkpoint's Aloha camera layout.
"""

import os
import subprocess
import sys

HF_USER = os.environ.get("HF_USER", "CursedRock17")
DATASET_NAME = os.environ.get("HF_DATASET", "so101_block_grab")
JOB_NAME = os.environ.get("JOB_NAME", f"{DATASET_NAME}_smolvla")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"outputs/train/smolvla_{DATASET_NAME}")

cmd = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.pretrained_path=lerobot/smolvla_base",
    f"--dataset.repo_id={HF_USER}/{DATASET_NAME}",
    "--batch_size=16",
    "--steps=80000",
    "--eval_freq=20000",
    "--save_freq=20000",
    "--policy.chunk_size=25",
    "--policy.n_action_steps=25",
    "--policy.optimizer_lr=1e-5",
    "--policy.resize_imgs_with_padding=(480,640)",
    "--policy.rtc_config.enabled=true",
    "--policy.rtc_config.execution_horizon=10",
    "--policy.device=cuda",
    f"--output_dir={OUTPUT_DIR}",
    f"--job_name={JOB_NAME}",
    "--wandb.enable=false",
]

print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
