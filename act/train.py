"""Train ACT from scratch on the block-pick-up dataset.

ACT (Action Chunking Transformer) is trained from scratch: there is no
pretrained checkpoint analogous to SmolVLA's `smolvla_base`. It uses a
ResNet18 vision backbone (ImageNet weights) + a small encoder/decoder
transformer and outputs chunks of `chunk_size` actions per forward pass.
"""

import os
import subprocess
import sys

HF_USER = os.environ.get("HF_USER", "CursedRock17")
DATASET_NAME = os.environ.get("HF_DATASET", "so101_block_grab_act")
POLICY_NAME = os.environ.get("HF_POLICY", f"{DATASET_NAME}_act_0")
JOB_NAME = os.environ.get("JOB_NAME", f"{DATASET_NAME}_act")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"outputs/train/act_{DATASET_NAME}")

cmd = [
    "lerobot-train",
    "--policy.type=act",
    f"--dataset.repo_id={HF_USER}/{DATASET_NAME}",
    "--batch_size=16",
    "--steps=15000",
    "--eval_freq=5000",
    "--save_freq=5000",
    "--policy.chunk_size=25",
    "--policy.n_action_steps=25",
    "--policy.optimizer_lr=1e-5",
    "--policy.device=cuda",
    f"--policy.repo_id={HF_USER}/{POLICY_NAME}",
    f"--output_dir={OUTPUT_DIR}",
    f"--job_name={JOB_NAME}",
    "--wandb.enable=false",
]

print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
