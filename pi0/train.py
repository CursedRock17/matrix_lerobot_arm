"""Fine-tune Pi0 on the block-pick-up dataset.

Uses `--policy.type=pi0 --policy.pretrained_path=lerobot/pi0_base` so the
policy's `input_features` are re-derived from the dataset (keys
`observation.images.top` and `observation.images.wrist.top`) rather than
inherited from the base checkpoint's original camera layout.

Pi0 is memory-hungry because of the PaliGemma VLM backbone. We default to
bfloat16 + gradient checkpointing; bump the batch size back up if you have
the VRAM.
"""

import os
import subprocess
import sys

HF_USER = os.environ.get("HF_USER", "CursedRock17")
DATASET_NAME = os.environ.get("HF_DATASET", "so101_block_grab_pi0")
POLICY_NAME = os.environ.get("HF_POLICY", f"{DATASET_NAME}_pi0_0")
JOB_NAME = os.environ.get("JOB_NAME", f"{DATASET_NAME}_pi0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"outputs/train/pi0_{DATASET_NAME}")

cmd = [
    "lerobot-train",
    "--policy.type=pi0",
    "--policy.pretrained_path=lerobot/pi0_base",
    f"--dataset.repo_id={HF_USER}/{DATASET_NAME}",
    "--batch_size=16",
    "--steps=15000",
    "--eval_freq=5000",
    "--save_freq=5000",
    "--policy.chunk_size=25",
    "--policy.n_action_steps=25",
    "--policy.optimizer_lr=2.5e-5",
    "--policy.scheduler_decay_steps=15000",
    "--policy.dtype=bfloat16",
    "--policy.gradient_checkpointing=true",
    "--policy.freeze_vision_encoder=false",
    "--policy.train_expert_only=false",
    "--policy.rtc_config.enabled=true",
    "--policy.rtc_config.execution_horizon=10",
    "--policy.device=cuda",
    f"--policy.repo_id={HF_USER}/{POLICY_NAME}",
    f"--output_dir={OUTPUT_DIR}",
    f"--job_name={JOB_NAME}",
    "--wandb.enable=false",
]

print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
