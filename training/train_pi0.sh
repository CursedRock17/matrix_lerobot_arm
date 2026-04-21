#!/usr/bin/env bash
# Fine-tune Pi0 on a dataset recorded with two 640x480 cameras named
# `top` and `wrist.top` (dataset keys observation.images.top / observation.images.wrist.top).
#
# Pi0 uses flow matching by design (num_inference_steps controls the ODE steps).
# Real-Time Chunking (RTC) is an inference-time feature; we bake it into the saved
# checkpoint config so async inference picks it up automatically.
#
# Requires the pi0 extras: `pip install -e ".[pi]"` in the lerobot repo.
set -euo pipefail

export HF_USER=${HF_USER:-CursedRock17}
export HF_DATASET=${HF_DATASET:-so101_two_cam}
export JOB_NAME=${JOB_NAME:-${HF_DATASET}_pi0}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/train/pi0_${HF_DATASET}}

lerobot-train \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.type=pi0 \
    --dataset.repo_id=${HF_USER}/${HF_DATASET} \
    --batch_size=16 \
    --steps=80000 \
    --policy.chunk_size=25 \
    --policy.n_action_steps=25 \
    --policy.image_resolution='(480,640)' \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.rtc_config.enabled=true \
    --policy.rtc_config.execution_horizon=10 \
    --policy.device=cuda \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${JOB_NAME} \
    --wandb.enable=false
