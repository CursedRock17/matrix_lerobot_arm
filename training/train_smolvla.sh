#!/usr/bin/env bash
# Fine-tune SmolVLA on a dataset recorded with two 640x480 cameras named
# `top` and `wrist.top` (dataset keys observation.images.top / observation.images.wrist.top).
#
# SmolVLA uses flow matching by design.
# Real-Time Chunking (RTC) is an inference-time feature; we bake it into the saved
# checkpoint config so async inference picks it up automatically.
set -euo pipefail

export HF_USER=${HF_USER:-CursedRock17}
export HF_DATASET=${HF_DATASET:-so101_block_grab}
export JOB_NAME=${JOB_NAME:-${HF_DATASET}_smolvla}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/train/smolvla_${HF_DATASET}}

lerobot-train \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --dataset.repo_id=${HF_USER}/${HF_DATASET} \
    --batch_size=16 \
    --steps=80000 \
    --eval_freq=20000 \
    --save_freq=20000 \
    --policy.chunk_size=25 \
    --policy.n_action_steps=25 \
    --policy.optimizer_lr=1e-05 \
    --policy.resize_imgs_with_padding='(480,640)' \
    --policy.rtc_config.enabled=true \
    --policy.rtc_config.execution_horizon=10 \
    --policy.device=cuda \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${JOB_NAME} \
    --wandb.enable=false
