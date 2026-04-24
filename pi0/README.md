# Pi0

**Pi0** (Physical Intelligence) — a Vision-Language-Action flow-matching
policy built on a PaliGemma VLM backbone (2B Gemma + SigLIP vision) with
a Gemma-300M "action expert". The VLM reads images + the task string,
the action expert denoises a flow-matching trajectory conditioned on the
VLM features. Larger and hungrier than SmolVLA; usually fine-tuned from
`lerobot/pi0_base` rather than trained from scratch.

## When to use it
- You already trust the block-pickup data and want the strongest VLA we
  can fit on one GPU.
- Multi-task or language-conditioned behavior is on the roadmap.
- RTC **is** supported — same flow-matching RTC path as SmolVLA.

## Compute notes
- PaliGemma-2B + 300M expert is heavy. We default to `bfloat16` and
  `gradient_checkpointing=true`; drop the batch size if you still OOM.
- `--policy.pretrained_path=lerobot/pi0_base` + `--policy.type=pi0` so the
  `input_features` are re-derived from our `top` / `wrist.top` keys
  instead of inheriting the base checkpoint's camera layout.

## Scripts

| File | Purpose |
| --- | --- |
| `record.py` | Teleop the SO101 for 50 episodes of the block-pickup task and push to the Hub. |
| `train.py` | Fine-tune `lerobot/pi0_base` with `--policy.type=pi0`, chunk size 25, RTC enabled, 80k steps. |
| `evaluate.py` | Two-thread RTC evaluation (`predict_action_chunk` + `ActionQueue`) — toggle `RTC_ENABLED = False` to compare against plain chunking. |

## Run order
```bash
python pi0/record.py       # build dataset
python pi0/train.py        # fine-tune pi0_base
python pi0/evaluate.py     # RTC rollout on the real robot
```

## RTC in one paragraph
The flow-matching denoiser normally produces a fresh action chunk every
call, causing a small discontinuity when the new chunk replaces the old.
RTC keeps the leftover tail of the previous chunk, feeds it to the
denoiser as a "prefix to respect" (via `prev_chunk_left_over` +
`inference_delay`), and inpaints the rest — so the action stream stays
smooth even while inference is in flight. `execution_horizon` is how many
overlapping steps the blender is allowed to touch, and
`max_guidance_weight` controls how strongly the prefix is enforced.
