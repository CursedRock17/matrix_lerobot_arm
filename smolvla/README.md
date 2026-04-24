# SmolVLA

**SmolVLA** — Hugging Face's lightweight Vision-Language-Action policy.
A SmolLM-2 language backbone + SigLIP vision encoder feed a small
flow-matching action expert that denoises chunks of future actions
conditioned on images, state, and the task string. Roughly one order of
magnitude smaller than Pi0, so it fits on a single consumer GPU and
trains faster, at the cost of a smaller pretraining prior.

## When to use it
- You want language conditioning but can't afford Pi0's memory footprint.
- Short/medium fine-tunes on a handful of SO101 tasks.
- RTC **is** supported — same flow-matching RTC path as Pi0.

## Scripts

| File | Purpose |
| --- | --- |
| `record.py` | Teleop the SO101 (Odin → Raven) for 50 episodes of the block-pickup task. |
| `train.py` | Fine-tune `lerobot/smolvla_base` with `--policy.type=smolvla`, chunk size 25, RTC enabled, 80k steps. |
| `train_ext.py` | Continue fine-tuning from a previously trained SmolVLA checkpoint (`pretrained_path=<your earlier run>`). |
| `evaluate.py` | Two-thread RTC evaluation (`predict_action_chunk` + `ActionQueue`) — toggle `RTC_ENABLED = False` to compare against plain chunking. |

## Run order
```bash
python smolvla/record.py        # build dataset
python smolvla/train.py         # fine-tune smolvla_base
python smolvla/evaluate.py      # RTC rollout on the real robot
```

## Camera-layout gotcha
`smolvla_base`'s config.json bakes in the Aloha camera keys
(`cam_high`, `cam_left_wrist`, `cam_right_wrist`). If you pass
`--policy.path=lerobot/smolvla_base`, the factory short-circuits and
keeps those keys — your SO101 `top` / `wrist.top` recordings then
mismatch the policy. Use `--policy.type=smolvla` **and**
`--policy.pretrained_path=lerobot/smolvla_base` together so
`input_features` are re-derived from the dataset.

## RTC in one paragraph
The flow-matching denoiser normally produces a fresh action chunk every
call, causing a small discontinuity when the new chunk replaces the old.
RTC keeps the leftover tail of the previous chunk, feeds it to the
denoiser as a "prefix to respect" (via `prev_chunk_left_over` +
`inference_delay`), and inpaints the rest — so the action stream stays
smooth even while inference is in flight. `execution_horizon` is how many
overlapping steps the blender is allowed to touch, and
`max_guidance_weight` controls how strongly the prefix is enforced.
