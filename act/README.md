# ACT

**Action Chunking Transformer** — a ResNet18 + small encoder/decoder
transformer that maps the current observation (images + proprioceptive
state) to a fixed-length chunk of future actions. Trained from scratch
per dataset with a VAE objective plus L1 reconstruction loss. No language
conditioning and no pretrained checkpoint to fine-tune from, so it's the
fastest of the three to train and the lightest to run at inference.

## When to use it
- Small/medium datasets (tens of episodes), one well-defined task.
- You don't need language conditioning — the task description is recorded
  into the dataset but the policy ignores it.
- RTC is **not** supported: ACT executes actions from its internal queue,
  refilling every `n_action_steps`.

## Scripts

| File | Purpose |
| --- | --- |
| `record.py` | Teleop the SO101 (Odin leader → Raven follower) for 50 episodes of the block-pickup task and push to the Hub. |
| `train.py` | Wrap `lerobot-train` with ACT-specific flags (`--policy.type=act`, chunk size 25, 80k steps, batch size 16). |
| `evaluate.py` | Load the trained checkpoint and run `record_loop` with `policy=...` for NUM_EPISODES rollouts. |

## Run order
```bash
python act/record.py      # build dataset
python act/train.py       # train from scratch
python act/evaluate.py    # roll out on the real robot
```

Change `DATASET_NAME` / `POLICY_PATH` at the top of each script (or via
`HF_USER` / `HF_DATASET` / `HF_POLICY` env vars for `train.py`).
