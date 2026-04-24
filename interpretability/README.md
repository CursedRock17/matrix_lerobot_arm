# Interpretability

Watch where the vision path of a policy focuses while it drives the arm.
Per-camera heatmaps are streamed to rerun next to the raw image, so you
can eyeball whether the model is locking onto the block, the gripper, a
cable in the background, etc.

Two policy variants are wired up:

- **SmolVLA** — attention rollout (Abnar & Zuidema 2020) across the
  SigLIP ViT layers.
- **ACT** — per-spatial-cell activation magnitude of the ResNet18
  backbone's final conv stage (`layer4`). ACT has no self-attention in
  the backbone, so this is the CNN analogue: "where does the encoder
  fire hardest before the transformer mixes tokens together?"

## What you get

Three rerun streams per camera:

```
attention/top/image       # raw 640x480 RGB
attention/top/attention   # heatmap (red = high attention)
attention/top/overlay     # blended 50/50
attention/wrist.top/...   # same, for the wrist camera
```

Updated once per RTC chunk (~every 10–20 control steps), not per frame —
enough to read the story without burning compute.

## Design

`torch.nn.functional.scaled_dot_product_attention` (which the HF vision
encoder uses under the hood) never materializes `softmax(QK^T/sqrt(d))`,
so we can't just read attention weights off the model. Instead we:

1. Install forward hooks on every attention layer's `q_proj` / `k_proj`
   inside the SigLIP vision encoder. Each hook caches the projected Q/K
   tensor.
2. After each `embed_image` call (one per camera per chunk), recompute
   `softmax(Q @ K^T / sqrt(d_head))` manually from the cached tensors,
   run attention rollout across all ViT layers with the residual trick,
   and store the (seq, seq) rollout.
3. Reduce the rollout to a per-patch importance vector (column-mean),
   reshape to the patch grid, bilinear-upsample to the camera image
   size, and overlay via rerun.

Cost: one extra matmul+softmax per ViT layer per image per chunk. On a
4090 laptop this is noise compared to the VLM forward pass.

## Scope

**What this tells you.** Where inside each image the vision encoder is
concentrating its own internal attention — i.e. which image regions the
encoder thinks are salient to represent.

**What this does NOT tell you.** Which image regions actually drive the
*action* output. Vision-encoder attention is a proxy — a cleaner answer
would inspect the expert→prefix cross-attention at the final joint
SmolVLM+expert layer, or run a gradient-based attribution
(IntegratedGradients, Grad-CAM) from the action logits back to the
input pixels. Both are straightforward follow-ups that reuse the same
rerun logging path.

**Pi0** uses the same SigLIP-style encoder as SmolVLA and would be a
near-identical port of `attention_capture.py`. **ACT** is handled
separately via `act_attention_capture.py` because its ResNet backbone
has no `q_proj`/`k_proj` to hook — we capture the `layer4` feature map
instead.

## ACT-specific notes

ACT's ResNet is a pure CNN, so the overlay you see is an *activation
heatmap*, not an attention map. Channel-wise L2 norm on the `layer4`
feature map highlights where the CNN has the largest pre-transformer
response. Caveats:

- Bright doesn't mean *important for the action* — the transformer
  still reweights these tokens via cross-attention. A cleaner "what
  drives the action" signal would be the decoder's cross-attention
  weights onto the image tokens, which is a natural follow-up on the
  same hook scaffold.
- The overlay only refreshes when ACT's internal action queue refills
  (every `n_action_steps` control steps). Between refills the last
  overlay stays on screen in rerun.

## Running it

```bash
python interpretability/evaluate_with_viz.py      # SmolVLA + RTC + rollout
python interpretability/evaluate_with_viz_act.py  # ACT + ResNet activation
```

Toggle `ATTENTION_ENABLED = False` at the top of either script to run
the same control loop without the capture — useful for A/B comparing
the policy's behavior with the instrumentation removed.

## Files

| File | Purpose |
| --- | --- |
| `attention_capture.py` | `VisionAttentionCapture` — hooks Q/K on every ViT layer, computes rollout on demand. (SmolVLA / Pi0) |
| `act_attention_capture.py` | `ACTBackboneCapture` — hooks the ResNet backbone, collects per-camera `layer4` feature maps. |
| `viz.py` | Patch-grid reshape, bilinear upsample, hot-ramp colormap, rerun logging. Shared. |
| `evaluate_with_viz.py` | SmolVLA + RTC eval loop with the capture wired in. |
| `evaluate_with_viz_act.py` | ACT eval loop with the ResNet-activation capture wired in. |
