# SO-101 World-Model Hackathon Plan

A scoped-down WorldGym / World-Gymnast reproduction targeted at the SO-101 arm, built on `nano-world-model` + LeRobot.

## Goal & scope

Build an action-conditioned video world model on SO-101 manipulation data, then use it for MPC-style planning (CEM) on a real SO-101 task.

**In scope (hackathon):**
- Pretrain NanoWM-B/2 on a mix of public SO-100/SO-101 LeRobot datasets.
- Fine-tune on ~100 self-collected SO-101 episodes for one target task.
- Evaluate via CEM planning over imagined rollouts with VLM-based reward.

**Out of scope (and intentionally so):**
- World-Gymnast-style GRPO RL fine-tuning of a VLA policy. Too risky in 36 hr; high probability of world-model reward-hacking. Listed as stretch goal only.
- WidowX / Bridge-platform reproduction. Different embodiment, different action space, doesn't transfer.
- Matching paper-quality FVD / 18× SFT improvement numbers.

**Hackathon win condition:** "CEM picks the action sequence that reaches the goal in imagination, robot executes a recognizable attempt at the task."

## Hardware assumptions

- 1× H100 (80GB) or 1× A100 (40GB) for training. Plan times below are for H100; halve batch and double wall-clock for A100.
- 1× SO-101 follower + leader for teleop and eval.
- Wrist camera + (optionally) top camera, both 480×640 @ 30fps via OpenCV.

---

# Part 1 — Datasets & transform pipeline

## 1.1 Source datasets (Tier 1)

The official LeRobot SmolVLA training set. All four have `wrist` + `top` cameras and matching SO-10X embodiment. Combined: ~208 episodes, ~90k frames.

| repo_id | episodes | frames | format | action key prefix |
|---|---|---|---|---|
| `lerobot/svla_so100_pickplace` | 50 | 19,631 | v2.1 | `main_*` |
| `lerobot/svla_so100_stacking` | 56 | 22,956 | v3.0 | `main_*` |
| `lerobot/svla_so100_sorting` | 52 | 35,713 | v3.0 | `main_*` |
| `lerobot/svla_so101_pickplace` | 50 | 11,939 | v2.1 | `*.pos` |

Plus your own ~100 episodes of the target task collected at the hackathon.

## 1.2 Acquisition & format unification

```bash
for ds in svla_so100_pickplace svla_so100_stacking svla_so100_sorting svla_so101_pickplace; do
  huggingface-cli download --repo-type dataset lerobot/$ds --local-dir ./data/$ds
  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=lerobot/$ds
done
```

Or use `StreamingLeRobotDataset` to skip local download entirely — saves ~30 GB of disk and the ~1 hour download time.

## 1.3 Remap layer

See `dataset_wrapper.py`. Three responsibilities:

- **Action key remap.** `main_shoulder_pan` and `shoulder_pan.pos` both map to canonical index 0 (`shoulder_pan`). Same for the other five joints. Stored per-source in `ACTION_KEY_MAPS`.
- **Camera key remap.** All Tier 1 sources happen to use `observation.images.wrist` / `observation.images.top` already; for community datasets in Tier 2, alias `wrist_cam`/`tip`/`hand` → `wrist`.
- **FPS unification.** Subsample to 15 fps via stride-2 indexing on the 30 fps sources. Roughly halves memory per clip; doubles the effective temporal horizon per training sample.

## 1.4 Normalization

**Do not** reuse per-source `meta/stats.json` — those are per-dataset and will misnormalize the mix. Run `compute_stats.py` once after remap to produce a single `combined_stats.json` over the unified action space.

For images: standard `[0, 1]` scaling. No per-channel normalization (the model learns that internally).

## 1.5 Augmentation pipeline

Applied at dataloader level, per-clip (not per-frame) for spatial augs:

```python
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)  # same params across clip
RandomResizedCrop(scale=(0.9, 1.0))                                   # same crop across clip
GaussianNoise(std=0.015)                                              # per-frame (sensor noise)
ActionNoise(std=0.01 normalized ≈ 0.5deg)                             # per-frame
CameraDropout(p=0.15, mutually exclusive between wrist/top)           # per-clip
```

**Do not use:** flips, rotations, time reversal, random erasing/CutOut.

At inference, additionally **histogram-match** the live camera to training distribution (linear remap of channel mean/std). Cheap and high-leverage for camera-domain gap.

---

# Part 2 — Model & training

## 2.1 Model: NanoWM-B/2

Use the smaller of the two nano-wm variants. L/2 is better but slower at CEM inference and harder to converge in hackathon time. CEM samples 64+ candidates per replan step; latency matters.

## 2.2 Best-config flags (match nano-wm's published defaults)

- Diffusion parameterization: `pred-v` (velocity prediction)
- Action conditioning: `additive` (action features added into denoising blocks)
- Schedule: cosine + ZTSNR (zero terminal SNR — important for sharp samples)
- Patch size: 2
- Context length: 8 frames (~0.5 s at 15 fps)
- Prediction horizon: 8 frames
- Image resolution: 256×256

## 2.3 Training hyperparameters

| param | value |
|---|---|
| optimizer | AdamW |
| lr (pretrain) | 1e-4 |
| lr (finetune) | 3e-5 |
| weight_decay | 0.01 |
| betas | (0.9, 0.999) |
| warmup | 2k steps, linear |
| schedule | cosine decay |
| batch size | 32–64 clips (H100) |
| mixed precision | bf16 (not fp16 — ZTSNR is numerically delicate) |
| EMA decay | 0.9999 |
| gradient clipping | 1.0 |

## 2.4 Schedule (36-hr hackathon, 1× H100)

| stage | input | steps | wall-clock |
|---|---|---|---|
| Pretrain | Tier 1 mix (uniform sample) | 60–80k | 18–24 hr |
| Finetune | your task data (~100 eps) + task language | 10–15k | 3–5 hr |
| CEM eval iteration | real SO-101 | — | 6–10 hr |

Pretrain runs **without** task-language conditioning (the four sources are different tasks; you want a dynamics prior, not a multitask policy). Finetune turns task conditioning on.

Start the pretrain run before you have your task data — the four Tier 1 datasets are sufficient. Fine-tune as soon as your task collection completes.

## 2.5 Inference (CEM planning)

| param | value |
|---|---|
| DDIM steps | 25 (50 if quality is poor) |
| CEM horizon | 8–12 actions |
| CEM population | 64 |
| CEM iterations | 3 |
| Elite fraction | 0.1 |
| Replan frequency | every 4 actions (4-step open-loop) |
| Reward signal | VLM scoring of final imagined frame, 0–1 scale |

Reward prompt should be stable across the whole project. Suggested: *"Did this robot arm complete the task: \<task\>? Reply with a single number between 0 and 1, where 1 means complete success and 0 means no progress."* Use `claude-haiku-4-5-20251001` or `gpt-4o-mini` for latency.

---

# Monitoring & failure modes

Watch these during training:

1. **Diffusion loss** on holdout — smooth drop, watch for plateau around 30–50k steps.
2. **Validation FVD** every 5k steps on a held-out split — primary quality metric.
3. **Action-conditioning sensitivity.** Sample the same rollout with action `+a` vs `-a`. If frames are nearly identical, action conditioning has collapsed (known failure). Diverging is what you want.
4. **Eyeball check** every 5k steps: 16 rollouts, watched at 1× speed. If the gripper doesn't visibly respond to actions, no metric will catch this fast enough.

---

# Risk hedges

- Keep last 3 EMA checkpoints. Final isn't always best for CEM.
- Bake a fallback: 2-hr behavioral-cloning ACT policy on the same task data. If the WM+MPC pipeline doesn't converge, ACT gives you a working robot demo.
- If pretrain fidelity is bad by the 24-hr mark, **drop CEM and just demo the world model**. "Conditional video generation on robot data" is still a hackathon-grade demo.
- Calibrate your SO-101 *before* recording your task data — joint-position values are arm-dependent; miscalibration will break alignment with the svla pretrain corpus.

---

# Deliverables checklist

- [ ] `dataset_wrapper.py` — unified loader with remap + augmentation
- [ ] `compute_stats.py` — compute combined normalization stats
- [ ] `configs/so101/mix.yaml` — dataset Hydra config
- [ ] `configs/experiment/so101_pretrain.yaml` — experiment Hydra config
- [ ] `configs/experiment/so101_finetune.yaml` — experiment Hydra config (override of pretrain)
- [ ] Pretrained NanoWM-B/2 checkpoint on Tier 1 mix
- [ ] Finetuned NanoWM-B/2 on your task data
- [ ] CEM planner script with VLM reward
- [ ] Demo recording: real SO-101 executing CEM-planned actions

---

# References

- nano-world-model: https://github.com/simchowitzlabpublic/nano-world-model
- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA training datasets: `lerobot/svla_so100_*`, `lerobot/svla_so101_pickplace`
- WorldGym paper (Quevedo et al., 2025) — proxy world models for VLA eval
- World-Gymnast paper (Sharma et al., 2026) — GRPO RL fine-tuning of VLAs inside a world model
