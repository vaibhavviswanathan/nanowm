# Experiments log

Append-only record of training runs. Each entry has: setup, config diffs from defaults, data, training trajectory, evaluation, artifacts.

The companion `experiments/<exp_id>/` directory holds the actual GIFs, stats.json, and any other immortal artifacts. Checkpoints live in `$RESULTS_DIR/` and are referenced by absolute path here (they're too big to commit).

---

## exp01 — 8 trajectories, stopped at step 5,000

**Tag:** `poc-v1`
**Commit:** `79799d0` (Phase 5+6 results)
**Date:** 2026-05-11

### Setup
- Model: NanoWM-S/2, 256² resolution, 8-frame clips at frame_interval=2
- VAE: `stabilityai/sd-vae-ft-mse` (fp32, scaling 0.18215)
- bs=1 × grad_accum=4 = effective bs 4, bf16
- LR 1e-4, AdamW, log_every=100, val_every=2000

### Data
- 1 shard: `20210828_heightmaps_1`, bags 2, 3, 4, 7, 8, 10, 11, 12, 13 (9 trajectories)
- Bag 14 skipped (stream-cutoff partial)
- Split: 8 train / 1 val (val_fraction=0.1 + force-min-val fallback)
- Total train frames: 1,990 across 8 trajectories (~3.3 min driving)
- Action ranges: throttle `[0.000, 0.913]` mean 0.251; steer `[-1.000, 1.000]` mean 0.090

### Training
- Configured for 30k steps, **actually stopped at step 5,000** when metrics callback crashed on the pixel/latent shape mismatch (since-fixed via `evaluation.metrics.evaluate=false`)
- Wall-clock: ~25 min
- Step rate: 3.4 steps/s
- Train loss 0.493 → 0.366
- Val loss: 0.4052 → **0.3884 (best, step 2499)** → 0.3993 (step 4499, drifting up — overfit)

### Evaluation
- **Diagnosis: overfit** by step ~2500. Val loss best-by-val occurred barely past warmup; further training degraded generalization.
- Demo GIFs visually plausible terrain, sharp first ~10 frames, drift / soften in the back half.
- Action conditioning hard to verify visually with only 1 val trajectory (all val demos came from the same scene).

### Artifacts
- Checkpoint: `~/results/nanowm/20260511_120442-NanoWM-S-2-F8S2-tartandrive/checkpoints/latest/latest-epoch=2499-step=5000.ckpt` (812 MB)
- TensorBoard logs: `~/results/nanowm/20260511_120442-NanoWM-S-2-F8S2-tartandrive/tb/`
- `experiments/exp01_8trajs_step5k/demo_val_singletraj/` — 6 GIFs, all from the single val trajectory (showed lack of GT diversity)
- `experiments/exp01_8trajs_step5k/demo_train_8trajs/` — 8 GIFs, one per train trajectory (after the one-slice-per-trajectory rollout patch). These are train data so they don't measure generalization; they do confirm the model can produce diverse outputs given diverse contexts.
- `experiments/exp01_8trajs_step5k/stats.json` — action distribution

### What we learned (informs exp02)
- 8 trajectories from a single date is too narrow — model memorizes the scene's terrain distribution and overfits by step ~2500.
- Need more data: 5 shards × 10 bags should give ~50 trajectories across 4–5 dates (better terrain diversity).
- Pull more val too: with only 1 val trajectory all 6 default demo samples are from the same scene.

---
