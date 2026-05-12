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

## exp02 — 38 trajectories from 5 shards, stopped at step 22,800

**Tag:** `poc-v2`
**Date:** 2026-05-11

### Setup
Same as exp01 (NanoWM-S/2, 256², 8-frame clips at frame_interval=2, bs=1 × grad_accum=4, bf16, LR 1e-4) — the only thing that changed is the data.

### Data
- **5 shards** (one date each):
  - `20210828_heightmaps_1` (9 trajs)
  - `20210903_heightmaps_7` (9)
  - `20210829_heightmaps_3` (9)
  - `20210829_heightmaps_2` (9)
  - `20210902_heightmaps_2` (9)
- 1 truncated bag per shard skipped → **45 usable trajectories**
- Split: 38 train / 7 val (hash-based, val_fraction=0.1)
- Total: **9,340 train frames + 1,597 val frames = ~11k frames** (~18 min driving)
- Action ranges: throttle `[0.000, 0.992]` mean 0.312; steer `[-1.000, 1.000]` mean -0.006 (closer to symmetric than exp01's biased 0.090)

### Training
- Configured 30k steps; **manually stopped at step ~22,800** (laptop time constraint, val was clearly past its best)
- Wall-clock: ~1h 40min
- Step rate: ~3.7 steps/s (slightly faster than exp01's 3.4)
- train_loss: 0.441 → 0.289 (best 0.049 at a noisy minimum)
- **val_loss best: 0.3164 at step 15,262** (vs exp01's 0.3884 — **18.5% better**)
- val_loss trajectory: 0.376 → 0.350 (step 2.1k) → 0.317 (step 15.3k, best) → 0.330 (step 22.6k)
- No NaN/Inf

### Evaluation
- **Hypothesis confirmed: more diverse data fixes the early-overfit pattern.** exp01's best val was at step 2,499; exp02's best was at step 15,262 — pushed ~6× later, matching the 5× data scale.
- Past step 15k, val_loss started drifting up the same way exp01 did past step 2.5k. So the **useful ceiling for this dataset size is around 15k steps**.
- To push further, need more data, not more steps.

### Artifacts
- Checkpoint (final): `~/results/nanowm/20260511_162258-NanoWM-S-2-F8S2-tartandrive/checkpoints/latest/latest-epoch=2199-step=22000.ckpt` (812 MB)
- Checkpoint (best-val region): `~/results/nanowm/20260511_162258-NanoWM-S-2-F8S2-tartandrive/checkpoints/across_timesteps/epoch=1999-step=20000.ckpt` (closest snapshot to step 15k best)
- TensorBoard: `~/results/nanowm/20260511_162258-NanoWM-S-2-F8S2-tartandrive/tb/`
- `experiments/exp02_38trajs_step22k/stats.json` — action distribution
- Demo GIFs: deferred (laptop time constraint) — re-run with `scripts/rollout_demo.py` pointing at the step-22000 checkpoint when convenient.

### What we learned (informs the next iteration)
- 5× more data → 18.5% lower val_loss + ~6× longer training before overfit. Linear-ish scaling, so dataset size remains the bottleneck.
- To get a meaningfully better model, the next iteration should:
  1. **Pull all 23 shards** (~280k frames, ~25× current) — needs ~17 hr of downloads. Now that we know the pipeline works end-to-end, this is just a longer download.
  2. **Move training to a cloud A100** — at our 3.7 steps/s, 100k steps on full data would take ~7.5 hr. On an A100 with bs=8, maybe ~1.5 hr (~$2 at Lambda/RunPod).
- Other deferred enhancements (best-by-val checkpoint policy, h-flip augmentation, fixing the metrics callback for cached latents) are smaller-impact and can wait.

---

## exp03 — IN PROGRESS (all 23 shards on Tensorbook)

**Status as of commit:** pulling shards 6–23 (already have 1–5 from exp02); preprocess+latents+train 100k steps will fire automatically when downloads complete.

### Plan
- 23 shards × 10 bags = ~210 trajectories after truncated-bag drops (~50k frames, ~25× exp01).
- Disk-conscious ingest: `scripts/pull_convert_clean.sh` chains the pulls and converts+deletes per shard so peak disk stays ~30 GB.
- FRESH preprocess + latents (staging retains all conversions).
- Train **100,000 steps** (3.3× exp02's actual 30k cap) — extrapolating from exp02's val bottom at step 15k with 5× data, val should keep improving into the 60k+ range on full data.

### Auto-triggered chain
- `bash scripts/pull_convert_clean.sh` for shards 6–10 (running)
- `bash scripts/pull_convert_clean.sh` for shards 11–23 (queued after the above)
- Then: `rm -rf tartandrive/ tartandrive_latents/` + preprocess + latents + `STEPS=100000 bash scripts/train.sh`

### ETA at submission time
- Downloads: ~13–14 hr
- Preprocess + latents: ~30 min
- Training: ~7.5 hr at 3.7 steps/s
- **Total ~21 hr unattended.**

### Will be filled in on completion
- Final train/val split sizes
- Step rate observed
- val_loss trajectory + best-by-val step
- Demo GIFs (will live in `experiments/exp03_<N>trajs_step<X>k/demo/`)
- Tag: `poc-v3`

---
