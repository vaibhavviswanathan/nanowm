# Results — TartanDrive × nano-world-model POC

Tagged at: `poc-v1` (see `git tag`).

## Setup

- Upstream: nano-world-model @ `b60e9905` (see `UPSTREAM_PIN`)
- Patches: `patches/01-upstream-integration.patch` (~150 lines, 7 files)
- Hardware: RTX 3080 Laptop (16 GB), CUDA 12.1
- Model: NanoWM-S/2, 256² resolution, 8-frame clips at frame_interval=2
- VAE: `stabilityai/sd-vae-ft-mse` (scaling 0.18215), fp32
- Action injection: 2-D (throttle = `linear.x`, steering = `angular.z` from `/cmd`)

## Data

| Split | Trajectories | Frames | Source bags |
|---|---|---|---|
| train | 8 | 1,990 | TartanDrive 1.0 `20210828_heightmaps_1` shard, bags 2,3,4,7,8,10,11,12,13 |
| val | 1 | ~250 | same shard |

Total: ~3.3 min of driving data, 256² at 10 Hz. One bag (`20210828_14.bag`) was the truncated stream-cutoff and skipped automatically by `convert_bags.py`.

Action stats (train):
- throttle range `[0.000, 0.913]`, mean `0.251`, std `0.168`
- steering range `[-1.000, 1.000]`, mean `0.090`, std `0.614`

Latent storage: `[T, 4, 32, 32]` float32 per trajectory, scaling-factor pre-applied. Cache integrity verified bit-equivalent via `precompute_latents.py --verify`. 5.6 MB total latents.

## Smoke test (Phase 3)

200 steps, bs=1, no grad accumulation, NanoWM-S/2 on cached latents:

| Metric | Value |
|---|---|
| Loss first half → second half | 0.586 → 0.493 |
| Step rate | 13.0 steps/s |
| 30k-step extrapolation | ~40 min |
| NaN/Inf count | 0 |
| `check_smoke.py` | 4 / 4 passes |

## Training (Phase 5)

Stopped at step 5,000 — val_loss had clearly plateaued and started overfitting (small dataset).

| Metric | Value |
|---|---|
| Configured max_steps | 30,000 |
| **Actual steps trained** | **5,000** |
| Wall-clock | ~25 min |
| Step rate (mean) | 3.4 steps/s |
| Effective batch size | 4 (bs=1 × grad_accum=4) |
| Mixed precision | bf16 (master fp32) |
| LR | 1e-4 |
| Optim updates per epoch | 2 (8 train slices / grad_accum 4) |
| Total epochs traversed | ~2,500 |

Val loss trajectory (logged every ~500 steps):

| step | val_loss |
|---|---|
| 499 | 0.4052 |
| 999 | 0.3888 |
| 1499 | 0.3896 |
| 1999 | 0.3890 |
| **2499** | **0.3884** (best) |
| 2999 | 0.3898 |
| 3499 | 0.3892 |
| 3999 | 0.3949 |
| 4499 | 0.3993 |

Train loss: 0.493 (step 99) → 0.366 (step 4999). Train loss continued to drop while val plateaued — overfitting on the 8-trajectory train set.

Checkpoints (`$RESULTS_DIR/<run>/checkpoints/`):
- `latest/latest-epoch=2499-step=5000.ckpt` (812 MB) — final stopping point
- No `across_timesteps/` snapshots (their interval defaults to 10k steps; we stopped before)

## Rollouts (Phase 6)

`demo/` — 6 GIFs side-by-side prediction-vs-ground-truth from held-out val trajectory:

```
demo/sample_0000_compare.gif  (5.8 MB)
demo/sample_0001_compare.gif
demo/sample_0002_compare.gif
demo/sample_0003_compare.gif
demo/sample_0004_compare.gif
demo/sample_0005_compare.gif
```

Each is a 50-frame rollout with 4 context frames + 50 DDIM steps per generated frame. Resolution 256×256 per side (512×256 total).

Generated via `scripts/rollout_demo.py`, which is a thin Python wrapper over upstream's `src/sample/rollout.py` (it does the dfot_sample autoregressive loop + VAE decode), then converts the comparison MP4s to GIFs via imageio.

## Where the value is

What the POC delivers (engineering completeness):
- End-to-end pipeline from raw TartanDrive rosbags to demo GIFs, fully scripted and reproducible from a single `bash scripts/run_full_pipeline.sh`.
- Single consolidated upstream patch (~150 lines) handles the 7 specific compatibility issues we hit; re-applies automatically on a fresh clone.
- 47 tests stay green throughout — DataSource construction, dtypes, ranges, factory dispatch, frame/action alignment property test through `WorldModelDataset` consumer at `frame_interval=2`, cached-latent passthrough, preprocess split + finalize, force-min-val.
- `check_smoke.py` codifies pass criteria as a binary gate, not vibes; `training_health.py` for AM/PM check-ins.
- Resumable everywhere: stream-extract bags (skip-if-N-bags + tar `-k`), bag→tensor convert (skip-if-converted), preprocess (append + canonical re-numbering), latent precompute (skip-if-exists), training (`experiment.resume_from_checkpoint`).
- Cached-latent path delivers 2.8× speedup over on-the-fly VAE.

What the POC does NOT deliver (scope-limited):
- A useful trained model. 8 train trajectories from one date is not enough data — val_loss plateaued by step 2500 and overfit beyond that. The demo GIFs reflect a model that has memorized one driving scene's distribution, not a generalizing world model. To get a real result, pull more shards (each ~70 GB) for diversity across dates/terrain.
- No best-by-val checkpoint policy (deferred — upstream defaults: latest every 1k + every-10k snapshots).
- FID/FVD eval not run (disabled via `evaluation.metrics.evaluate=false` for cached-latent runs — upstream's metrics callback compares pixel-pred against latent-gt; needs latent-decode in the callback, deferred).
- No cloud bail-out — deliberately scoped out per plan.

## Repro

```bash
git checkout poc-v1
uv sync --extra dev
git clone https://github.com/simchowitzlabpublic/nano-world-model.git
git -C nano-world-model checkout $(cat UPSTREAM_PIN)
bash scripts/apply_upstream_patches.sh
uv run pytest                                       # 47 pass in ~1s

export DATASET_DIR=~/data/nanowm
export RESULTS_DIR=~/results/nanowm
bash scripts/download_shard.sh 20210828_heightmaps_1.tar.gz 10
FRESH=1 bash scripts/run_full_pipeline.sh
# -> demo/ contains the GIFs; $RESULTS_DIR/<run>/ contains the checkpoint
```
