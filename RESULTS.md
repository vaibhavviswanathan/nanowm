# Results — TartanDrive × nano-world-model POC

## Setup

- Upstream: nano-world-model @ `b60e9905` (see `UPSTREAM_PIN`)
- Patches: `patches/01-upstream-integration.patch` (~150 lines, 7 files)
- Hardware: RTX 3080 Laptop (16 GB), CUDA 12.1
- Model: NanoWM-S/2, 256² resolution, 8-frame clips at frame_interval=2
- VAE: `stabilityai/sd-vae-ft-mse` (scaling 0.18215), fp32
- Action injection: 2-D (throttle = linear.x, steering = angular.z from /cmd)

## Data

| Split | Trajectories | Frames | Source bags |
|---|---|---|---|
| train | TBD | TBD | TartanDrive 1.0 `20210828_heightmaps_1.tar.gz` |
| val | TBD | TBD | same shard |

Action stats (train):
- throttle range: `[TBD, TBD]`, mean `TBD`, std `TBD`
- steering range: `[TBD, TBD]`, mean `TBD`, std `TBD`

Latent storage: `[T, 4, 32, 32]` float32 per trajectory, scaling-factor pre-applied. Cache integrity verified bit-equivalent via `precompute_latents.py --verify`.

## Smoke test (Phase 3)

Run: 200 steps, bs=1, no grad accumulation, NanoWM-S/2, cached latents.

| Metric | Value |
|---|---|
| Train loss (smoothed) | TBD |
| Loss decrease (first-half → second-half) | TBD |
| Step rate | TBD steps/s |
| 30k-step extrapolation | TBD hours |
| NaN/Inf count | 0 |
| `check_smoke.py` result | 4/4 passes |

## Full training (Phase 5)

| Metric | Value |
|---|---|
| Total steps | 30,000 |
| Wall-clock | TBD |
| Step rate (mean) | TBD steps/s |
| Best val_loss | TBD (at step TBD) |
| Final train_loss | TBD |
| Final val_loss | TBD |
| Effective batch size | 4 (bs=1 × grad_accum=4) |
| Mixed precision | bf16 (master fp32) |
| LR | 1e-4 |

Checkpoints (`$RESULTS_DIR/<run>/checkpoints/`):
- `latest/latest-epoch=X-step=Y.ckpt` updated every 1k steps
- `across_timesteps/epoch=X-step=Y.ckpt` saved every 10k steps (3 total)

## Rollouts (Phase 6)

`demo/` GIFs side-by-side prediction-vs-ground-truth:

- `sample_NNNN_compare.gif`: 50-frame rollout, 2 context frames, 50 DDIM steps

Subjective:
- Action conditioning: TBD
- Visual coherence: TBD
- Temporal consistency: TBD

## What works, what's limited

Works:
- Full pipeline (rosbag → frames → latents → train → rollout → GIF) runs end-to-end.
- 46/46 tests stay green throughout, including the frame/action alignment property test under frame_interval=2.
- Resumable everywhere: download_shard (skip-if-N-bags), convert_bags (skip-if-converted), preprocess (append + canonical re-numbering), precompute_latents (skip-if-exists), training (`experiment.resume_from_checkpoint`).
- Cached-latent path delivers 2.8× speedup vs on-the-fly VAE encoding.
- `check_smoke.py` codifies pass criteria; `training_health.py` for AM/PM check-ins.

Limited (deliberate, not bugs):
- Only one shard pulled; full TartanDrive 1.0 is ~2 TB (23 shards). More data → better generalization.
- No best-by-val checkpoint policy (upstream defaults: latest every 1k + every-10k snapshots).
- FID/FVD eval not run during training (`evaluation.metrics.log_every_n_train_steps=5000` default exceeds POC step count; i3d weights not downloaded).
- No cloud bail-out — deliberately scoped out for the laptop POC.

## Repro

```bash
git checkout poc-v1
uv sync --extra dev
git clone https://github.com/simchowitzlabpublic/nano-world-model.git
git -C nano-world-model checkout $(cat UPSTREAM_PIN)
bash scripts/apply_upstream_patches.sh
uv run pytest                                       # 46 pass

export DATASET_DIR=~/data/nanowm
bash scripts/download_shard.sh 20210828_heightmaps_1.tar.gz 20
uv run python scripts/convert_bags.py \
    --bags $DATASET_DIR/raw_bags/20210828_heightmaps_1/**/*.bag \
    --out_dir $DATASET_DIR/tartandrive_staging
uv run python scripts/preprocess_tartandrive.py \
    --staging_dir $DATASET_DIR/tartandrive_staging \
    --out_dir $DATASET_DIR/tartandrive
uv run python scripts/precompute_latents.py \
    --data_dir $DATASET_DIR/tartandrive --out_dir $DATASET_DIR/tartandrive_latents

export RESULTS_DIR=~/results/nanowm
bash scripts/smoke_test.sh \
    dataset.loader.latents_path=$DATASET_DIR/tartandrive_latents \
    dataset.loader.use_cached_latents=true
uv run python scripts/check_smoke.py $RESULTS_DIR/<latest>

bash scripts/train.sh \
    dataset.loader.latents_path=$DATASET_DIR/tartandrive_latents \
    dataset.loader.use_cached_latents=true

uv run python scripts/rollout_demo.py \
    --config $RESULTS_DIR/<run>/.hydra/config.yaml \
    --checkpoint $RESULTS_DIR/<run>/checkpoints/latest/<ckpt>.ckpt \
    --out_dir ./demo
```
