# nanowm — TartanDrive × nano-world-model POC

## Layout
- `nano-world-model/` — upstream checkout, pinned via `UPSTREAM_PIN` (not tracked here).
- `patches/` — patches applied to upstream after clone (via `scripts/apply_upstream_patches.sh`).
- `scripts/` — download, convert, preprocess, latents, smoke, train, rollout.
- `src/` — tracked DataSource + configs; symlinked into upstream's tree at apply time.
- `tests/` — 46 fast pytests for data layer (DataSource, factory, integration, preprocess).
- `docs/` — design notes (`phase0_findings.md`, `dataset_notes.md`).

## Conventions
- Python env via `uv`. Re-create: `uv sync --extra dev`. Run via `uv run <cmd>`.
- Upstream commit pinned in `UPSTREAM_PIN`. To re-clone:
  ```
  git clone https://github.com/simchowitzlabpublic/nano-world-model.git
  git -C nano-world-model checkout $(cat UPSTREAM_PIN)
  bash scripts/apply_upstream_patches.sh
  ```
- Data: `$DATASET_DIR` (default `~/data/nanowm`); results: `$RESULTS_DIR` (default `~/results/nanowm`). Both gitignored.
- Tests: `uv run pytest`. Run before any long-running script.

## Data pipeline (Phase 1)
TartanDrive 1.0 is 2 TB of rosbags. Incremental flow:
1. `bash scripts/download_shard.sh <SHARD> <N_BAGS>` — stream-extract N bags, never holds the full tarball.
2. `uv run python scripts/convert_bags.py --bags ... --out_dir <staging>` — rosbags -> scaffold format at 256² / 10 Hz. Pure-Python via `rosbags`; uses standard ROS messages only (no ROS install needed). The last bag in a stream-extract is often truncated — `convert_bags.py` skips bags with damaged headers automatically.
3. `uv run python scripts/preprocess_tartandrive.py --staging_dir <staging> --out_dir <out>` — hash-deterministic train/val split + stats.json. Re-run-safe; appends new bags.

Inspect bag topics: `uv run python scripts/convert_bags.py --inspect <bag>`

## Phase 3 — smoke test
```
DATASET_DIR=~/data/nanowm RESULTS_DIR=~/results/nanowm bash scripts/smoke_test.sh
uv run python scripts/check_smoke.py $RESULTS_DIR/<latest_run_dir>
```
Pass criteria (4): checkpoint saved, no NaN, loss decreased, 30k-step ETA <= 6 days.

## Phase 4 — latent precompute
```
uv run python scripts/precompute_latents.py \
    --data_dir $DATASET_DIR/tartandrive \
    --out_dir $DATASET_DIR/tartandrive_latents \
    --batch_size 16
# Verify cache integrity (bit-equivalent re-encode of 1%):
uv run python scripts/precompute_latents.py \
    --data_dir $DATASET_DIR/tartandrive \
    --out_dir $DATASET_DIR/tartandrive_latents --verify
```

Then run smoke test with `dataset.loader.latents_path=$DATASET_DIR/tartandrive_latents dataset.loader.use_cached_latents=true` to confirm the cached-latent path (~2.8x faster).

## Phase 5 — full training
```
DATASET_DIR=~/data/nanowm RESULTS_DIR=~/results/nanowm bash scripts/train.sh \
    dataset.loader.latents_path=$DATASET_DIR/tartandrive_latents \
    dataset.loader.use_cached_latents=true
```
Defaults: NanoWM-S/2, 30k steps, bs=1 × grad_accum=4, 8-frame clips. Resume from a previous run via `experiment.resume_from_checkpoint=$RESULTS_DIR/<run>/checkpoints/latest/<latest>.ckpt`. Monitor with `uv run python scripts/training_health.py $RESULTS_DIR/<run>` AM/PM.

## Phase 6 — demo rollouts
```
uv run python scripts/rollout_demo.py \
    --config $RESULTS_DIR/<run>/.hydra/config.yaml \
    --checkpoint $RESULTS_DIR/<run>/checkpoints/latest/<latest>.ckpt \
    --out_dir ./demo --num_samples 6 --rollout_length 50 --history_length 4
```
Invokes upstream `src/sample/rollout.py` then converts the produced `*_compare.mp4` files to GIFs.

## Don't do
- Don't commit `*.npy`, `data/`, `results/`, or `nano-world-model/`.
- Don't run training before `uv run pytest` is green.
- Don't change preprocessing without rebuilding latents — they're keyed by source-frames SHA in `latents.meta.json`.
- Don't install `lerobot` to "fix" the rt1 import — upstream is patched to lazy-load it. Installing it would force a newer `diffusers` and break the VAE training path.
