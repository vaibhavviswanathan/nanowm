# nanowm — TartanDrive × Nano World Model POC

Train a video world model on CMU's TartanDrive 1.0 off-road ATV dataset using
the [nano-world-model](https://github.com/simchowitzlabpublic/nano-world-model)
codebase. The goal is a working autoregressive video rollout: given 1–2 context
frames of off-road terrain plus a sequence of throttle/steering actions,
generate plausible future frames that track the action conditioning.

## Repository layout

```
nanowm/
├── README.md                                 # this file
├── CLAUDE.md                                 # re-create / re-run notes
├── pyproject.toml                            # uv-managed deps (CUDA 12.1 wheels)
├── UPSTREAM_PIN                              # SHA of nano-world-model we build against
├── nano-world-model/                         # upstream checkout (gitignored)
├── patches/
│   └── 01-upstream-integration.patch         # applied by apply_upstream_patches.sh
├── scripts/
│   ├── apply_upstream_patches.sh             # patch + symlink scaffold into upstream
│   ├── download_shard.sh                     # stream-extract rosbags incrementally
│   ├── convert_bags.py                       # rosbags -> per-traj scaffold format
│   ├── preprocess_tartandrive.py             # staging -> train/val + stats
│   ├── precompute_latents.py                 # VAE pre-encoding (Phase 4)
│   ├── smoke_test.sh                         # 200-step sanity run (Phase 3)
│   ├── train.sh                              # full POC training (Phase 5)
│   └── rollout_demo.py                       # GIF demos (Phase 6)
├── src/wm_datasets/data_source/offroad/
│   ├── __init__.py
│   └── tartandrive.py                        # the DataSource
├── src/configs/dataset/offroad/
│   ├── base.yaml
│   └── tartandrive.yaml
├── tests/                                    # 46 tests, ~1s, runs without GPU or bags
└── docs/
    ├── phase0_findings.md                    # upstream API notes
    ├── dataset_notes.md                      # TartanDrive size/format reality check
    └── factory_patch.md                      # superseded; points at the patch file
```

The `src/` tree mirrors upstream's. `scripts/apply_upstream_patches.sh`
symlinks our `offroad` dirs into the upstream checkout so the imports
`from src.wm_datasets.data_source.offroad...` resolve to our tracked files.

## Environment

```bash
# Python 3.10 via uv (mirrors upstream's environment.yml)
uv sync --extra dev

# Run anything via `uv run`; or activate the venv if you prefer:
source .venv/bin/activate
```

CUDA 12.1 wheels are pulled for torch + torchvision. The 16 GB RTX 3080 Laptop
GPU is plenty for the S/2 model at 256² with bf16.

## One-time upstream setup

```bash
# Clone the pinned upstream (gitignored; not part of this repo)
git clone https://github.com/simchowitzlabpublic/nano-world-model.git
git -C nano-world-model checkout "$(cat UPSTREAM_PIN)"
bash scripts/apply_upstream_patches.sh
```

The patch lazy-imports `lerobot` (so the `diffusers==0.24.0` pin holds without
installing it), adds a `tartandrive` branch to the data-source factory, and
extends the kwarg forwarding in `world_model_dataset.py`. See
`docs/phase0_findings.md` for the source-of-truth analysis.

## Data pipeline (Phase 1)

TartanDrive 1.0 is published as 23 rosbag tarballs (69–127 GB each, ~2 TB
total) on CMU's Swift bucket. We pull *incrementally* — one shard, stream-stop
once we have enough bags, never store the full tarball on disk.

```bash
export DATASET_DIR=~/data/nanowm
mkdir -p "$DATASET_DIR"

# 1. Stream-extract 10 bags from the smallest shard (~69 GB on the wire, ~5 GB
#    on disk). The pipeline auto-terminates once 10 .bag files exist locally.
bash scripts/download_shard.sh 20210828_heightmaps_1.tar.gz 10

# 2. rosbag -> per-trajectory scaffold format (uint8 frames at 256² + cmd
#    actions at 10 Hz, zero-order-hold aligned to image timestamps).
#    Uses the pure-Python `rosbags` library — no ROS install needed because
#    we only consume standard message types (sensor_msgs/Image,
#    geometry_msgs/Twist).
uv run python scripts/convert_bags.py \
    --bags "$DATASET_DIR"/raw_bags/20210828_heightmaps_1/*.bag \
    --out_dir "$DATASET_DIR/tartandrive_staging" \
    --resolution 256 --decimate 2

# 3. Split staging into train/ and val/ with deterministic per-bag hashing,
#    compute action stats.
uv run python scripts/preprocess_tartandrive.py \
    --staging_dir "$DATASET_DIR/tartandrive_staging" \
    --out_dir "$DATASET_DIR/tartandrive" \
    --val_fraction 0.1 --seed 42 --mode move

# 4. (Optional) Free up disk:
rm -rf "$DATASET_DIR/raw_bags"
```

Disk budget per shard's worth of pulled data:

| Stage | Peak disk | Kept after step |
|---|---|---|
| Stream extract (10 bags) | ~5 GB | discard after step 2 |
| `convert_bags.py` | ~1 GB | the staging dir |
| `preprocess_tartandrive.py` | same | `$DATASET_DIR/tartandrive` |
| **Steady state per shard's worth** | | **~1 GB** |

To pull more data, run steps 1–3 again with a different shard / more bags.
Step 3 is re-run safe: new bags get appended (numbering continues), existing
trajectories are untouched.

Inspect a bag's topic list before converting:
```bash
uv run python scripts/convert_bags.py --inspect path/to/some.bag
```

## Phase 0 — Validate setup

```bash
uv run pytest                   # all 46 tests, ~1s
```

Tests cover: DataSource construction, dtypes, ranges, frame/action alignment
through the WorldModelDataset consumer at frame_interval=1 and =2, the
patched factory dispatch, cached-latent mode, and the staging-finalize logic
(deterministic splits, resumable re-runs, end-to-end loader compatibility).

## Phase 3 — Smoke test (after data is in)

```bash
export DATASET_DIR=~/data/nanowm
export RESULTS_DIR=~/results/nanowm
bash scripts/smoke_test.sh
```

Runs 200 steps with batch size 1 on NanoWM-S/2. `scripts/check_smoke.py`
codifies pass criteria: monotone-decreasing 50-step MA loss, no NaN/Inf,
loadable checkpoint, val rollout completed without error, and step-rate
extrapolation says we'll finish 30k steps in <6 days.

## Phase 4 — Latent precompute

```bash
uv run python scripts/precompute_latents.py \
    --data_dir "$DATASET_DIR/tartandrive" \
    --out_dir "$DATASET_DIR/tartandrive_latents"
```

Encodes every frame once with `stabilityai/sd-vae-ft-mse` (the same VAE
upstream training loads). Resumable — skip-if-exists. Each `latents.npy` is
written with a sidecar `latents.meta.json` capturing the source-frames SHA so
re-runs after a preprocess re-do are detected.

Flip `use_cached_latents=true` and re-run the smoke test to confirm the path
swap.

## Phase 5 — Full training

```bash
bash scripts/train.sh
```

Defaults: NanoWM-S/2, 30k steps, 8-frame clips, batch size 1 × grad-accum 4,
bf16. Checkpoint policy: every 1k steps, rolling 3 + best-by-val + a `latest`
symlink. Health-check helper:

```bash
uv run python scripts/training_health.py "$RESULTS_DIR/<run>"
```

prints loss slope over the last 1k steps, time/step, ETA, NaN count. Run it
AM/PM rather than watching live.

The **10k-step decision gate** decides continue vs debug vs cloud-pivot:
- Val loss falling smoothly + sampled rollout shows *any* terrain structure -> continue.
- Loss plateaued or rollouts pure noise -> stop, debug (suspect data alignment).
- Step rate so slow we'll miss the 5-day budget -> pivot to cloud.

## Phase 6 — Demo rollouts

```bash
uv run python scripts/rollout_demo.py \
    --checkpoint "$RESULTS_DIR/<run>/checkpoints/latest/checkpoint.ckpt" \
    --data_dir "$DATASET_DIR/tartandrive" \
    --out_dir ./demo \
    --num_episodes 6 --rollout_length 50 --context_frames 2
```

Side-by-side pred-vs-GT GIFs from held-out val trajectories. That's the POC
artifact.

## Phase 7 — Reproducibility close-out

```bash
git tag poc-v1 -m "TartanDrive nano-world-model POC, upstream=$(cat UPSTREAM_PIN)"
```

`CLAUDE.md` already documents the re-create flow.

## Realistic budget

| Phase | Active time | Wall-clock |
|---|---|---|
| 0 — Upstream API analysis | done | done |
| 1 — Data (1 shard, 10 bags) | 30 min | 1–2 hr download + 30 min convert |
| 2 — DataSource + tests | done | done (46 tests, 1s) |
| 3 — Smoke test | 30 min | 1 hr |
| 4 — Latent precompute | 30 min | 1–2 hr for 1 shard's worth |
| 5 — Training | 30 min/day checkin | 3–5 days on Tensorbook |
| 6 — Demo | 30 min | 1 hr |
| 7 — Close-out | 15 min | 15 min |

## References

- nano-world-model: <https://github.com/simchowitzlabpublic/nano-world-model>
- TartanDrive: <https://github.com/castacks/tartan_drive>, paper arXiv:2205.01791
- TartanDrive 2.0: <https://github.com/castacks/tartan_drive_2.0>,
  paper arXiv:2402.01913 (alternative data source if we expand)
- rosbags (pure-Python bag reader): <https://gitlab.com/ternaris/rosbags>
