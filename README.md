# TartanDrive × Nano World Model — POC

Train a video world model on CMU's TartanDrive off-road ATV dataset using the
[nano-world-model](https://github.com/simchowitzlabpublic/nano-world-model) codebase.

## Goal

A working autoregressive video rollout: given 1–2 context frames of off-road
terrain plus a sequence of throttle/steering actions, generate plausible future
frames. The model should track action conditioning (e.g., turning left when
steering goes left) and produce visually coherent terrain.

## Why TartanDrive

- Real ATV driving with actual control inputs (throttle + steering = 2D action).
- ~5 hours, ~200k frames, ~630 trajectories — comparable in scale to
  nano-world-model's DINO-WM environments.
- 2D continuous actions match nano-world-model's PushT recipe; the simple
  "additive" injection mechanism is known to work well at this dim.
- Single test site = lower scene-generalization burden, faster convergence
  for a POC.

## Prerequisites

- nano-world-model cloned, `nanowm` conda env created and activated.
- The i3d torchscript downloaded (per nano-world-model README).
- Disk: ~30 GB for raw TartanDrive 1.0, ~5 GB for preprocessed frames at 256²,
  ~2 GB for cached VAE latents. Total ~40 GB headroom.
- GPU: 16 GB+ VRAM. The plan targets a Lambda Tensorbook (RTX 3080 Max-Q,
  16 GB) but works on any 16 GB+ card.

## File map

```
tartandrive_nanowm/
├── README.md                                   # this file
├── scripts/
│   ├── preprocess_tartandrive.py               # Phase 1: raw → frames+actions
│   ├── precompute_latents.py                   # Phase 4: frames → VAE latents
│   ├── rollout_demo.py                         # Phase 6: generate demo GIFs
│   ├── smoke_test.sh                           # Phase 3: 200-step sanity check
│   └── train.sh                                # Phase 5: full POC run
├── src/wm_datasets/data_source/offroad/
│   ├── __init__.py
│   └── tartandrive.py                          # custom DataSource
├── src/configs/dataset/offroad/
│   ├── base.yaml                               # shared offroad config
│   └── tartandrive.yaml                        # TartanDrive-specific config
└── docs/
    └── factory_patch.md                        # how to register the DataSource
```

The `src/` layout mirrors nano-world-model's tree — copy these into your repo
clone in the same paths.

## Plan of action

### Phase 0 — Setup (30 min)

```bash
# Inside your nano-world-model checkout:
conda activate nanowm

# Smoke-test that vanilla training works on a tiny subset before adding new code.
python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2 \
    experiment.training.max_steps=50
```

If that runs and saves a checkpoint, your environment is good. If not, fix
infra issues *before* adding TartanDrive on top.

### Phase 1 — Preprocess TartanDrive (half a day)

Download TartanDrive 1.0 from https://theairlab.org/datasets/. Then:

```bash
python scripts/preprocess_tartandrive.py \
    --raw_dir /path/to/tartandrive_raw \
    --out_dir $DATASET_DIR/tartandrive \
    --resolution 256 \
    --val_fraction 0.1
```

Output layout:
```
$DATASET_DIR/tartandrive/
├── train/
│   ├── traj_0001/
│   │   ├── frames.npy        # uint8 [T, 256, 256, 3]
│   │   ├── actions.npy       # float32 [T, 2]
│   │   └── meta.json         # {length, source_traj_id, fps}
│   └── ...
├── val/
│   └── ...
└── stats.json                # action min/max/mean/std for verification
```

⚠️ **Format adapter**: TartanDrive is distributed as `.pt` files (or ROS bags
in some releases). The script's `_load_raw_trajectory` function has TODOs
where you may need to adapt to your exact download. Check the script's
docstring before running.

After preprocessing, sanity-check that throttle/steering ranges look right
(both should be roughly [-1, 1]).

### Phase 2 — DataSource and configs (half a day)

The custom `DataSource` is at `src/wm_datasets/data_source/offroad/tartandrive.py`.
Copy it into your repo, then register it — see `docs/factory_patch.md`.

Drop the YAML configs at `src/configs/dataset/offroad/`.

### Phase 3 — Smoke test (1 hour)

```bash
bash scripts/smoke_test.sh
```

This runs 200 steps with batch size 1 on NanoWM-S/2. What "passing" looks like:

- ✅ Loss starts ~0.5–1.0 and decreases steadily.
- ✅ No NaNs, no OOMs.
- ✅ A checkpoint saves to `$RESULTS_DIR/<run>/checkpoints/latest/`.
- ✅ A validation rollout runs without erroring (it'll look like noise — that's fine).

If anything's off, fix here before committing to the long run. Most common
issues are action/frame misalignment in the DataSource or wrong action_dim.

### Phase 4 — Pre-encode VAE latents (2–4 hours)

Biggest single speedup on a Tensorbook. Encode every frame with the VAE once
and cache to disk so training only runs the diffusion transformer.

```bash
python scripts/precompute_latents.py \
    --data_dir $DATASET_DIR/tartandrive \
    --out_dir $DATASET_DIR/tartandrive_latents \
    --vae_config configs/vae/your_vae.yaml \
    --batch_size 16
```

Then flip `use_cached_latents: true` in `tartandrive.yaml`. The DataSource
will load latents instead of frames. Re-run the smoke test to confirm it still
works with cached latents.

### Phase 5 — Full POC training (3–5 days hands-off)

```bash
bash scripts/train.sh
```

Defaults: NanoWM-S/2, 30k steps, 8-frame clips, batch size 1 with gradient
accumulation, bf16 mixed precision, val every 2k, FID every 10k.

Monitor with `tensorboard --logdir $RESULTS_DIR/<run_dir>/tb`. Look at it once
a day, not constantly.

**Decision checkpoint at ~10k steps (~day 1.5):**

- Val loss falling smoothly, sampled rollout shows *some* terrain structure
  → continue.
- Loss plateaued or rollouts pure noise → stop, debug. Usually frame/action
  alignment.
- Step rate so slow you'll miss the 5-day budget → bail to cloud (next).

**Bail-out to cloud** if needed:

```bash
# Rent an A100 80GB on Lambda or RunPod (~$1–2/hr).
# Rsync preprocessed data + cached latents + latest checkpoint:
rsync -aP $DATASET_DIR/tartandrive_latents user@cloud:/data/
rsync -aP $RESULTS_DIR/<run_dir>/checkpoints/latest user@cloud:/results/

# Resume on the cloud box:
python src/main.py experiment=tartandrive dataset=offroad/tartandrive \
    model=nanowm_b2 \
    resume_from_checkpoint=/results/latest/checkpoint.ckpt
```

Cached latents make this transition cheap (~2 GB transfer instead of 30+).

### Phase 6 — Demo rollouts (2 hours)

```bash
python scripts/rollout_demo.py \
    --checkpoint $RESULTS_DIR/<run_dir>/checkpoints/latest/checkpoint.ckpt \
    --data_dir $DATASET_DIR/tartandrive \
    --out_dir ./demo \
    --num_episodes 6 \
    --rollout_length 50 \
    --context_frames 2
```

Produces side-by-side GIFs of predicted vs ground-truth video for 6 held-out
trajectories. That's your POC artifact.

## Realistic budget

| Phase | Active time | Wall-clock |
|---|---|---|
| 0 — Setup | 30 min | 30 min |
| 1 — Preprocess | 2 hr | 2–4 hr (download dominates) |
| 2 — Code | 4 hr | 4 hr |
| 3 — Smoke test | 30 min | 1 hr |
| 4 — Latent caching | 1 hr | 2–4 hr |
| 5 — Full training | 30 min checkins/day | 3–5 days |
| 6 — Demo | 1 hr | 2 hr |
| **Total** | **~12 hr active** | **~1 week** |

## Notes and assumptions

- Code paths assume nano-world-model's structure as documented in
  `docs/datasets/README.md` and `docs/training.md`. Where I'm guessing at
  exact API shapes (especially the `DataSource` base class, the VAE wrapper,
  and the rollout API), the relevant file has a `# TODO:` comment.
  Check those against your actual checkout.
- TartanDrive's exact on-disk format varies by release. The preprocessing
  script handles `.pt` per-trajectory tensors as the most common case; adapt
  `_load_raw_trajectory` if your download differs.
- Action normalization is left **off** because TartanDrive throttle/steering
  are nominally in [-1, 1] already. Verify with the printed `stats.json` after
  preprocessing — if the ranges are wildly different, flip
  `normalize_action: true` in the config.
- The model recommendation is S/2. If your run is on a beefier GPU and Phase 5
  finishes early, swap to B/2 by overriding `model=nanowm_b2` in `train.sh`.

## References

- nano-world-model: https://github.com/simchowitzlabpublic/nano-world-model
- TartanDrive 1.0: https://theairlab.org/datasets/ (paper: arXiv:2205.01791)
- TartanDrive 2.0: arXiv:2402.01913 (use this if you want more data later)
