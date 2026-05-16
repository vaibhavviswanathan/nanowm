# Cloud pretrain bring-up

End-to-end recipe for running SO-101 manipulation pretrain on a fresh cloud
H100 (or A100). Tested locally through the smoke step; cloud commands are
analogous but with the H100-scale hyperparameters.

## Prereqs the cloud image must provide

- NVIDIA driver supporting CUDA 12.6+ (so torch 2.7+cu126 wheels resolve)
- ~50 GB free on a `RESULTS_DIR`-writable filesystem (~25 GB datasets + ~20 GB checkpoints + logs)
- `git`, `curl` (uv installer pulls itself); Python 3.12+ available (uv will install if not)
- Outbound network to GitHub, PyPI, HuggingFace Hub

A barebones Lambda Cloud H100 image or RunPod's `runpod/pytorch:2.4.0-py3.12`
base satisfies all of the above. Skip any image that pins torch < 2.7.

## The four commands

```bash
# 0. SSH in, then:
git clone -b manipulation-wm <your-fork-of-nanowm> nanowm
cd nanowm

# 1. Bring-up: clone upstream, apply patches, uv sync, run tests.  ~15-20 min
bash scripts/setup_cloud.sh

# 2. Pull all 4 datasets into HF cache.  ~25 GB, ~10-30 min depending on link.
bash scripts/download_so101_full.sh

# 3. Compute combined action stats over the mix.  ~5 min.
export RESULTS_DIR=$HOME/results/nanowm
bash scripts/compute_so101_stats_full.sh

# 4. Pretrain.  PLAN.md: 18-24 hr on H100/80GB, 36-48 hr on A100/40GB.
nohup bash scripts/pretrain_so101_h100.sh > pretrain.out 2>&1 &
tail -f pretrain.out
```

For A100/40GB, swap step 4 for `pretrain_so101_a100.sh` — same effective
batch size (32) but `bs=16 + grad_accum=2` to fit in 40GB.

For an **8× H100 (or 8× A100/80GB) node**, swap step 4 for
`pretrain_so101_h100_x8.sh`. Lightning DDP is already wired in upstream
(`devices=torch.cuda.device_count()`); the script auto-detects the GPU
count and sets `BATCH_SIZE_PER_GPU=4` so effective batch stays at 32
(matches single-H100 dynamics, no LR adjustment needed). ETA **~3-5 hr**.
NCCL env vars are pre-set for a single-node InfiniBand-less topology
(NVLink P2P only) which matches typical PI 8×H100 instances.

## Sanity checkpoints

After each step, verify before moving on:

- **After `setup_cloud.sh`**: it prints torch/lerobot versions + GPU name + runs the 82-test suite. If pytest fails, dependencies didn't resolve cleanly — STOP and inspect.
- **After `download_so101_full.sh`**: `du -sh ~/.cache/huggingface/hub/datasets--lerobot--svla_so10*` should show ~20-25 GB total.
- **After stats compute**: `$RESULTS_DIR/so101_combined_stats.json` exists, `action_mean`/`action_std` are 6-element lists, `n_samples` ~17k+.
- **First 200 training steps**: train_loss should drop from ~0.55 to ~0.20 (matches the local smoke result). If it stays flat or NaNs, something is wrong.

## Where outputs land

- `$RESULTS_DIR/<TIMESTAMP>-NanoWM-B-2-F16S1-so101/`
  - `checkpoints/latest/latest-epoch=*-step=*.ckpt` — overwritten every 1k steps
  - `checkpoints/across_timesteps/epoch=*-step=*.ckpt` — kept every 10k steps
  - `tb/` — TensorBoard event files (start `tensorboard --logdir=tb` to view)
  - `rank_0.log` — per-step loss + grad norm log
  - `.hydra/config.yaml` — resolved Hydra config (CEM planning needs this)

## Override surface

`pretrain_so101_h100.sh` reads these env vars (defaults in parens):

- `STEPS` (70000)
- `BATCH_SIZE` (32)
- `GRAD_ACCUM` (1)
- `NUM_FRAMES` (16)
- `NUM_WORKERS` (16)
- `COMPILE` (true)
- `VAL_EVERY` (5000)

Anything else can be passed as Hydra overrides on the CLI, e.g.
`bash scripts/pretrain_so101_h100.sh experiment.training.optimizer.lr=5e-5`.

## Pulling checkpoints back

```bash
# On local machine, after pretrain completes:
LATEST=$(ssh <cloud> "ls -1t \$HOME/results/nanowm | head -1")
rsync -avzP <cloud>:results/nanowm/${LATEST}/checkpoints/ ~/results/nanowm/${LATEST}/checkpoints/
```

The checkpoint files are ~3 GB each (B/2 + opt state). The `latest/` symlink
target is what you usually want.

## Known gaps (carry over from local)

- **No EMA**: PLAN.md asks for EMA decay 0.9999; upstream Lightning module doesn't have it. ~10-line patch in `train_experiment.configure_optimizers` if needed.
- **No cosine LR**: upstream hardcodes `name="constant"` with warmup. Constant-after-warmup is fine for diffusion at this scale, but cosine could buy a few % at the end.
- **`metrics.evaluate=true`** requires an I3D checkpoint at `$PRETRAINED_MODELS_DIR/i3d/i3d_torchscript.pt` for FVD/LPIPS. If the cloud image doesn't have it, FVD logging will error mid-train. Disable with `experiment.evaluation.metrics.evaluate=false` if you can live without it.
