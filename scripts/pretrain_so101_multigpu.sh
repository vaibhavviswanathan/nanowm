#!/usr/bin/env bash
# Multi-GPU pretrain (8x H100 / 8x A100 40GB / 8x A100 80GB / 8x L40S etc.)
#
# Lightning DDP is already wired in upstream's train_experiment.Trainer
# (devices=torch.cuda.device_count(), strategy=ddp_find_unused_parameters_false).
# Just run with all GPUs visible — the script auto-detects count.
#
# Effective batch = batch_size_per_gpu × num_gpus.
# Default: 4 per GPU × 8 = 32 effective (matches PLAN.md single-H100 target,
# so LR stays at 1e-4 unchanged). At bs=4/GPU + B/2 + 16f, peak memory is
# ~10 GB/GPU — fits 40GB cards comfortably.
#
# Bump BATCH_SIZE_PER_GPU to 8 for effective_bs=64 on 80GB cards if you want
# a beefier batch (linear-scaling rule: lr ~1.5-2x in that case).
#
# ETA at 8 GPUs (30k steps default, eff_bs=32):
#   8x H100 80GB:    ~1.5-2 hr   (~$25-40 on PI)
#   8x A100 80GB:    ~2-3 hr     (~$25-40)
#   8x A100 40GB:    ~2-3 hr     (~$15-25, cost minimum)
# Multiply by ~2.3x if you override to STEPS=70000.
#
# Usage:
#   nohup bash scripts/pretrain_so101_multigpu.sh > pretrain.out 2>&1 &
#   tail -f pretrain.out
#
# Optional: stream checkpoints to HuggingFace Hub during training. Set
# both env vars BEFORE launching; the script runs a pre-flight auth +
# write-access check (fails fast in seconds, not hours) and starts a
# background syncer that uploads new .ckpt files every 5 min.
#   export HF_TOKEN=hf_...                    # or `hf auth login`
#   export HF_MODEL_REPO=<user>/so101-wm
#   nohup bash scripts/pretrain_so101_multigpu.sh > pretrain.out 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${RESULTS_DIR:?RESULTS_DIR not set (e.g. export RESULTS_DIR=$HOME/results/nanowm)}"
mkdir -p "${RESULTS_DIR}"

STATS_PATH="${STATS_PATH:-${RESULTS_DIR}/so101_combined_stats.json}"
if [ ! -f "${STATS_PATH}" ]; then
    echo "ERROR: ${STATS_PATH} not found. Run scripts/compute_so101_stats_full.sh first." >&2
    exit 1
fi

# --- DDP / topology ------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
echo "[ddp] detected ${NUM_GPUS} GPUs"
if [ "${NUM_GPUS}" -lt 2 ]; then
    echo "WARN: only ${NUM_GPUS} GPU visible — use pretrain_so101_h100.sh (or _a100.sh) for single-GPU." >&2
fi

# NCCL knobs that PI's typical InfiniBand-less single-node setup likes:
#  - INFO logging surfaces NCCL init issues fast if the run hangs at startup
#  - ASYNC_ERROR_HANDLING bubbles a slow rank's OOM to the whole job
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Single-node DDP: P2P + SHM are the right transports; disable IB explicitly.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

# --- Override surface ---------------------------------------------------
# STEPS default of 30000 is calibrated against the TartanDrive precedent:
# exp02 (S/2, 9k frames) peaked at step 15k; exp06 (B/2, 9k frames) peaked
# at step 4k (bigger model overfits faster). SO-101 has ~90k frames so the
# linear extrapolation puts peak val_loss around 20-40k. PLAN.md's 70k was
# uncalibrated — past 30-40k you're more likely to be raising val_loss than
# improving it. Resume from checkpoint via experiment.resume_from_checkpoint
# if val_loss is still falling at 30k.
STEPS="${STEPS:-30000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"   # per rank; 8 ranks × 8 = 64 worker procs total
COMPILE="${COMPILE:-true}"
VAL_EVERY="${VAL_EVERY:-5000}"

EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "[ddp] effective batch = ${BATCH_SIZE_PER_GPU} × ${NUM_GPUS} GPU × ${GRAD_ACCUM} accum = ${EFFECTIVE_BS}"

export WANDB_MODE="${WANDB_MODE:-disabled}"

# --- HF checkpoint streaming (optional) --------------------------------
# Pre-flight check NOW so broken auth fails in seconds, not hours.
# Syncer is spawned just before the trainer launches and runs alongside.
HF_SYNCER_PID=""
if [ -n "${HF_MODEL_REPO:-}" ]; then
    echo "[multigpu] HF checkpoint streaming requested -> ${HF_MODEL_REPO}"
    bash "${REPO_ROOT}/scripts/hf_setup.sh"

    nohup bash "${REPO_ROOT}/scripts/sync_checkpoints_hf.sh" \
        >> "${RESULTS_DIR}/sync.out" 2>&1 &
    HF_SYNCER_PID=$!
    echo "[multigpu] checkpoint syncer started (pid=${HF_SYNCER_PID}, log=${RESULTS_DIR}/sync.out)"
    # Ensure syncer dies with the script
    trap "echo '[multigpu] stopping syncer ${HF_SYNCER_PID}'; kill ${HF_SYNCER_PID} 2>/dev/null || true" EXIT
fi

cd "${UPSTREAM_DIR}"

uv run --project "${REPO_ROOT}" python src/main.py \
    experiment=so101_pretrain \
    dataset=manipulation/so101_mix \
    model=nanowm_b2 \
    model.num_frames="${NUM_FRAMES}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE_PER_GPU}" \
    experiment.training.gradient_accumulation="${GRAD_ACCUM}" \
    experiment.training.val_every_n_steps="${VAL_EVERY}" \
    experiment.infra.num_workers="${NUM_WORKERS}" \
    experiment.infra.compile="${COMPILE}" \
    experiment.infra.mixed_precision=true \
    experiment.infra.vae_precision=fp32 \
    dataset.loader.stats_path="${STATS_PATH}" \
    dataset.loader.camera_dropout_p=0.15 \
    logger.name=tensorboard \
    wandb.enabled=false \
    "$@"
