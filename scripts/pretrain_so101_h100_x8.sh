#!/usr/bin/env bash
# Full pretrain on an 8x H100 / 8x A100 80GB node. Lightning DDP via
# upstream's already-wired `devices=torch.cuda.device_count(), strategy=ddp...`
# path — no extra launcher needed, just run with all 8 GPUs visible.
#
# Effective batch = batch_size_per_gpu × num_gpus.
# Default: 4 per GPU × 8 = 32 effective (matches PLAN.md single-H100 target,
# so LR stays at 1e-4 unchanged).
#
# Bump BATCH_SIZE_PER_GPU to 8 for effective_bs=64 if you want a beefier batch;
# the linear-scaling rule suggests lr ~1.5-2x in that case (override via Hydra).
#
# ETA: ~3-5 hr wall-clock for 70k steps depending on NCCL & data loader.
#
# Usage:
#   nohup bash scripts/pretrain_so101_h100_x8.sh > pretrain.out 2>&1 &
#   tail -f pretrain.out

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
STEPS="${STEPS:-70000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"   # per rank; 8 ranks × 8 = 64 worker procs total
COMPILE="${COMPILE:-true}"
VAL_EVERY="${VAL_EVERY:-5000}"

EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "[ddp] effective batch = ${BATCH_SIZE_PER_GPU} × ${NUM_GPUS} GPU × ${GRAD_ACCUM} accum = ${EFFECTIVE_BS}"

export WANDB_MODE="${WANDB_MODE:-disabled}"

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
