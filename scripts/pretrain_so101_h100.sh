#!/usr/bin/env bash
# Full pretrain on an H100/80GB. Matches PLAN.md spec:
#   - NanoWM-B/2, 16 frames, 256², bf16
#   - bs=32 (no grad-accum), num_workers=16, torch.compile=true
#   - 70k steps, AdamW lr=1e-4 with 2k linear warmup
#   - PLAN.md ETA: 18-24 hr wall-clock
#
# Background-friendly: tails to a log file and detaches.
#
# Usage (foreground):
#   bash scripts/pretrain_so101_h100.sh
#
# Usage (background, recommended):
#   nohup bash scripts/pretrain_so101_h100.sh > pretrain.out 2>&1 &
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

# --- Override surface (env vars; can also pass Hydra overrides via "$@") ---
STEPS="${STEPS:-70000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
COMPILE="${COMPILE:-true}"
VAL_EVERY="${VAL_EVERY:-5000}"

# wandb defaults to disabled; flip via WANDB_MODE=online + WANDB_API_KEY if desired.
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "${UPSTREAM_DIR}"

uv run --project "${REPO_ROOT}" python src/main.py \
    experiment=so101_pretrain \
    dataset=manipulation/so101_mix \
    model=nanowm_b2 \
    model.num_frames="${NUM_FRAMES}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE}" \
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
