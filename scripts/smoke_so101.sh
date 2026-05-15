#!/usr/bin/env bash
# Phase-3-style smoke test for SO-101 manipulation: 200 steps on the
# smallest source (svla_so101_pickplace, ~11.9k frames) at laptop-friendly
# bs=1 / grad_accum=4. Validates the pipeline end-to-end before any
# multi-hour run.
#
# Pass criteria (cribbed from TartanDrive's check_smoke.py):
#   - checkpoint written
#   - no NaN/Inf in train loss
#   - 50-step moving-avg loss trends down
#   - step rate gives plausible 70k-step ETA
#
# Usage:
#   RESULTS_DIR=~/results/nanowm bash scripts/smoke_so101.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
mkdir -p "${RESULTS_DIR}"

STATS_PATH="${STATS_PATH:-${RESULTS_DIR}/so101_smoke_stats.json}"
STEPS="${STEPS:-200}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
VAL_EVERY="${VAL_EVERY:-100}"

export WANDB_MODE=disabled

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
    experiment.training.log_every=20 \
    experiment.training.checkpointing.latest.every_n_train_steps=100 \
    experiment.training.checkpointing.across_timesteps.every_n_train_steps=200 \
    experiment.infra.mixed_precision=true \
    experiment.infra.vae_precision=fp32 \
    experiment.infra.compile=false \
    experiment.evaluation.metrics.evaluate=false \
    dataset.loader.source_repo_ids='["lerobot/svla_so101_pickplace"]' \
    dataset.loader.stats_path="${STATS_PATH}" \
    dataset.loader.camera_dropout_p=0.0 \
    logger.name=tensorboard \
    wandb.enabled=false \
    "$@"
