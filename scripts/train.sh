#!/usr/bin/env bash
# Phase 5: full POC training run on TartanDrive.
#
# Defaults targeted at a 16 GB laptop GPU. Override via env or CLI, e.g.
#   STEPS=50000 BATCH_SIZE=1 GRAD_ACCUM=8 bash scripts/train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

MODEL="${MODEL:-nanowm_s2}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"     # effective batch size = BATCH_SIZE * GRAD_ACCUM
NUM_FRAMES="${NUM_FRAMES:-8}"     # clip length per training step
VAL_EVERY="${VAL_EVERY:-2000}"
WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_MODE

cd "${UPSTREAM_DIR}"

uv run --project "${REPO_ROOT}" python src/main.py \
    experiment=tartandrive \
    dataset=offroad/tartandrive \
    model="${MODEL}" \
    model.num_frames="${NUM_FRAMES}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE}" \
    experiment.training.gradient_accumulation="${GRAD_ACCUM}" \
    experiment.training.val_every_n_steps="${VAL_EVERY}" \
    experiment.infra.mixed_precision=true \
    experiment.infra.vae_precision=fp32 \
    experiment.infra.compile=false \
    logger.name=tensorboard \
    wandb.enabled=false \
    experiment.evaluation.metrics.evaluate=false \
    "$@"
