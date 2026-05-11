#!/usr/bin/env bash
# Phase 5: full POC training run on TartanDrive.
#
# Defaults targeted at a 16 GB laptop GPU. Adjust if you have more headroom.

set -euo pipefail

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

# Override these from the CLI if needed, e.g.:
#   MODEL=nanowm_b2 STEPS=50000 bash scripts/train.sh
MODEL="${MODEL:-nanowm_s2}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"  # effective batch size = BATCH_SIZE * GRAD_ACCUM
CLIP_LEN="${CLIP_LEN:-8}"

python src/main.py \
    experiment=tartandrive \
    dataset=offroad/tartandrive \
    model="${MODEL}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE}" \
    experiment.training.gradient_accumulation_steps="${GRAD_ACCUM}" \
    experiment.training.clip_length="${CLIP_LEN}" \
    experiment.training.val_every_n_steps=2000 \
    metrics.log_every_n_train_steps=10000 \
    experiment.infra.mixed_precision=bf16 \
    experiment.infra.compile=false \
    "$@"
