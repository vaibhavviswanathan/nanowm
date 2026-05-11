#!/usr/bin/env bash
# Phase 3: 200-step smoke test on the new TartanDrive dataset.
#
# Goal: confirm the DataSource loads, the model trains for a few steps without
# OOM/NaN, validation runs, and a checkpoint saves. The output will look like
# noise — that's expected at 200 steps.

set -euo pipefail

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

python src/main.py \
    experiment=tartandrive \
    dataset=offroad/tartandrive \
    model=nanowm_s2 \
    experiment.training.max_steps=200 \
    experiment.training.batch_size=1 \
    experiment.infra.mixed_precision=bf16 \
    experiment.training.val_every_n_steps=100 \
    metrics.log_every_n_train_steps=999999 \
    "$@"

echo
echo "Smoke test complete. Check:"
echo "  - Loss decreased (look at tensorboard or stdout)"
echo "  - Checkpoint exists: \$RESULTS_DIR/<run>/checkpoints/latest/"
echo "  - No NaNs or OOMs"
