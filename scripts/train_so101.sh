#!/usr/bin/env bash
# SO-101 manipulation world-model training launcher. Defaults run the
# pretrain on the four SmolVLA-paired LeRobot datasets at NanoWM-B/2,
# 16-frame clips, batch 32 on H100.
#
# Override via env or CLI, e.g.
#   STEPS=10000 BATCH_SIZE=16 bash scripts/train_so101.sh
# or to fine-tune, pass through:
#   bash scripts/train_so101.sh experiment=so101_finetune \
#       dataset=manipulation/so101_task \
#       dataset.loader.repo_id=<user>/so101_my_task \
#       experiment.resume_from_checkpoint=<path>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${RESULTS_DIR:?RESULTS_DIR not set}"

MODEL="${MODEL:-nanowm_b2}"
EXPERIMENT="${EXPERIMENT:-so101_pretrain}"
DATASET="${DATASET:-manipulation/so101_mix}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEPS="${STEPS:-70000}"
VAL_EVERY="${VAL_EVERY:-5000}"
STATS_PATH="${STATS_PATH:-${RESULTS_DIR}/so101_combined_stats.json}"
WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_MODE

cd "${UPSTREAM_DIR}"

uv run --project "${REPO_ROOT}" python src/main.py \
    experiment="${EXPERIMENT}" \
    dataset="${DATASET}" \
    model="${MODEL}" \
    model.num_frames="${NUM_FRAMES}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE}" \
    experiment.training.gradient_accumulation="${GRAD_ACCUM}" \
    experiment.training.val_every_n_steps="${VAL_EVERY}" \
    dataset.loader.stats_path="${STATS_PATH}" \
    logger.name=tensorboard \
    wandb.enabled=false \
    "$@"
