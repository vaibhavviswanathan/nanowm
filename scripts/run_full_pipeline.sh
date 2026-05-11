#!/usr/bin/env bash
# Convert + preprocess + latent-encode + train + demo, in sequence.
#
# Intended to fire after a download_shard.sh has produced bags. Each step is
# resumable on its own, so re-runs are cheap.
#
# Usage:
#   DATASET_DIR=~/data/nanowm RESULTS_DIR=~/results/nanowm \
#     bash scripts/run_full_pipeline.sh [SHARD_NAME]
# SHARD_NAME defaults to 20210828_heightmaps_1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SHARD_STEM="${1:-20210828_heightmaps_1}"
FRESH="${FRESH:-0}"   # FRESH=1 wipes tartandrive/ + tartandrive_latents/ first

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

BAGS_DIR="${DATASET_DIR}/raw_bags/${SHARD_STEM}/${SHARD_STEM}"
STAGING_DIR="${DATASET_DIR}/tartandrive_staging"
DATA_DIR="${DATASET_DIR}/tartandrive"
LATENTS_DIR="${DATASET_DIR}/tartandrive_latents"

if [ "${FRESH}" = "1" ]; then
    echo "FRESH=1: removing ${DATA_DIR} and ${LATENTS_DIR} (staging retained)"
    rm -rf "${DATA_DIR}" "${LATENTS_DIR}"
fi

echo "=== [1/5] Convert bags -> staging"
uv run python scripts/convert_bags.py \
    --bags "${BAGS_DIR}"/*.bag \
    --out_dir "${STAGING_DIR}" \
    --resolution 256 --decimate 3

echo
echo "=== [2/5] Preprocess: staging -> train/val + stats.json"
uv run python scripts/preprocess_tartandrive.py \
    --staging_dir "${STAGING_DIR}" \
    --out_dir "${DATA_DIR}" \
    --val_fraction 0.1 --seed 42 --mode move

echo
echo "=== [3/5] Precompute latents"
uv run python scripts/precompute_latents.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${LATENTS_DIR}" \
    --batch_size 16

echo
echo "=== [4/5] Train 30k steps on cached latents"
bash scripts/train.sh \
    dataset.loader.latents_path="${LATENTS_DIR}" \
    dataset.loader.use_cached_latents=true

echo
echo "=== [5/5] Rollout demo from latest checkpoint"
RUN_DIR="$(ls -dt ${RESULTS_DIR}/*/ | head -1)"
CFG="${RUN_DIR}/.hydra/config.yaml"
CKPT="$(ls ${RUN_DIR}/checkpoints/latest/*.ckpt | head -1)"
uv run python scripts/rollout_demo.py \
    --config "${CFG}" \
    --checkpoint "${CKPT}" \
    --out_dir "${REPO_ROOT}/demo" \
    --num_samples 6 --rollout_length 50 --history_length 4

echo
echo "=== All phases complete. Run ${RUN_DIR}, demo ${REPO_ROOT}/demo"
