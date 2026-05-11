#!/usr/bin/env bash
# exp02 chain: convert all 5-shard bags, fresh preprocess + latents, train 30k.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

STAGING_DIR="${DATASET_DIR}/tartandrive_staging"
DATA_DIR="${DATASET_DIR}/tartandrive"
LATENTS_DIR="${DATASET_DIR}/tartandrive_latents"

echo "=== [1/4] FRESH wipe of preprocessed + latents (raw bags + staging kept)"
rm -rf "${DATA_DIR}" "${LATENTS_DIR}"

echo
echo "=== [2/4] Convert all bags from 5 shards (skip-if-staged)"
# Glob all .bag files across all 5 shard dirs; convert_bags skips those already
# in staging.
ALL_BAGS=( "${DATASET_DIR}"/raw_bags/*/*/*.bag )
echo "  $(ls "${DATASET_DIR}"/raw_bags/*/*/*.bag | wc -l) bag files found"
uv run python scripts/convert_bags.py \
    --bags "${ALL_BAGS[@]}" \
    --out_dir "${STAGING_DIR}" \
    --resolution 256 --decimate 3

echo
echo "=== [3/4] Preprocess: split into train/val + stats"
uv run python scripts/preprocess_tartandrive.py \
    --staging_dir "${STAGING_DIR}" \
    --out_dir "${DATA_DIR}" \
    --val_fraction 0.1 --seed 42 --mode move

echo
echo "=== [4/4] Precompute latents"
uv run python scripts/precompute_latents.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${LATENTS_DIR}" \
    --batch_size 16

echo
echo "=== Train: 30k steps with cached latents"
bash scripts/train.sh \
    dataset.loader.latents_path="${LATENTS_DIR}" \
    dataset.loader.use_cached_latents=true
