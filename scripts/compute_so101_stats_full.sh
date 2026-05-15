#!/usr/bin/env bash
# Compute the combined-stats JSON over all four pretrain sources.
# Writes $RESULTS_DIR/so101_combined_stats.json which the pretrain configs
# consume via dataset.loader.stats_path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RESULTS_DIR:?RESULTS_DIR not set}"
mkdir -p "${RESULTS_DIR}"

cd "${REPO_ROOT}"

uv run python scripts/compute_so101_stats.py \
    --output "${RESULTS_DIR}/so101_combined_stats.json" \
    --stride 5
