#!/usr/bin/env bash
# Phase 7: reproducibility close-out.
#
# Snapshot the project state at a milestone (e.g. after a successful Phase 5
# training + Phase 6 demo) so future-us can re-create the exact run.
#
# Usage:
#   bash scripts/close_out.sh poc-v1 "TartanDrive nano-world-model POC v1"
#   bash scripts/close_out.sh  # uses default tag name + message

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TAG_NAME="${1:-poc-v1}"
TAG_MSG="${2:-TartanDrive nano-world-model POC}"
UPSTREAM_SHA="$(cat UPSTREAM_PIN)"

# 0. Verify state
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree has uncommitted changes. Commit or stash before tagging."
    git status --short
    exit 1
fi
if ! uv run pytest -q 2>&1 | tail -1 | grep -q "passed"; then
    echo "ERROR: tests are not green. Fix before tagging."
    exit 1
fi

# 1. Tag the commit
echo "Tagging ${TAG_NAME} at $(git rev-parse --short HEAD)"
git tag -a "${TAG_NAME}" -m "${TAG_MSG}

upstream=${UPSTREAM_SHA}
tests=$(uv run pytest -q 2>&1 | grep -E '[0-9]+ passed' | head -1)
"

# 2. Print a re-create recipe
cat <<EOF

Tag ${TAG_NAME} created locally. To push:
    git push origin ${TAG_NAME}

To re-create this exact state from a fresh clone of nanowm:
    git checkout ${TAG_NAME}
    uv sync --extra dev
    git clone https://github.com/simchowitzlabpublic/nano-world-model.git
    git -C nano-world-model checkout ${UPSTREAM_SHA}
    bash scripts/apply_upstream_patches.sh
    uv run pytest

Data pipeline (TartanDrive raw -> trained model):
    bash scripts/download_shard.sh 20210828_heightmaps_1.tar.gz 20
    uv run python scripts/convert_bags.py \\
        --bags \$DATASET_DIR/raw_bags/20210828_heightmaps_1/**/*.bag \\
        --out_dir \$DATASET_DIR/tartandrive_staging
    uv run python scripts/preprocess_tartandrive.py \\
        --staging_dir \$DATASET_DIR/tartandrive_staging \\
        --out_dir \$DATASET_DIR/tartandrive
    uv run python scripts/precompute_latents.py \\
        --data_dir \$DATASET_DIR/tartandrive \\
        --out_dir \$DATASET_DIR/tartandrive_latents
    bash scripts/train.sh \\
        dataset.loader.latents_path=\$DATASET_DIR/tartandrive_latents \\
        dataset.loader.use_cached_latents=true
EOF
