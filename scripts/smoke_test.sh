#!/usr/bin/env bash
# Phase 3: 200-step smoke test on the TartanDrive dataset.
#
# Pass criteria are codified in scripts/check_smoke.py — run that after this
# completes (or pipe through it) to gate the next phase.
#
# Goal of this script: confirm the DataSource loads, training advances for a
# few hundred steps without OOM/NaN, validation runs, and a checkpoint saves.
# Output will look like noise at 200 steps — that's expected.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${DATASET_DIR:?DATASET_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"

# Disable wandb for unattended POC runs. Override with WANDB_MODE=online to use it.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Hydra resolves dataset paths relative to invocation cwd; run from upstream
# root so `from src.* import ...` resolves.
cd "${UPSTREAM_DIR}"

uv run --project "${REPO_ROOT}" python src/main.py \
    experiment=tartandrive \
    dataset=offroad/tartandrive \
    model=nanowm_s2 \
    experiment.training.max_steps=200 \
    experiment.training.batch_size=1 \
    experiment.training.val_every_n_steps=100 \
    experiment.training.log_every=10 \
    experiment.training.gradient_accumulation=1 \
    experiment.infra.mixed_precision=true \
    experiment.infra.vae_precision=fp32 \
    experiment.infra.compile=false \
    "$@"

echo
echo "Smoke run complete. Validate with:"
echo "  uv run python scripts/check_smoke.py \$RESULTS_DIR/<latest_run>"
