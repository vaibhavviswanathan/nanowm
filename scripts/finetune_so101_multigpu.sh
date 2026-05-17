#!/usr/bin/env bash
# SO-101 finetune (8x A100 / 8x H100). Starts from the pretrain checkpoint
# on HF Hub, finetunes on the user's self-collected task mix.
#
# Defaults (see src/configs/experiment/so101_finetune.yaml):
#   lr=1e-5, max_steps=8000, val_every=1000, evaluate=false, bs=4/GPU
#
# Prerequisites:
#   1. Pretrain stats file exists OR pass STATS_PATH explicitly.
#   2. Pretrain checkpoint either present locally OR on HF (downloads).
#   3. Set HF_TOKEN + HF_MODEL_REPO_FT for finetune ckpt streaming.
#
# What this script does (single command, no manual prep needed):
#   1. Pre-fetches the 4 finetune source datasets from HF Hub.
#   2. Computes combined action mean/std stats if not already on disk.
#   3. Downloads the pretrain checkpoint from HF Hub if not local.
#   4. Launches DDP finetune across all visible GPUs, streaming new ckpts
#      to HF_MODEL_REPO_FT every 5 minutes.
#
# Required env:
#   RESULTS_DIR=$HOME/results/nanowm
#   HF_TOKEN=hf_...                          (read+write scope)
#   HF_MODEL_REPO=vaiviswanathan/so101-wm   (source of pretrain ckpt)
#   HF_MODEL_REPO_FT=vaiviswanathan/so101-wm-ft  (destination for finetune)
#
# Optional env:
#   PRETRAIN_CKPT=/path/to/ckpt.ckpt        (skip HF download if present)
#   STEPS=8000  BATCH_SIZE_PER_GPU=4  GRAD_ACCUM=1  NUM_FRAMES=16
#   STATS_PATH=$RESULTS_DIR/so101_finetune_stats.json
#
# Usage:
#   nohup bash scripts/finetune_so101_multigpu.sh > finetune.out 2>&1 &
#   tail -f finetune.out
#
# ETA: 8000 steps × eff_bs=32 ≈ ~1 hr on 8xA100 80GB, ~45min on 8xH100.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

: "${RESULTS_DIR:?RESULTS_DIR not set (e.g. export RESULTS_DIR=$HOME/results/nanowm)}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO not set (e.g. vaiviswanathan/so101-wm — source of pretrain ckpt)}"
mkdir -p "${RESULTS_DIR}"

# Canonical finetune source list — keep in sync with FINETUNE_SOURCES in
# src/wm_datasets/data_source/manipulation/so101.py.
FINETUNE_SOURCES=(
    "ofcourseistillloveyou/so101_recording_20260516_105727"
    "ofcourseistillloveyou/so101_recording_strawberry_20260516_161110"
    "ofcourseistillloveyou/so101_recording_marshmellow_40ep"
    "ofcourseistillloveyou/so101_recording_oreo_40ep"
)

# --- Step A: pre-download finetune datasets ----------------------------
# v3.0 LeRobot datasets aren't always pulled on-demand by the
# LeRobotDataset constructor — pre-fetching with `hf download` guarantees
# the parquets are in the HF cache before LeRobotDataset opens them.
echo "[finetune] pre-fetching ${#FINETUNE_SOURCES[@]} finetune datasets..."
for repo in "${FINETUNE_SOURCES[@]}"; do
    echo "  -> ${repo}"
    uv run --project "${REPO_ROOT}" hf download --repo-type dataset "${repo}" >/dev/null
done
echo "[finetune] datasets cached."

# --- Step B: compute combined finetune stats if missing -----------------
STATS_PATH="${STATS_PATH:-${RESULTS_DIR}/so101_finetune_stats.json}"
if [ ! -f "${STATS_PATH}" ]; then
    echo "[finetune] computing combined action stats over the finetune mix..."
    uv run --project "${REPO_ROOT}" python "${REPO_ROOT}/scripts/compute_so101_stats.py" \
        --sources "${FINETUNE_SOURCES[@]}" \
        --output "${STATS_PATH}" \
        --stride 5
    echo "[finetune] wrote ${STATS_PATH}"
else
    echo "[finetune] reusing existing stats: ${STATS_PATH}"
fi

# --- Pretrain checkpoint -------------------------------------------------
# Pull from HF if not already on disk. The pretrain syncer uploads to
# checkpoints/latest.ckpt as the fixed remote path.
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${RESULTS_DIR}/pretrain_latest.ckpt}"
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[finetune] pretrain ckpt not local; downloading from ${HF_MODEL_REPO}..."
    : "${HF_TOKEN:?HF_TOKEN required to pull pretrain ckpt from HF}"
    uv run hf download "${HF_MODEL_REPO}" checkpoints/latest.ckpt \
        --local-dir "${RESULTS_DIR}/_hf_cache" --repo-type model
    # `hf download` resolves to <local-dir>/<remote-path>
    cp -v "${RESULTS_DIR}/_hf_cache/checkpoints/latest.ckpt" "${PRETRAIN_CKPT}"
fi
echo "[finetune] resuming from: ${PRETRAIN_CKPT}"

# --- DDP topology --------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
echo "[ddp] detected ${NUM_GPUS} GPUs"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

STEPS="${STEPS:-8000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
COMPILE="${COMPILE:-true}"
VAL_EVERY="${VAL_EVERY:-1000}"

EFFECTIVE_BS=$((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "[ddp] effective batch = ${BATCH_SIZE_PER_GPU} × ${NUM_GPUS} GPU × ${GRAD_ACCUM} accum = ${EFFECTIVE_BS}"

export WANDB_MODE="${WANDB_MODE:-disabled}"

# --- HF checkpoint streaming for the finetune run -----------------------
# Uses a DIFFERENT repo than the pretrain source so we don't overwrite
# pretrain artifacts. The syncer script auto-detects the most-recent run
# dir under RESULTS_DIR — we'll override its HF_MODEL_REPO env var for the
# spawned child so finetune ckpts land in the finetune repo.
HF_SYNCER_PID=""
if [ -n "${HF_MODEL_REPO_FT:-}" ]; then
    echo "[finetune] HF streaming -> ${HF_MODEL_REPO_FT}"
    HF_MODEL_REPO="${HF_MODEL_REPO_FT}" bash "${REPO_ROOT}/scripts/hf_setup.sh"

    HF_MODEL_REPO="${HF_MODEL_REPO_FT}" nohup bash "${REPO_ROOT}/scripts/sync_checkpoints_hf.sh" \
        >> "${RESULTS_DIR}/sync_ft.out" 2>&1 &
    HF_SYNCER_PID=$!
    echo "[finetune] syncer pid=${HF_SYNCER_PID} log=${RESULTS_DIR}/sync_ft.out"
    trap "echo '[finetune] stopping syncer ${HF_SYNCER_PID}'; kill ${HF_SYNCER_PID} 2>/dev/null || true" EXIT
fi

cd "${UPSTREAM_DIR}"

# CRITICAL: wrap the checkpoint path in Hydra-style single quotes so the
# `=` chars in `latest-epoch=N-step=M.ckpt` aren't parsed as new overrides.
uv run --project "${REPO_ROOT}" python src/main.py \
    experiment=so101_finetune \
    dataset=manipulation/so101_finetune_mix \
    model=nanowm_b2 \
    model.num_frames="${NUM_FRAMES}" \
    experiment.training.max_steps="${STEPS}" \
    experiment.training.batch_size="${BATCH_SIZE_PER_GPU}" \
    experiment.training.gradient_accumulation="${GRAD_ACCUM}" \
    experiment.training.val_every_n_steps="${VAL_EVERY}" \
    experiment.infra.num_workers="${NUM_WORKERS}" \
    experiment.infra.compile="${COMPILE}" \
    experiment.infra.mixed_precision=true \
    experiment.infra.vae_precision=fp32 \
    dataset.loader.stats_path="${STATS_PATH}" \
    "experiment.resume_from_checkpoint='${PRETRAIN_CKPT}'" \
    logger.name=tensorboard \
    wandb.enabled=false \
    "$@"
