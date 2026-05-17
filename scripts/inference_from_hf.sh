#!/usr/bin/env bash
# Generate prediction-vs-GT rollout demos using a checkpoint hosted on
# HuggingFace Hub. Pulls the ckpt + config from HF, overrides the config
# to use just one source dataset (so we don't re-download 25 GB), and
# runs scripts/rollout_demo.py end-to-end.
#
# Designed for an inference-only machine (single GPU is plenty; B/2
# inference uses ~5 GB VRAM). Don't run this on the same box as a live
# pretrain — DDP grabs all GPUs.
#
# Usage:
#   export HF_MODEL_REPO=vaiviswanathan/so101-wm
#   export HF_TOKEN=hf_<read-token>          # optional for public repos
#   bash scripts/inference_from_hf.sh
#
# Override surface (env vars):
#   HF_REVISION       — git rev/tag (default: main)
#   CKPT_PATH         — remote path within the repo (default: checkpoints/latest.ckpt)
#   NUM_SAMPLES       — how many rollouts to generate (default: 4)
#   ROLLOUT_LENGTH    — frames to predict (default: 16)
#   HISTORY_LENGTH    — context frames (default: 4)
#   DDIM_STEPS        — diffusion sampling steps (default: 25)
#   SOURCE_REPO_ID    — which LeRobot source to draw val episodes from
#                       (default: lerobot/svla_so101_pickplace)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${HF_MODEL_REPO:?HF_MODEL_REPO not set (e.g. vaiviswanathan/so101-wm)}"
HF_REVISION="${HF_REVISION:-main}"
CKPT_PATH="${CKPT_PATH:-checkpoints/latest.ckpt}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-16}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
DDIM_STEPS="${DDIM_STEPS:-25}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-lerobot/svla_so101_pickplace}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/demo}"

WORK_DIR=$(mktemp -d /tmp/so101-infer.XXXXXX)
echo "[infer] work dir: ${WORK_DIR}"

cd "${REPO_ROOT}"

# --- 1. Pull ckpt + config + stats from HF ---
echo "[infer] downloading from ${HF_MODEL_REPO}@${HF_REVISION}"
uv run hf download "${HF_MODEL_REPO}" config.yaml \
    --revision "${HF_REVISION}" --local-dir "${WORK_DIR}" >/dev/null
uv run hf download "${HF_MODEL_REPO}" "${CKPT_PATH}" \
    --revision "${HF_REVISION}" --local-dir "${WORK_DIR}" >/dev/null

# stats file is needed for action normalization; we look for one inside the
# HF repo, falling back to the worktree if not present.
if uv run hf download "${HF_MODEL_REPO}" so101_combined_stats.json \
    --revision "${HF_REVISION}" --local-dir "${WORK_DIR}" >/dev/null 2>&1; then
    STATS_FILE="${WORK_DIR}/so101_combined_stats.json"
elif [ -f "${REPO_ROOT}/so101_combined_stats.json" ]; then
    STATS_FILE="${REPO_ROOT}/so101_combined_stats.json"
else
    echo "[infer] WARNING: no stats file found on HF or in worktree."
    echo "[infer] Inference will still run but action normalization will mismatch training."
    echo "[infer] Upload stats from your training box: hf upload ${HF_MODEL_REPO} <local-stats.json>"
    STATS_FILE=""
fi

# --- 2. Pull one LeRobot source so we have val frames ---
echo "[infer] ensuring ${SOURCE_REPO_ID} is in the HF cache..."
uv run hf download --repo-type dataset "${SOURCE_REPO_ID}" >/dev/null

# --- 3. Patch config to use local paths + single source ---
echo "[infer] patching config for local inference..."
uv run python - <<PY
import yaml
from pathlib import Path

cfg_path = Path("${WORK_DIR}/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["dataset"]["loader"]["source_repo_ids"] = ["${SOURCE_REPO_ID}"]
cfg["dataset"]["loader"]["n_rollout"] = 12  # cap eps to keep load fast
cfg["dataset"]["loader"]["validation_size"] = max(8, ${NUM_SAMPLES} * 2)
if "${STATS_FILE}":
    cfg["dataset"]["loader"]["stats_path"] = "${STATS_FILE}"
cfg["wandb"]["enabled"] = False
cfg["logger"]["name"] = "tensorboard"
cfg_path.write_text(yaml.safe_dump(cfg))
print(f"wrote {cfg_path}")
PY

# --- 4. Run rollout demo ---
mkdir -p "${OUT_DIR}"
echo "[infer] running rollout (this takes ~30s/sample on A100)..."
uv run python scripts/rollout_demo.py \
    --config "${WORK_DIR}/config.yaml" \
    --checkpoint "${WORK_DIR}/${CKPT_PATH}" \
    --out_dir "${OUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --rollout_length "${ROLLOUT_LENGTH}" \
    --history_length "${HISTORY_LENGTH}" \
    --num_sampling_steps "${DDIM_STEPS}" \
    --fps 10

echo ""
echo "[infer] done. GIFs in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"*.gif 2>/dev/null
