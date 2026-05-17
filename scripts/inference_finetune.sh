#!/usr/bin/env bash
# Generate 10-second rollouts from a SO-101 finetune checkpoint.
#
# Thin wrapper over inference_from_hf.sh — just sets the finetune-friendly
# defaults (longer rollouts, source from the finetune mix).
#
# Required env:
#   HF_MODEL_REPO=vaiviswanathan/so101-wm-ft  (the finetune destination repo)
#   HF_TOKEN=hf_<read-token>                  (optional for public repos)
#
# Override surface:
#   ROLLOUT_LENGTH (default 150 = 10s @ 15 fps)
#   NUM_SAMPLES    (default 6)
#   SOURCE_REPO_ID (default: ofcourseistillloveyou/so101_recording_strawberry_...)
#   GIF_FPS        (default 15 — display at training fps)
#
# Notes on rollout length:
#   - Training horizon = 16 frames (~1s). 150 frames is ~10× beyond, so
#     expect some drift. The 75-frame (5s) variant is the "clean" demo.
#   - At ~0.6s/frame on A100, 150-frame rollout ≈ 1.5 min/sample.
#     6 samples ≈ 10 min total.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-150}"
export NUM_SAMPLES="${NUM_SAMPLES:-6}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
export SOURCE_REPO_ID="${SOURCE_REPO_ID:-ofcourseistillloveyou/so101_recording_strawberry_20260516_161110}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/demo_finetune}"

# `inference_from_hf.sh` hardcodes `--fps 10` for the GIF playback rate. For
# finetune the model is trained at 15 fps so real-time playback is 15 fps;
# we can't override that without editing the parent script. Acceptable: 10
# fps GIFs are slow-motion but motion is still legible.

echo "[infer-ft] ROLLOUT_LENGTH=${ROLLOUT_LENGTH} (~$((ROLLOUT_LENGTH / 15))s @ 15 fps)"
echo "[infer-ft] SOURCE_REPO_ID=${SOURCE_REPO_ID}"
echo "[infer-ft] OUT_DIR=${OUT_DIR}"

bash "${REPO_ROOT}/scripts/inference_from_hf.sh"
