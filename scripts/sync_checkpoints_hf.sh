#!/usr/bin/env bash
# Background checkpoint syncer: watches the latest training run's
# checkpoints/ directory and uploads new .ckpt files to HF Hub.
#
# Run alongside training:
#   nohup bash scripts/sync_checkpoints_hf.sh > sync.out 2>&1 &
#
# What it uploads:
#   - latest/latest-*.ckpt    overwritten every 1k steps (mid-training pull)
#   - across_timesteps/*.ckpt kept forever (every 10k steps)
#   - config.yaml + tb/       once at startup
#
# Polling interval: 5 min by default (HF_SYNC_INTERVAL=N to override).
# Skipped uploads: files whose mtime changed within the last 30 s (likely
# still being written by Lightning's atomic-save shuffle).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${HF_MODEL_REPO:?HF_MODEL_REPO not set (e.g. export HF_MODEL_REPO=user/so101-wm)}"
INTERVAL="${HF_SYNC_INTERVAL:-300}"

cd "${REPO_ROOT}"

# Track uploaded file -> mtime so we re-upload on overwrite.
declare -A UPLOADED

# --- Find latest run dir (created by Hydra at training start) ---
echo "[sync] waiting for a run dir under ${RESULTS_DIR}..."
RUN_DIR=""
for _ in $(seq 1 120); do  # up to ~10 min
    candidate=$(ls -1dt "${RESULTS_DIR}"/*so101* 2>/dev/null | head -1 || true)
    if [ -n "${candidate}" ] && [ -d "${candidate}/checkpoints" ]; then
        RUN_DIR="${candidate}"
        break
    fi
    sleep 5
done
if [ -z "${RUN_DIR}" ]; then
    echo "[sync] ERROR: no run dir with checkpoints/ found after 10 min." >&2
    exit 1
fi
echo "[sync] watching ${RUN_DIR}"

# --- Upload startup artifacts once ---
if [ -f "${RUN_DIR}/config.yaml" ]; then
    echo "[sync] uploading config.yaml"
    uv run hf upload "${HF_MODEL_REPO}" "${RUN_DIR}/config.yaml" "config.yaml" \
        --repo-type model --commit-message "config snapshot" || \
        echo "[sync] config.yaml upload failed (will retry next pass)"
fi

upload_if_changed() {
    local localfile="$1"
    local remote_path="$2"
    local mtime
    mtime=$(stat -c %Y "${localfile}" 2>/dev/null || echo 0)

    # Skip if just-written (likely Lightning's atomic-save shuffle)
    local age=$(( $(date +%s) - mtime ))
    if [ "${age}" -lt 30 ]; then
        return 0
    fi

    if [ "${UPLOADED[$localfile]:-0}" = "${mtime}" ]; then
        return 0  # already uploaded this version
    fi

    local size_mb=$(du -m "${localfile}" 2>/dev/null | cut -f1)
    echo "[sync] uploading ${remote_path} (${size_mb} MB, mtime=${mtime})"
    if uv run hf upload "${HF_MODEL_REPO}" "${localfile}" "${remote_path}" \
        --repo-type model --commit-message "ckpt sync $(basename "${localfile}")" 2>&1 | tail -3; then
        UPLOADED[$localfile]="${mtime}"
    else
        echo "[sync] upload failed; will retry next pass"
    fi
}

echo "[sync] entering poll loop (interval=${INTERVAL}s)"
while true; do
    # 1. latest/latest-*.ckpt — overwrites; we upload to fixed remote path.
    for f in "${RUN_DIR}"/checkpoints/latest/latest-*.ckpt; do
        [ -e "${f}" ] || continue
        upload_if_changed "${f}" "checkpoints/latest.ckpt"
    done

    # 2. across_timesteps/*.ckpt — kept forever; preserve filenames.
    for f in "${RUN_DIR}"/checkpoints/across_timesteps/*.ckpt; do
        [ -e "${f}" ] || continue
        upload_if_changed "${f}" "checkpoints/across_timesteps/$(basename "${f}")"
    done

    sleep "${INTERVAL}"
done
