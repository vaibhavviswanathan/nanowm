#!/usr/bin/env bash
# Pre-flight HuggingFace Hub auth + write-access check.
#
# Run BEFORE any long training so broken auth is caught in minute 1, not
# hour 5. Verifies:
#   - HF_TOKEN is set (or `hf auth whoami` is logged in)
#   - HF_MODEL_REPO is set (target repo to push checkpoints to)
#   - The repo exists (creates it private if missing)
#   - We have write access (uploads a tiny test file and deletes it)
#
# Usage:
#   export HF_TOKEN=hf_...                          # or use `hf auth login`
#   export HF_MODEL_REPO=<your-user>/so101-wm       # required
#   bash scripts/hf_setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${HF_MODEL_REPO:-}" ]; then
    echo "ERROR: HF_MODEL_REPO not set. Example: export HF_MODEL_REPO=vaibhavviswanathan/so101-wm" >&2
    exit 1
fi

cd "${REPO_ROOT}"

# --- 1. Auth check ----------------------------------------------------------
echo "[hf] checking auth..."
WHOAMI=$(uv run hf auth whoami 2>&1 || true)
if echo "${WHOAMI}" | grep -qiE "not logged in|error"; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: not logged in. Set HF_TOKEN or run 'uv run hf auth login'." >&2
        echo "  Got: ${WHOAMI}" >&2
        exit 1
    fi
    echo "[hf] using HF_TOKEN from env"
fi
echo "[hf] whoami: ${WHOAMI}"

# --- 2. Repo exists / create (idempotent via --exist-ok) -------------------
echo "[hf] ensuring repo ${HF_MODEL_REPO} exists..."
uv run hf repos create "${HF_MODEL_REPO}" --type model --private --exist-ok >/dev/null

# --- 3. Write-access smoke test --------------------------------------------
TEST_DIR=$(mktemp -d)
trap "rm -rf ${TEST_DIR}" EXIT
TEST_FILE="${TEST_DIR}/.hf_write_test"
echo "smoke $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${TEST_FILE}"

echo "[hf] uploading test file..."
uv run hf upload "${HF_MODEL_REPO}" "${TEST_FILE}" .hf_write_test \
    --type model --commit-message "auth smoke test" >/dev/null

echo "[hf] deleting test file..."
uv run hf repos delete-files "${HF_MODEL_REPO}" .hf_write_test \
    --type model >/dev/null 2>&1 || \
    echo "[hf] (couldn't auto-delete test file; ignore — write access still verified)"

echo "[hf] OK. ${HF_MODEL_REPO} ready for checkpoint syncs."
