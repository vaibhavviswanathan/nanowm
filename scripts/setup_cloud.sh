#!/usr/bin/env bash
# One-shot cloud bring-up for the SO-101 manipulation pretrain.
#
# Run on a fresh cloud box (Lambda, RunPod, etc.) after cloning this branch:
#   git clone -b manipulation-wm <your-fork> nanowm
#   cd nanowm
#   bash scripts/setup_cloud.sh
#
# Idempotent — re-running is safe. Skips work that's already done.
#
# Prereqs the cloud image must already provide:
#   - NVIDIA driver supporting CUDA 12.6+ (so torch 2.7+cu126 wheels load)
#   - ~/.cache or RESULTS_DIR / DATASET_DIR-writable filesystem with ~50GB free

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 1. uv (Python pkg manager) -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- 2. Upstream nano-world-model clone -----------------------------------
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"
if [ ! -e "${UPSTREAM_DIR}" ]; then
    echo "[setup] cloning nano-world-model at $(cat "${REPO_ROOT}/UPSTREAM_PIN")..."
    git clone https://github.com/simchowitzlabpublic/nano-world-model.git "${UPSTREAM_DIR}"
    git -C "${UPSTREAM_DIR}" checkout "$(cat "${REPO_ROOT}/UPSTREAM_PIN")"
fi

# --- 3. Apply patches + symlink scaffold ----------------------------------
echo "[setup] applying patches + symlinking scaffold..."
bash "${REPO_ROOT}/scripts/apply_upstream_patches.sh"

# --- 4. Python deps -------------------------------------------------------
echo "[setup] resolving Python deps (this takes 10-20 min on first run)..."
cd "${REPO_ROOT}"
uv sync --extra dev

# --- 5. Smoke import check ------------------------------------------------
echo "[setup] verifying lerobot/diffusers/torch wired correctly..."
uv run python -c "
import torch, lerobot, diffusers, pytorch_lightning
assert torch.cuda.is_available(), 'CUDA not available — check NVIDIA driver'
print(f'  torch: {torch.__version__} (cuda {torch.version.cuda})')
print(f'  lerobot: {lerobot.__version__}')
print(f'  diffusers: {diffusers.__version__}')
print(f'  lightning: {pytorch_lightning.__version__}')
print(f'  GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- 6. Tests -------------------------------------------------------------
echo "[setup] running test suite..."
uv run pytest -q

echo ""
echo "[setup] done."
echo "Next: bash scripts/download_so101_full.sh    # ~25 GB, lazy via HF cache"
echo "Then: bash scripts/pretrain_so101_h100.sh    # or pretrain_so101_a100.sh"
