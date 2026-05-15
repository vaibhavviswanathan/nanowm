#!/usr/bin/env bash
# Download all four SO-101 pretrain sources into the HF cache.
#
# Total: ~25 GB on disk (mostly mp4 video files, lazily resolved on first
# decode). Safe to interrupt and resume — HF cache is content-addressed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCES=(
    "lerobot/svla_so100_pickplace"
    "lerobot/svla_so100_stacking"
    "lerobot/svla_so100_sorting"
    "lerobot/svla_so101_pickplace"
)

for repo_id in "${SOURCES[@]}"; do
    echo "[dl] $repo_id"
    uv run --project "${REPO_ROOT}" hf download --repo-type dataset "${repo_id}"
done

echo ""
echo "[dl] All four sources cached. Next:"
echo "  bash scripts/compute_so101_stats_full.sh   # ~5 min"
echo "  bash scripts/pretrain_so101_h100.sh"
