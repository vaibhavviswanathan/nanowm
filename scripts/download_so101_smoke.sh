#!/usr/bin/env bash
# Pull just the smallest SO-101 source for the smoke pipeline.
# svla_so101_pickplace: 50 episodes / 11,939 frames / ~few GB on disk.
#
# Full pretrain mix is downloaded separately via download_so101_full.sh
# (which iterates all four).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_ID="${REPO_ID:-lerobot/svla_so101_pickplace}"

# hf cli ships with huggingface_hub which lerobot pulls in.
uv run --project "${REPO_ROOT}" huggingface-cli download \
    --repo-type dataset \
    "${REPO_ID}" \
    --quiet

echo "Done. Cache location:"
uv run --project "${REPO_ROOT}" python -c "
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()
for r in info.repos:
    if '${REPO_ID}' in r.repo_id:
        print(f'  {r.repo_id}: {r.size_on_disk_str} at {r.repo_path}')
"
