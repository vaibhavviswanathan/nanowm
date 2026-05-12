#!/usr/bin/env bash
# Disk-conscious incremental shard ingestion:
# for each shard, stream-pull bags -> convert to staging -> delete raw bags.
# Each shard's footprint drops from ~30 GB raw to ~250 MB preprocessed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_DIR:?DATASET_DIR not set}"

SHARDS=("$@")
if [ ${#SHARDS[@]} -eq 0 ]; then
    echo "usage: $0 <shard1.tar.gz> [<shard2> ...]"
    exit 2
fi

STAGING="${DATASET_DIR}/tartandrive_staging"

for shard in "${SHARDS[@]}"; do
    stem="${shard%.tar.gz}"
    bags_dir="${DATASET_DIR}/raw_bags/${stem}"

    echo
    echo "=== $(date) pull ${shard}"
    bash scripts/download_shard.sh "${shard}" 10

    echo "=== $(date) convert ${stem}"
    uv run python scripts/convert_bags.py \
        --bags "${bags_dir}/${stem}"/*.bag \
        --out_dir "${STAGING}" \
        --resolution 256 --decimate 3 || true

    echo "=== $(date) remove raw bags ${stem}"
    rm -rf "${bags_dir}"
done

echo "=== $(date) all shards processed; staging at ${STAGING}"
du -sh "${STAGING}"
ls "${STAGING}" | wc -l
