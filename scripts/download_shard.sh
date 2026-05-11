#!/usr/bin/env bash
# Stream-extract N rosbags from a TartanDrive 1.0 shard without ever holding
# the full ~69-127 GB tarball on disk.
#
# How it works:
#   curl streams the .tar.gz from CMU's Swift bucket; tar extracts on the fly
#   into a staging dir; once we have N bags on disk we terminate the pipeline.
#   The remaining tarball bytes never touch the drive.
#
# Usage:
#   $DATASET_DIR=~/data/nanowm bash scripts/download_shard.sh [SHARD] [N_BAGS]
# Defaults:
#   SHARD   = 20210828_heightmaps_1.tar.gz  (the smallest at ~69 GB)
#   N_BAGS  = 10
# Outputs:
#   $DATASET_DIR/raw_bags/<SHARD_STEM>/*.bag
# Re-run safety:
#   If $DATASET_DIR/raw_bags/<SHARD_STEM> already exists with >= N_BAGS .bag
#   files, exits without re-downloading.

set -euo pipefail

SHARD="${1:-20210828_heightmaps_1.tar.gz}"
N_BAGS="${2:-10}"

: "${DATASET_DIR:?DATASET_DIR not set (e.g. export DATASET_DIR=~/data/nanowm)}"

BUCKET_URL="https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive"
SHARD_URL="${BUCKET_URL}/${SHARD}"
SHARD_STEM="${SHARD%.tar.gz}"
OUT_DIR="${DATASET_DIR}/raw_bags/${SHARD_STEM}"

mkdir -p "${OUT_DIR}"

existing_bags=$(find "${OUT_DIR}" -maxdepth 2 -name '*.bag' 2>/dev/null | wc -l)
if [ "${existing_bags}" -ge "${N_BAGS}" ]; then
    echo "Already have ${existing_bags} bags in ${OUT_DIR} (>= ${N_BAGS}). Skipping download."
    find "${OUT_DIR}" -maxdepth 2 -name '*.bag' | sort | head -"${N_BAGS}"
    exit 0
fi

echo "Streaming ${SHARD_URL}"
echo "  -> extracting up to ${N_BAGS} bags into ${OUT_DIR}"

# Pipe-stop trick: curl pumps the tarball; tar extracts entries as it goes;
# a watcher kills the pipeline as soon as ${N_BAGS} .bag files exist on disk.
# `set -m` would normally be required for job control, but here we keep PIDs.
WATCHER_DONE=$(mktemp -u)
(
    while sleep 2; do
        bags=$(find "${OUT_DIR}" -maxdepth 2 -name '*.bag' 2>/dev/null | wc -l)
        if [ "${bags}" -ge "${N_BAGS}" ]; then
            touch "${WATCHER_DONE}"
            # Kill the curl/tar pipeline by closing the pipe (the tar process
            # will exit cleanly with SIGPIPE handling).
            pkill -P $$ -f 'curl -fsSL' 2>/dev/null || true
            return 0
        fi
    done
) &
WATCHER_PID=$!

# tar exits non-zero when the pipe closes early; allow that.
set +e
curl -fsSL "${SHARD_URL}" | tar -xz -C "${OUT_DIR}" --no-same-owner 2>/dev/null
set -e

# Stop the watcher in case curl finished first
kill "${WATCHER_PID}" 2>/dev/null || true
wait "${WATCHER_PID}" 2>/dev/null || true
rm -f "${WATCHER_DONE}"

final_bags=$(find "${OUT_DIR}" -maxdepth 2 -name '*.bag' | wc -l)
echo
echo "Got ${final_bags} bags in ${OUT_DIR}:"
find "${OUT_DIR}" -maxdepth 2 -name '*.bag' | sort | head -"${N_BAGS}"
du -sh "${OUT_DIR}"
