#!/usr/bin/env bash
# Full pretrain on an A100/40GB. Same effective batch as the H100 config
# but halved per-step batch + grad_accum=2 to fit in 40GB. ETA ~36-48 hr.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BATCH_SIZE=16 GRAD_ACCUM=2 \
    bash "${REPO_ROOT}/scripts/pretrain_so101_h100.sh" "$@"
