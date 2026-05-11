#!/usr/bin/env bash
# Re-apply upstream patches after a fresh clone of nano-world-model.
#
# Usage:
#   git clone https://github.com/simchowitzlabpublic/nano-world-model.git
#   git -C nano-world-model checkout $(cat UPSTREAM_PIN)
#   bash scripts/apply_upstream_patches.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/nano-world-model"

if [ ! -d "${UPSTREAM_DIR}" ]; then
    echo "ERROR: ${UPSTREAM_DIR} does not exist. Clone it first:"
    echo "  git clone https://github.com/simchowitzlabpublic/nano-world-model.git ${UPSTREAM_DIR}"
    echo "  git -C ${UPSTREAM_DIR} checkout \$(cat ${REPO_ROOT}/UPSTREAM_PIN)"
    exit 1
fi

cd "${UPSTREAM_DIR}"
for patch in "${REPO_ROOT}"/patches/*.patch; do
    echo "Applying $(basename "${patch}")..."
    git apply --check "${patch}" 2>/dev/null && git apply "${patch}" || {
        echo "  (already applied or conflict — skipping)"
    }
done

# Symlink our tracked scaffold dirs into the upstream tree. These are gitignored
# in upstream (which is itself gitignored), so they don't fight upstream's tree.
echo "Linking scaffold dirs into upstream..."
rm -rf "${UPSTREAM_DIR}/src/wm_datasets/data_source/offroad"
ln -snf ../../../../src/wm_datasets/data_source/offroad \
    "${UPSTREAM_DIR}/src/wm_datasets/data_source/offroad"

rm -rf "${UPSTREAM_DIR}/src/configs/dataset/offroad"
ln -snf ../../../../src/configs/dataset/offroad \
    "${UPSTREAM_DIR}/src/configs/dataset/offroad"

echo "Done. Upstream is at $(git rev-parse HEAD), $(ls "${REPO_ROOT}/patches" | wc -l) patches applied, scaffold symlinked."
