# nanowm — TartanDrive × nano-world-model POC

## Layout
- `nano-world-model/` — upstream checkout, pinned via `UPSTREAM_PIN` (not tracked here).
- `patches/` — patches applied to upstream after clone (via `scripts/apply_upstream_patches.sh`).
- `scripts/` — download, convert, preprocess, latents, smoke, train, rollout.
- `src/` — tracked DataSource + configs; symlinked into upstream's tree at apply time.
- `tests/` — 46 fast pytests for data layer (DataSource, factory, integration, preprocess).
- `docs/` — design notes (`phase0_findings.md`, `dataset_notes.md`).

## Conventions
- Python env via `uv`. Re-create: `uv sync --extra dev`. Run via `uv run <cmd>`.
- Upstream commit pinned in `UPSTREAM_PIN`. To re-clone:
  ```
  git clone https://github.com/simchowitzlabpublic/nano-world-model.git
  git -C nano-world-model checkout $(cat UPSTREAM_PIN)
  bash scripts/apply_upstream_patches.sh
  ```
- Data: `$DATASET_DIR` (default `~/data/nanowm`); results: `$RESULTS_DIR` (default `~/results/nanowm`). Both gitignored.
- Tests: `uv run pytest`. Run before any long-running script.

## Data pipeline (Phase 1)
TartanDrive 1.0 is 2 TB of rosbags. Incremental flow:
1. `bash scripts/download_shard.sh <SHARD> <N_BAGS>` — stream-extract N bags, never holds the full tarball.
2. `uv run python scripts/convert_bags.py --bags ... --out_dir <staging>` — rosbags -> scaffold format at 256² / 10 Hz. Pure-Python via `rosbags`; uses standard ROS messages only (no ROS install needed).
3. `uv run python scripts/preprocess_tartandrive.py --staging_dir <staging> --out_dir <out>` — hash-deterministic train/val split + stats.json. Re-run-safe; appends new bags.

Inspect bag topics: `uv run python scripts/convert_bags.py --inspect <bag>`

## Don't do
- Don't commit `*.npy`, `data/`, `results/`, or `nano-world-model/`.
- Don't run training before `uv run pytest` is green.
- Don't change preprocessing without rebuilding latents — they're keyed by source-frames SHA in `latents.meta.json`.
- Don't install `lerobot` to "fix" the rt1 import — upstream is patched to lazy-load it. Installing it would force a newer `diffusers` and break the VAE training path.
