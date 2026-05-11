# nanowm — TartanDrive × nano-world-model POC

## Layout
- `nano-world-model/` — upstream checkout, pinned via `UPSTREAM_PIN` (not tracked here).
- `scripts/` — preprocess, latent precompute, smoke, train, rollout.
- `src/` — DataSource + configs designed to be copied INTO `nano-world-model/src/`.
- `tests/` — pytest suite for the data layer (preprocess + DataSource).
- `docs/` — design notes (factory patch, dataset notes).

## Conventions
- Python env via `uv`. Re-create: `uv sync --extra dev`. Activate: `source .venv/bin/activate`.
- Upstream commit pinned in `UPSTREAM_PIN`. To re-clone: `git clone https://github.com/simchowitzlabpublic/nano-world-model.git && git -C nano-world-model checkout $(cat UPSTREAM_PIN)`.
- Data path: `$DATASET_DIR` (default `~/data/nanowm`). Results: `$RESULTS_DIR` (default `~/results/nanowm`). Both gitignored.
- Tests: `uv run pytest`. Run before any long-running script.

## Don't do
- Don't commit `*.npy`, `data/`, `results/`, or `nano-world-model/`.
- Don't run training before `uv run pytest` is green.
- Don't change preprocessing without rebuilding latents — they're keyed by source-frame SHA in `latents.meta.json`.
