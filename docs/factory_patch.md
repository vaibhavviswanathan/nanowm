# Upstream factory patch — superseded by `patches/01-upstream-integration.patch`

This file used to speculate about the upstream factory dispatch pattern. As of
Phase 0 we've read the real upstream code (pinned at the SHA in `UPSTREAM_PIN`)
and committed a single patch file that captures all required upstream edits.

To apply: run `bash scripts/apply_upstream_patches.sh` after cloning upstream.

The patch:
- Lazy-imports `lerobot` so the `diffusers==0.24.0` pin holds without installing it.
- Adds a `tartandrive` branch to `create_data_source` that forwards
  `latents_path` and `use_cached_latents` to `TartanDriveDataSource`.
- Extends `world_model_dataset.datasource_params` (two sites) with
  `latents_path`/`use_cached_latents` so Hydra-side YAML kwargs reach the
  DataSource.

The script also symlinks our tracked scaffold dirs into the upstream tree:

- `nano-world-model/src/wm_datasets/data_source/offroad` ->
  `src/wm_datasets/data_source/offroad`
- `nano-world-model/src/configs/dataset/offroad` ->
  `src/configs/dataset/offroad`

That way edits land in tracked files and propagate to the runtime tree on save.

See `docs/phase0_findings.md` for the full API reading that motivated this
patch.
