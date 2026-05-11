# Phase 0 — Upstream API findings

Upstream pinned at `UPSTREAM_PIN` = `b60e9905e01a9fa249a54c9148addd971b1fc9bb`.

## DataSource contract

From `src/wm_datasets/data_source/base.py`:

- `DataSource` is an ABC with abstract methods:
  - `load_trajectory(index) -> TrajectoryData`
  - `load_visual_frames(index, start, end, step=1) -> torch.Tensor`
  - `get_num_trajectories() -> int`
  - Properties: `action_dim`, `state_dim` (both required)
- `TrajectoryData` is a dataclass with `states: Tensor`, `actions: Tensor`, `seq_length: int`, `meta: Dict`.
- `__post_init__` asserts `states.shape[0] == actions.shape[0]` — both must have the same outer length.
- CSGO sets `states = torch.zeros(T, 0, float32)` and `state_dim = 0` for pure-vision datasets. Same pattern fits TartanDrive.

## load_visual_frames contract

From CSGO reference (`src/wm_datasets/data_source/game/csgo_data_source.py`):

- Returns `torch.Tensor [T, C, H, W]`, **float32 in [0, 1]**.
- WorldModelDataset later resizes to `image_size` and rescales to `[-1, 1]` if `normalize_pixel=True`.
- **Scaffold's `tartandrive.py` returns `np.ndarray uint8 [T, H, W, 3]` — this is wrong and must be fixed.**

## Factory dispatch

From `src/wm_datasets/data_source/factory.py`:

- Signature: `create_data_source(dataset_name, data_path, n_rollout=None, **kwargs)`.
- Uses `if/return` chains (not elif), filters kwargs per branch.
- We need to add a branch for `tartandrive` with kwargs `{latents_path, use_cached_latents}`.

## WorldModelDataset consumer

From `src/wm_datasets/world_model_dataset.py`:

- Slices the DataSource: `traj.actions[start:end]`, `traj.states[start:end:step]`, `data_source.load_visual_frames(index, start, end, step)`.
- Filters DataSource kwargs via the `datasource_params` set inside `create_world_model_dataset`. To add tartandrive kwargs we need to extend that set (patch).
- Stats caching exists via `stats_cache.py`; can opt in by exposing data-source-level stats.

## VAE

From `src/experiments/train_experiment.py` and `src/utils/vae_ops.py`:

- Default `VAE_MODEL_PATH = stabilityai/sd-vae-ft-mse`, `scaling_factor = 0.18215`.
- Encode: `vae.encode(x).latent_dist.sample().mul_(scaling_factor)` (note: sample, not mode, in training).
- Decode: `vae.decode(z / scaling_factor).sample`.
- **Upstream loads with `subfolder="vae"`, which fails against the default `stabilityai/sd-vae-ft-mse` (no `vae/` subdir).** Needs a try/fallback patch before Phase 3.
- Precision policy is configurable (`vae_precision = fp32|bf16|match_trainer`); default fp32.

## Rollout API

From `src/sample/rollout.py`:

- Rollout is a standalone script using `dfot_sample(...)` from `src/diffusion/df_sample.py`.
- **No `model.rollout()` method exists.** Scaffold's `rollout_demo.py` must mimic the `rollout.py` flow (load model+VAE+diffusion, call `dfot_sample`).

## Upstream patches already applied (in `patches/`)

1. `01-lazy-lerobot.patch` — Lazy-import lerobot in factory.py to allow `diffusers==0.24.0` + skipping lerobot install. The eager imports were in three places: `wm_datasets/__init__.py`, `wm_datasets/data_source/__init__.py`, `wm_datasets/data_source/factory.py`.

## Upstream patches still pending (will be added in later phases)

- `02-vae-subfolder-fallback.patch` — make `AutoencoderKL.from_pretrained(..., subfolder="vae")` fall back to no-subfolder for standalone VAE repos like `sd-vae-ft-mse`. Affects three files: `train_experiment.py`, `sample/rollout.py`, `sample/sample_dfot.py`.
- `03-tartandrive-factory.patch` — add tartandrive branch in factory.py and extend `datasource_params` in `world_model_dataset.py`.

## Scaffold rewrites pending (Phase 2)

- `src/wm_datasets/data_source/offroad/tartandrive.py`:
  - Return `TrajectoryData` (not dict)
  - Tensors, not numpy
  - Frames in `[0, 1]` float `[T,C,H,W]`
  - Implement `state_dim` property
  - Accept `n_rollout` kwarg
  - Drop `split` kwarg — factory dispatches `data_path` (split-aware path) directly
- `scripts/precompute_latents.py`:
  - Load VAE without subfolder (or use the patched upstream loader)
  - Use `.mode()` for deterministic latents
  - Multiply by `vae.config.scaling_factor` to match training
- `scripts/rollout_demo.py`:
  - Replace fake `model.rollout()` with `dfot_sample(...)` pattern from `src/sample/rollout.py`
- `src/configs/dataset/offroad/tartandrive.yaml`:
  - Use `data_path_train`/`data_path_val` (mirrors `pusht.yaml`)
  - Move `latents_path`/`use_cached_latents` into kwargs forwarded via `datasource_params`
