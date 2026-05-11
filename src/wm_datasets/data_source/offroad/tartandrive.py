"""TartanDrive DataSource for nano-world-model.

Loads the layout produced by `scripts/preprocess_tartandrive.py`:

    data_path/
      traj_0001/
        frames.npy   # uint8 [T, H, W, 3]
        actions.npy  # float32 [T, 2]   -> (throttle, steer)
        meta.json    # {length, source_traj_id, fps}
      traj_0002/
      ...

Pass `data_path = $DATASET_DIR/tartandrive/train` for the train split, and
`data_path = $DATASET_DIR/tartandrive/val` for the val split — same pattern as
PushT (`data_path_train`/`data_path_val` in the dataset YAML).

Optional cached-latent mode: if `use_cached_latents=True`, frames are replaced
by VAE latents loaded from a mirrored layout under `latents_path`. The
DataSource appends the split basename (`data_path.name`) to `latents_path`, so
a single shared `latents_path` works for both train and val splits:

    latents_path/<data_path.name>/
      traj_0001/
        latents.npy   # float32 [T, C', H', W']  (already scaling-factor-scaled)
      ...

Latents must be pre-computed with `scripts/precompute_latents.py` using the
same VAE the training pipeline loads (see `docs/phase0_findings.md`).
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..base import DataSource, TrajectoryData


class TartanDriveDataSource(DataSource):
    """DataSource for preprocessed TartanDrive trajectories.

    Args:
        data_path: Directory containing `traj_XXXX/` subdirs. For TartanDrive
            this is one of `$DATASET_DIR/tartandrive/{train,val}`.
        n_rollout: Limit the number of loaded trajectories (None = all). Used
            by smoke tests; also forwarded by the factory when `n_rollout` is
            in the YAML.
        latents_path: Root containing `train/` and `val/` subdirs each with
            `traj_XXXX/latents.npy`. The DataSource resolves the split-specific
            dir as `latents_path/<data_path.name>`. Required iff
            `use_cached_latents=True`.
        use_cached_latents: When True, `load_visual_frames` returns cached VAE
            latents instead of pixel frames.
    """

    ACTION_DIM = 2  # throttle, steer

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        latents_path: Optional[str] = None,
        use_cached_latents: bool = False,
    ):
        self.data_path = Path(data_path)
        self.use_cached_latents = use_cached_latents
        # latents_path is the root that contains <split>/ subdirs. The
        # split-specific dir is derived from data_path.name (e.g., "train").
        self.latents_root = Path(latents_path) if latents_path is not None else None
        self.latents_split_dir = (
            self.latents_root / self.data_path.name if self.latents_root else None
        )

        if not self.data_path.exists():
            raise FileNotFoundError(f"data_path does not exist: {self.data_path}")

        if self.use_cached_latents:
            if self.latents_split_dir is None:
                raise ValueError(
                    "use_cached_latents=True but latents_path is not set. "
                    "Pass latents_path to the factory or via the dataset YAML."
                )
            if not self.latents_split_dir.exists():
                raise FileNotFoundError(
                    f"latents split dir does not exist: {self.latents_split_dir}. "
                    "Did you run scripts/precompute_latents.py?"
                )

        self._traj_dirs = sorted(p for p in self.data_path.iterdir() if p.is_dir())
        if not self._traj_dirs:
            raise FileNotFoundError(
                f"No trajectory directories under {self.data_path}. "
                "Did you run scripts/preprocess_tartandrive.py?"
            )

        if n_rollout is not None:
            self._traj_dirs = self._traj_dirs[:n_rollout]

        # Cache lengths and actions in memory. Actions are tiny (T x 2 float32).
        # Frames stay on disk and are mmap-loaded on demand.
        self._lengths: List[int] = []
        self._actions: List[torch.Tensor] = []
        for traj_dir in self._traj_dirs:
            with open(traj_dir / "meta.json") as f:
                meta = json.load(f)
            length = int(meta["length"])
            actions_np = np.load(traj_dir / "actions.npy").astype(np.float32)
            if actions_np.ndim != 2 or actions_np.shape[1] != self.ACTION_DIM:
                raise ValueError(
                    f"{traj_dir}/actions.npy must be [T, {self.ACTION_DIM}], "
                    f"got {actions_np.shape}"
                )
            if len(actions_np) != length:
                raise ValueError(
                    f"meta.length={length} but actions has {len(actions_np)} rows "
                    f"in {traj_dir}"
                )
            self._lengths.append(length)
            self._actions.append(torch.from_numpy(actions_np))

        self.num_trajectories = len(self._traj_dirs)
        cached_str = (
            f" cached_latents=True (from {self.latents_split_dir})"
            if use_cached_latents
            else ""
        )
        print(
            f"[TartanDrive] {self.num_trajectories} trajectories, "
            f"{sum(self._lengths)} total frames{cached_str}"
        )

    # --- DataSource API -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def state_dim(self) -> int:
        return 0  # pure-vision; no proprioceptive state

    def get_num_trajectories(self) -> int:
        return self.num_trajectories

    def get_seq_length(self, index: int) -> int:
        self._check_index(index)
        return self._lengths[index]

    def load_trajectory(self, index: int) -> TrajectoryData:
        self._check_index(index)
        seq_length = self._lengths[index]
        actions = self._actions[index]
        states = torch.zeros(seq_length, 0, dtype=torch.float32)
        return TrajectoryData(
            states=states,
            actions=actions,
            seq_length=seq_length,
            meta={"episode_id": self._traj_dirs[index].name},
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> torch.Tensor:
        """Return frames (or cached latents) for `[start, end)` with stride `step`.

        Pixel mode: float32 `[N, 3, H, W]` in `[0, 1]`. WorldModelDataset resizes
        and rescales to `[-1, 1]` downstream.

        Latent mode: float32 `[N, C', H', W']` — already scaled by the VAE's
        scaling_factor at precompute time, ready to feed the diffusion
        transformer directly.
        """
        self._check_index(index)
        traj_dir = self._traj_dirs[index]

        if self.use_cached_latents:
            latents_file = self.latents_split_dir / traj_dir.name / "latents.npy"
            arr = np.load(latents_file, mmap_mode="r")
            sliced = np.array(arr[start:end:step], dtype=np.float32, copy=True)
            return torch.from_numpy(sliced)

        frames_file = traj_dir / "frames.npy"
        arr = np.load(frames_file, mmap_mode="r")  # uint8 [T, H, W, 3]
        # `.copy()` detaches from the mmap so the resulting tensor is writable.
        sliced = np.array(arr[start:end:step], dtype=np.uint8, copy=True)
        # uint8 [N, H, W, 3] -> float32 [N, 3, H, W] in [0, 1]
        frames = (
            torch.from_numpy(sliced).permute(0, 3, 1, 2).contiguous().float() / 255.0
        )
        return frames

    # --- internal -------------------------------------------------------------

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= self.num_trajectories:
            raise IndexError(
                f"index {index} out of range [0, {self.num_trajectories})"
            )
