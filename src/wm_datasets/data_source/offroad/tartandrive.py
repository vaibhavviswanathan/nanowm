"""TartanDrive DataSource for nano-world-model.

Loads the layout produced by `preprocess_tartandrive.py`:

    data_root/
      train/
        traj_0001/
          frames.npy   # uint8 [T, H, W, 3]
          actions.npy  # float32 [T, 2]
          meta.json
        ...
      val/
        ...

Optionally loads pre-encoded VAE latents instead of raw frames if
`use_cached_latents=True`. Latents are expected at:

    latents_root/
      <split>/
        traj_0001/
          latents.npy  # float32 [T, C, H', W']

# TODO: verify the DataSource base class signature against your actual
# nano-world-model checkout. The CSGO data source under
# `src/wm_datasets/data_source/game/` is the closest reference — match its
# method signatures exactly.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

# TODO: adjust this import to match your repo. The base class lives somewhere
# under src/wm_datasets/data_source/. Check `csgo.py` for the canonical import.
from src.wm_datasets.data_source.base import DataSource  # noqa: E402


class TartanDriveDataSource(DataSource):
    """DataSource for preprocessed TartanDrive trajectories.

    Args:
        data_path: root containing train/ and val/ subdirs.
        split: "train" or "val".
        latents_path: optional root with pre-encoded VAE latents (mirrors
            the train/val/<traj> layout). If provided, `load_visual_frames`
            returns latents instead of pixel frames.
        use_cached_latents: switch on cached-latent mode.
    """

    ACTION_DIM = 2  # throttle, steer

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        latents_path: Optional[str] = None,
        use_cached_latents: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_path = Path(data_path)
        self.split = split
        self.split_dir = self.data_path / split

        self.use_cached_latents = use_cached_latents
        self.latents_path = Path(latents_path) if latents_path else None
        if self.use_cached_latents:
            assert self.latents_path is not None, (
                "use_cached_latents=True but latents_path not provided"
            )
            self.latents_split_dir = self.latents_path / split

        self._traj_dirs = sorted(
            [p for p in self.split_dir.iterdir() if p.is_dir()]
        )
        if not self._traj_dirs:
            raise FileNotFoundError(
                f"No trajectory directories under {self.split_dir}. "
                "Did you run preprocess_tartandrive.py?"
            )

        # Cache lengths and actions in memory — small, and avoids re-reading
        # meta.json on every __getitem__. Frames stay on disk (loaded on demand).
        self._lengths: list = []
        self._actions: list = []
        for traj_dir in self._traj_dirs:
            with open(traj_dir / "meta.json") as f:
                meta = json.load(f)
            self._lengths.append(int(meta["length"]))
            self._actions.append(
                np.load(traj_dir / "actions.npy").astype(np.float32)
            )

        print(
            f"[TartanDrive/{split}] {len(self._traj_dirs)} trajectories, "
            f"{sum(self._lengths)} total frames"
            + (" (cached latents)" if use_cached_latents else "")
        )

    # --------- DataSource API ------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    def get_num_trajectories(self) -> int:
        return len(self._traj_dirs)

    def get_seq_length(self, index: int) -> int:
        return self._lengths[index]

    def load_trajectory(self, index: int) -> dict:
        """Load actions (and any other per-trajectory metadata) for index.

        Frames are NOT loaded here — they're loaded on demand by
        `load_visual_frames` to keep memory bounded.
        """
        return {
            "actions": self._actions[index],
            "length": self._lengths[index],
        }

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> np.ndarray:
        """Load frames (or cached latents) for [start, end) with step.

        Returns:
            If `use_cached_latents`: float32 [N, C, H', W'] latents.
            Else: uint8 [N, H, W, 3] frames.
        """
        traj_dir = self._traj_dirs[index]

        if self.use_cached_latents:
            latents_file = self.latents_split_dir / traj_dir.name / "latents.npy"
            # mmap so we don't load the whole trajectory into RAM
            arr = np.load(latents_file, mmap_mode="r")
            return np.asarray(arr[start:end:step], dtype=np.float32)

        frames_file = traj_dir / "frames.npy"
        arr = np.load(frames_file, mmap_mode="r")
        return np.asarray(arr[start:end:step])
