"""SO-101 multi-source LeRobot DataSource for nano-world-model.

Loads one or more LeRobot v2.1/v3.0 datasets that share the SO-100/SO-101
6-DoF arm embodiment, remaps each source's per-joint action keys onto a
canonical ordering, subsamples 30 fps -> target_fps via stride indexing, and
concatenates wrist + top camera views horizontally to fake multi-view input
on a model that only understands single-view square images.

Canonical action order (6-DoF):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

`ACTION_KEY_MAPS` lists the raw per-joint column names per source in canonical
order. Some datasets store a packed `action` Tensor of shape `[6]` instead of
per-joint columns; we detect that case and use it directly. Add your own SO-101
task dataset by registering its repo_id in `ACTION_KEY_MAPS` and
`CAMERA_KEY_MAPS`.

Camera layout: for every (wrist, top) pair we concat side-by-side along the
width dim, returning `[T, 3, H, 2W]`. Missing views are filled with zeros.
WorldModelDataset's bilinear resize to `image_size` (must be square — upstream
NanoWM hard-assumes a square patch grid via `grid_size = sqrt(num_patches)`)
then squishes width 2x per view. Pre-training and fine-tuning use the same
layout so checkpoints transfer.

Normalization stats: pass `stats_path=<combined_stats.json>` to load combined
per-joint action mean/std produced by `scripts/compute_so101_stats.py`.
WorldModelDataset will pick these up via `stats_cache.try_source_stats`
(reads `self.stats["action_mean"]` / `self.stats["action_std"]`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import DataSource, TrajectoryData


CANONICAL_JOINTS: Tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
CANONICAL_VIEWS: Tuple[str, ...] = ("wrist", "top")


# Per-source raw column names for the 6-DoF arm, in canonical order.
# Datasets that pack the action as a single 6-vector under the `action` key
# (more common) are detected at load time and don't need an entry here, but
# we keep the per-joint mapping for v2.1-style sources for completeness.
ACTION_KEY_MAPS: Dict[str, List[str]] = {
    "lerobot/svla_so100_pickplace": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so100_stacking": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so100_sorting": [
        "main_shoulder_pan", "main_shoulder_lift", "main_elbow_flex",
        "main_wrist_flex", "main_wrist_roll", "main_gripper",
    ],
    "lerobot/svla_so101_pickplace": [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ],
}

# Camera key names vary per source. PLAN.md called these "wrist"/"top" but the
# actual LeRobot v3.0 datasets ship them under different names. We resolve to
# the canonical CANONICAL_VIEWS = (wrist, top) ordering at remap time.
#
# Verified by probing each downloaded source. svla_so101_pickplace ships
# `observation.images.up` (downward-facing wrist-like) + `.side` (3rd-person).
# Probe new sources before adding them.
CAMERA_KEY_MAPS: Dict[str, Dict[str, str]] = {
    "lerobot/svla_so100_pickplace": {
        "wrist": "observation.images.wrist",
        "top":   "observation.images.top",
    },
    "lerobot/svla_so100_stacking": {
        "wrist": "observation.images.wrist",
        "top":   "observation.images.top",
    },
    "lerobot/svla_so100_sorting": {
        "wrist": "observation.images.wrist",
        "top":   "observation.images.top",
    },
    "lerobot/svla_so101_pickplace": {
        "wrist": "observation.images.up",
        "top":   "observation.images.side",
    },
}


PRETRAIN_SOURCES: Tuple[str, ...] = (
    "lerobot/svla_so100_pickplace",
    "lerobot/svla_so100_stacking",
    "lerobot/svla_so100_sorting",
    "lerobot/svla_so101_pickplace",
)


def register_source(
    repo_id: str,
    action_keys: List[str],
    camera_keys: Dict[str, str],
) -> None:
    """Register a custom SO-101 source (e.g. your self-collected task data)
    at runtime. Call this before constructing the DataSource."""
    ACTION_KEY_MAPS[repo_id] = list(action_keys)
    CAMERA_KEY_MAPS[repo_id] = dict(camera_keys)


class SO101DataSource(DataSource):
    """Multi-source SO-101 DataSource.

    Args:
        source_repo_ids: LeRobot repo_ids. Each must be present in
            ACTION_KEY_MAPS and CAMERA_KEY_MAPS (or registered via
            register_source). Pass a single-element list for finetune.
        target_fps: subsampled frame rate. Source is assumed 30 fps; stride
            per source = max(1, source_fps // target_fps). Pass 15 to keep
            half the frames; 30 for no subsampling.
        n_rollout: cap total trajectories (post-flatten) for smoke tests.
        stats_path: path to combined_stats.json (from
            scripts/compute_so101_stats.py). If given, exposes
            self.stats={"action_mean":..., "action_std":..., ...} so
            WorldModelDataset.normalize_action picks it up via
            stats_cache.try_source_stats.
        root: optional LeRobotDataset cache root (forwarded to every source).
        episodes: optional list of episode indices per source. If passed,
            applied uniformly to every source. For per-source filtering use
            multiple DataSources / a wrapper.
        pad_action_dim: zero-pad action's trailing dim to this size. Mirrors
            LeRobotDataSource; used when fine-tuning a checkpoint trained
            against a larger effective action dim.
        camera_dropout_p: probability of zeroing one (random) view per slice
            call. 0 disables. Applied inside load_visual_frames so it stays
            clip-coherent (a single coin flip per slice).
    """

    ACTION_DIM = 6
    SOURCE_FPS_ASSUMED = 30

    def __init__(
        self,
        source_repo_ids: List[str],
        target_fps: int = 15,
        n_rollout: Optional[int] = None,
        stats_path: Optional[str] = None,
        root: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        pad_action_dim: Optional[int] = None,
        camera_dropout_p: float = 0.0,
        video_backend: str = "pyav",
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from torchvision.transforms import v2 as tv_v2
        except ImportError as e:
            raise ImportError(
                "LeRobot is required for SO101DataSource. "
                "Install with: pip install lerobot"
            ) from e

        if not source_repo_ids:
            raise ValueError("source_repo_ids must be non-empty")

        self.source_repo_ids = list(source_repo_ids)
        self.target_fps = int(target_fps)
        self.root = root
        self.episodes = episodes
        self.pad_action_dim = pad_action_dim
        self.camera_dropout_p = float(camera_dropout_p)
        self._v2 = tv_v2

        # Stable identifier for stats_cache._identity (joins all sources).
        self.repo_id = "+".join(self.source_repo_ids)

        # Build per-source state.
        self._sources: List[Dict] = []
        for repo_id in self.source_repo_ids:
            if repo_id not in ACTION_KEY_MAPS:
                raise KeyError(
                    f"No ACTION_KEY_MAPS entry for {repo_id!r}. "
                    "Use so101.register_source(...) before constructing the DataSource."
                )
            if repo_id not in CAMERA_KEY_MAPS:
                raise KeyError(
                    f"No CAMERA_KEY_MAPS entry for {repo_id!r}. "
                    "Use so101.register_source(...) before constructing the DataSource."
                )
            print(f"[SO101] Loading {repo_id} (root={root}, video_backend={video_backend})")
            ds = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                image_transforms=None,
                episodes=episodes,
                video_backend=video_backend,
            )
            src_fps = int(getattr(ds.meta, "fps", self.SOURCE_FPS_ASSUMED))
            stride = max(1, src_fps // self.target_fps)
            self._sources.append({
                "repo_id": repo_id,
                "dataset": ds,
                "action_keys": ACTION_KEY_MAPS[repo_id],
                "camera_keys": CAMERA_KEY_MAPS[repo_id],
                "src_fps": src_fps,
                "stride": stride,
                "num_episodes": int(ds.num_episodes),
            })

        # Flatten (src_idx, episode_idx) into a single trajectory index.
        self._traj_table: List[Tuple[int, int]] = []
        for src_idx, src in enumerate(self._sources):
            for ep_idx in range(src["num_episodes"]):
                self._traj_table.append((src_idx, ep_idx))

        if n_rollout is not None:
            self._traj_table = self._traj_table[: int(n_rollout)]

        if not self._traj_table:
            raise RuntimeError("SO101DataSource: no episodes loaded across sources")

        # Cache actions and canonical lengths so load_trajectory is O(1) after
        # first hit. Frames stay on disk / in the LeRobotDataset video cache.
        self._cached_actions: List[Optional[torch.Tensor]] = [None] * len(self._traj_table)
        self._cached_lengths: List[Optional[int]] = [None] * len(self._traj_table)

        self._native_action_dim = self.ACTION_DIM
        self._action_dim = (
            int(self.pad_action_dim)
            if self.pad_action_dim is not None
            else self._native_action_dim
        )
        if self.pad_action_dim is not None and self.pad_action_dim < self._native_action_dim:
            raise ValueError(
                f"pad_action_dim ({self.pad_action_dim}) must be >= "
                f"native action_dim ({self._native_action_dim})"
            )

        self.stats: Optional[Dict] = None
        if stats_path is not None:
            self._load_combined_stats(stats_path)

        total_eps = len(self._traj_table)
        print(
            f"[SO101] {total_eps} episodes across {len(self._sources)} sources "
            f"(target_fps={self.target_fps}, action_dim={self._action_dim}, "
            f"stats={'combined' if self.stats else 'none'})"
        )

    # --- DataSource API -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return 0

    def get_num_trajectories(self) -> int:
        return len(self._traj_table)

    def get_seq_length(self, index: int) -> int:
        self._check_index(index)
        if self._cached_lengths[index] is None:
            src_idx, ep_idx = self._traj_table[index]
            src = self._sources[src_idx]
            start, end = self._episode_range(src["dataset"], ep_idx)
            source_length = end - start
            self._cached_lengths[index] = source_length // src["stride"]
        return int(self._cached_lengths[index])

    def load_trajectory(self, index: int) -> TrajectoryData:
        self._check_index(index)
        if self._cached_actions[index] is None:
            self._cache_trajectory(index)
        actions = self._cached_actions[index]
        seq_length = self._cached_lengths[index]
        states = torch.zeros(seq_length, 0, dtype=torch.float32)
        src_idx, ep_idx = self._traj_table[index]
        return TrajectoryData(
            states=states,
            actions=actions,
            seq_length=seq_length,
            meta={
                "source_repo_id": self._sources[src_idx]["repo_id"],
                "source_episode_index": ep_idx,
                "stride": self._sources[src_idx]["stride"],
            },
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> torch.Tensor:
        """Return canonical-fps frames in `[start, end)` with stride `step`.

        Output: float32 `[N, 3, H, 2W]` in [0, 1]. Wrist on the left, top on
        the right; missing views are zero-filled. Caller (WorldModelDataset)
        resizes to image_size and scales to [-1, 1].
        """
        self._check_index(index)
        src_idx, ep_idx = self._traj_table[index]
        src = self._sources[src_idx]
        ds = src["dataset"]
        stride = src["stride"]
        ep_start, _ = self._episode_range(ds, ep_idx)

        canonical_indices = list(range(start, end, step))
        if not canonical_indices:
            return torch.empty(0, 3, 0, 0, dtype=torch.float32)

        view_clips: Dict[str, List[torch.Tensor]] = {v: [] for v in CANONICAL_VIEWS}
        for ci in canonical_indices:
            source_frame = ep_start + ci * stride
            sample = ds[source_frame]
            for view in CANONICAL_VIEWS:
                cam_key = src["camera_keys"].get(view)
                view_clips[view].append(self._fetch_image(sample, cam_key))

        per_view = [torch.stack(view_clips[v], dim=0) for v in CANONICAL_VIEWS]
        per_view = self._match_spatial(per_view)

        if self.camera_dropout_p > 0 and torch.rand(1).item() < self.camera_dropout_p:
            drop_idx = int(torch.randint(0, len(per_view), (1,)).item())
            per_view[drop_idx] = torch.zeros_like(per_view[drop_idx])

        # Horizontal concat: V x [T, 3, H, W] -> [T, 3, H, V*W].
        return torch.cat(per_view, dim=-1)

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _episode_range(ds, ep_idx: int) -> Tuple[int, int]:
        """Resolve (global_start, global_end) frame offsets for an episode.

        Supports both old (lerobot <0.5) `episode_data_index` and new
        (>=0.5) `meta.episodes` API. The new API stores per-episode metadata
        as an HF Dataset row with `dataset_from_index` / `dataset_to_index`.
        """
        if hasattr(ds, "episode_data_index"):
            idx_map = ds.episode_data_index
            return int(idx_map["from"][ep_idx]), int(idx_map["to"][ep_idx])
        row = ds.meta.episodes[ep_idx]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= len(self._traj_table):
            raise IndexError(
                f"index {index} out of range [0, {len(self._traj_table)})"
            )

    def _cache_trajectory(self, index: int) -> None:
        src_idx, ep_idx = self._traj_table[index]
        src = self._sources[src_idx]
        ds = src["dataset"]
        action_keys = src["action_keys"]
        stride = src["stride"]
        start, end = self._episode_range(ds, ep_idx)
        source_length = end - start
        canonical_length = source_length // stride

        actions = []
        for ci in range(canonical_length):
            sample = ds[start + ci * stride]
            actions.append(self._fetch_action(sample, action_keys))
        actions_t = torch.stack(actions, dim=0).float()

        if self.pad_action_dim is not None and self.pad_action_dim > self._native_action_dim:
            pad = torch.zeros(
                actions_t.shape[0],
                self.pad_action_dim - self._native_action_dim,
                dtype=actions_t.dtype,
            )
            actions_t = torch.cat([actions_t, pad], dim=-1)

        self._cached_actions[index] = actions_t
        self._cached_lengths[index] = canonical_length

    @staticmethod
    def _fetch_action(sample: Dict, action_keys: List[str]) -> torch.Tensor:
        """Return a [6]-dim canonical action vector.

        Packed `action` Tensor (more common in v3.0) takes priority; falls back
        to per-joint columns in canonical order for v2.1-style sources.
        """
        packed = sample.get("action")
        if isinstance(packed, torch.Tensor) and packed.ndim == 1 and packed.numel() == 6:
            return packed.float()
        return torch.stack([sample[k].float() for k in action_keys], dim=-1)

    def _fetch_image(self, sample: Dict, cam_key: Optional[str]) -> torch.Tensor:
        """Return a [3, H, W] float tensor in [0, 1]. If `cam_key` is missing
        or absent from the sample, return a placeholder [3, 1, 1] that will
        be resized to match the present view at concat time."""
        if cam_key is None or cam_key not in sample:
            return torch.zeros(3, 1, 1, dtype=torch.float32)
        raw = sample[cam_key]
        if isinstance(raw, torch.Tensor) and raw.is_floating_point():
            return raw.float()
        return self._v2.functional.to_image(raw).float() / 255.0

    @staticmethod
    def _match_spatial(per_view: List[torch.Tensor]) -> List[torch.Tensor]:
        """Resize zero-filled placeholders to match the real view's (H, W)."""
        hw_candidates = [
            (v.shape[-2], v.shape[-1])
            for v in per_view
            if v.shape[-2] > 1 and v.shape[-1] > 1
        ]
        if not hw_candidates:
            return per_view
        H, W = hw_candidates[0]
        matched = []
        for v in per_view:
            if v.shape[-2] != H or v.shape[-1] != W:
                v = F.interpolate(v, size=(H, W), mode="nearest")
            matched.append(v)
        return matched

    def _load_combined_stats(self, stats_path: str) -> None:
        p = Path(stats_path)
        if not p.exists():
            print(
                f"[SO101] stats_path={stats_path} does not exist; "
                "WorldModelDataset will compute stats from scratch if asked."
            )
            return
        payload = json.loads(p.read_text())
        mean = torch.tensor(payload["action_mean"], dtype=torch.float32)
        std = torch.tensor(payload["action_std"], dtype=torch.float32)
        if mean.numel() != self._native_action_dim:
            raise ValueError(
                f"stats_path action_mean has {mean.numel()} dims, expected "
                f"{self._native_action_dim} for SO-101"
            )
        if self.pad_action_dim is not None and self.pad_action_dim > self._native_action_dim:
            extra = self.pad_action_dim - self._native_action_dim
            mean = torch.cat([mean, torch.zeros(extra)])
            std = torch.cat([std, torch.ones(extra)])
        self.stats = {
            "action_mean": mean,
            "action_std": std,
        }
        print(f"[SO101] Loaded combined stats from {stats_path}")
