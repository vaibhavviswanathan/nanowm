"""Unit tests for SO101DataSource.

LeRobot isn't installed (the project pin keeps `diffusers==0.24.0` and
installing lerobot would force a newer one). So instead of touching the
network, we inject a fake `lerobot.datasets.lerobot_dataset` module into
`sys.modules` before constructing the DataSource.

The fake encodes the canonical-frame index into pixel `[0, 0, 0]` of each
view and the action into the joint vector, mirroring the marker convention
used by the TartanDrive tests. That lets us verify alignment end-to-end:
30 fps source -> 15 fps canonical -> sliced clip -> the right markers.
"""

from __future__ import annotations

import sys
import types
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fake LeRobotDataset
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, fps: int):
        self.fps = fps


class FakeLeRobotDataset:
    """Minimal LeRobotDataset substitute for tests.

    Episodes are arranged contiguously in a global frame buffer; the
    `episode_data_index` dict-of-tensors mirrors the v2.1 layout that
    SO101DataSource consumes.

    Frame t of episode e contains:
      - action: torch.tensor([global_idx, -global_idx, e, 0, 0, 0])
      - observation.images.wrist: [3, H, W] with [0, 0, 0] = global_idx / 255
      - observation.images.top:   [3, H, W] with [0, 0, 0] = (global_idx + 100) / 255
    """

    H = 16
    W = 24  # rectangular to expose any HxW vs WxH bugs

    def __init__(
        self,
        repo_id: str,
        root=None,
        image_transforms=None,
        episodes=None,
        episode_lengths: Optional[List[int]] = None,
        fps: int = 30,
        video_backend: str = "pyav",  # accepted but no-op in the fake
    ):
        self.repo_id = repo_id
        episode_lengths = episode_lengths or [20, 16]
        if episodes is not None:
            episode_lengths = [episode_lengths[i] for i in episodes]
        self._episode_lengths = list(episode_lengths)
        self.num_episodes = len(self._episode_lengths)
        self.meta = _FakeMeta(fps)

        starts = [0]
        for n in self._episode_lengths:
            starts.append(starts[-1] + n)
        self._total = starts[-1]
        self.episode_data_index = {
            "from": torch.tensor(starts[:-1], dtype=torch.long),
            "to": torch.tensor(starts[1:], dtype=torch.long),
        }

    def __len__(self) -> int:
        return self._total

    def _episode_of(self, global_idx: int) -> int:
        # Linear search is fine for test sizes.
        froms = self.episode_data_index["from"].tolist()
        tos = self.episode_data_index["to"].tolist()
        for e, (lo, hi) in enumerate(zip(froms, tos)):
            if lo <= global_idx < hi:
                return e
        raise IndexError(global_idx)

    def __getitem__(self, global_idx: int) -> Dict:
        if global_idx < 0 or global_idx >= self._total:
            raise IndexError(global_idx)
        e = self._episode_of(global_idx)
        # Action: packed 6-vector with the global frame index as joint 0.
        action = torch.tensor(
            [float(global_idx), -float(global_idx), float(e), 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        wrist = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        top = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        wrist[0, 0, 0] = global_idx / 255.0
        top[0, 0, 0] = (global_idx + 100) / 255.0
        return {
            "action": action,
            "observation.images.wrist": wrist,
            "observation.images.top": top,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_lerobot(monkeypatch):
    """Install the fake lerobot module under the import path SO101DataSource uses.

    Yields a callable that lets tests register per-repo_id episode lengths.
    """
    fake_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    lengths_by_repo: Dict[str, List[int]] = {}
    fps_by_repo: Dict[str, int] = {}

    class _Bound(FakeLeRobotDataset):
        def __init__(self, repo_id, root=None, image_transforms=None, episodes=None,
                     video_backend="pyav"):
            super().__init__(
                repo_id=repo_id, root=root, image_transforms=image_transforms,
                episodes=episodes,
                episode_lengths=lengths_by_repo.get(repo_id, [20, 16]),
                fps=fps_by_repo.get(repo_id, 30),
                video_backend=video_backend,
            )

    fake_mod.LeRobotDataset = _Bound

    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", fake_mod)

    def configure(repo_id: str, episode_lengths: List[int], fps: int = 30) -> None:
        lengths_by_repo[repo_id] = list(episode_lengths)
        fps_by_repo[repo_id] = fps

    yield configure


@pytest.fixture
def synthetic_source(fake_lerobot):
    """A single registered SO-101 source with two episodes."""
    from src.wm_datasets.data_source.manipulation import so101

    repo_id = "test/so101_synthetic"
    fake_lerobot(repo_id, episode_lengths=[30, 16], fps=30)
    so101.register_source(
        repo_id,
        action_keys=so101.ACTION_KEY_MAPS["lerobot/svla_so101_pickplace"],
        camera_keys={"wrist": "observation.images.wrist", "top": "observation.images.top"},
    )
    return repo_id


@pytest.fixture
def two_synthetic_sources(fake_lerobot):
    from src.wm_datasets.data_source.manipulation import so101

    out = []
    for repo_id, lengths, fps in [
        ("test/so101_synth_A", [20, 24], 30),
        ("test/so101_synth_B", [18], 30),
    ]:
        fake_lerobot(repo_id, episode_lengths=lengths, fps=fps)
        so101.register_source(
            repo_id,
            action_keys=so101.ACTION_KEY_MAPS["lerobot/svla_so101_pickplace"],
            camera_keys={"wrist": "observation.images.wrist", "top": "observation.images.top"},
        )
        out.append(repo_id)
    return out


# ---------------------------------------------------------------------------
# Construction & basic API
# ---------------------------------------------------------------------------


def test_constructs_single_source(synthetic_source):
    from src.wm_datasets.data_source.base import DataSource
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source])
    assert isinstance(ds, DataSource)
    assert ds.get_num_trajectories() == 2  # two episodes


def test_action_and_state_dim(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source])
    assert ds.action_dim == 6
    assert ds.state_dim == 0


def test_subsampling_30_to_15_fps(synthetic_source):
    """Source 30 fps, target 15 fps -> stride 2 -> canonical_len = src_len // 2."""
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    # Episode 0: source length 30 -> canonical 15
    # Episode 1: source length 16 -> canonical 8
    assert ds.get_seq_length(0) == 15
    assert ds.get_seq_length(1) == 8


def test_no_subsampling_when_target_equals_source(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=30)
    assert ds.get_seq_length(0) == 30
    assert ds.get_seq_length(1) == 16


def test_multi_source_flattens_episodes(two_synthetic_sources):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=two_synthetic_sources)
    # 2 episodes from A + 1 from B = 3
    assert ds.get_num_trajectories() == 3


def test_n_rollout_truncates(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], n_rollout=1)
    assert ds.get_num_trajectories() == 1


def test_unregistered_repo_raises(fake_lerobot):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    fake_lerobot("test/unregistered", episode_lengths=[10])
    with pytest.raises(KeyError, match="ACTION_KEY_MAPS"):
        SO101DataSource(source_repo_ids=["test/unregistered"])


def test_out_of_range_index_raises(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source])
    with pytest.raises(IndexError):
        ds.load_trajectory(99)
    with pytest.raises(IndexError):
        ds.load_visual_frames(99, 0, 4)


# ---------------------------------------------------------------------------
# TrajectoryData contract
# ---------------------------------------------------------------------------


def test_load_trajectory_returns_canonical_actions(synthetic_source):
    """At 30->15 fps (stride 2), canonical action[i] should come from source[i*2]."""
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    traj = ds.load_trajectory(0)
    assert traj.actions.shape == (15, 6)
    # Fake encodes action[0] of each frame as global_frame_index. Episode 0
    # starts at global 0, stride 2 -> canonical i ↔ global 2i.
    for i in range(15):
        assert float(traj.actions[i, 0]) == float(2 * i)


def test_load_trajectory_distinct_per_episode(synthetic_source):
    """Episode 1 starts at global frame 30 in the fake; markers reflect that."""
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    t0 = ds.load_trajectory(0)
    t1 = ds.load_trajectory(1)
    assert float(t0.actions[0, 0]) == 0.0
    assert float(t1.actions[0, 0]) == 30.0  # start of episode 1 in global frames


def test_load_trajectory_meta(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    traj = ds.load_trajectory(1)
    assert traj.meta["source_repo_id"] == synthetic_source
    assert traj.meta["source_episode_index"] == 1
    assert traj.meta["stride"] == 2


# ---------------------------------------------------------------------------
# Pixel-frame contract (multi-view horizontal concat)
# ---------------------------------------------------------------------------


def test_load_visual_frames_shape(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    frames = ds.load_visual_frames(0, 0, 8)
    # [T, 3, H, 2W] with H=16, W=24 from the fake -> width 48
    H = FakeLeRobotDataset.H
    W = FakeLeRobotDataset.W
    assert frames.shape == (8, 3, H, 2 * W)


def test_load_visual_frames_range_and_dtype(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    frames = ds.load_visual_frames(0, 0, 8)
    assert frames.dtype == torch.float32
    assert 0.0 <= float(frames.min()) <= float(frames.max()) <= 1.0


def test_load_visual_frames_subsample_alignment(synthetic_source):
    """Canonical frame i comes from source frame i*stride.

    Wrist marker at pixel [0,0,0] should decode back to the global source
    frame index.
    """
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    frames = ds.load_visual_frames(0, 0, 8)  # canonical 0..7
    W = FakeLeRobotDataset.W

    for i in range(8):
        wrist_marker = round(float(frames[i, 0, 0, 0]) * 255.0)
        top_marker = round(float(frames[i, 0, 0, W]) * 255.0)  # first col of top half
        expected_global = 2 * i  # stride=2, episode 0 starts at 0
        assert wrist_marker == expected_global, (i, wrist_marker, expected_global)
        assert top_marker == expected_global + 100, (i, top_marker)


def test_horizontal_concat_wrist_then_top(synthetic_source):
    """Wrist occupies columns [0, W); top occupies columns [W, 2W).

    Order is the source of truth for the multi-view hack — flipping it
    would break inference-time alignment with the live robot camera feed.
    """
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    frames = ds.load_visual_frames(0, 0, 1)
    W = FakeLeRobotDataset.W
    wrist_left = round(float(frames[0, 0, 0, 0]) * 255.0)
    top_left = round(float(frames[0, 0, 0, W]) * 255.0)
    assert wrist_left == 0
    assert top_left == 100


def test_load_visual_frames_step(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], target_fps=15)
    frames = ds.load_visual_frames(0, 0, 8, step=2)  # canonical 0, 2, 4, 6
    assert frames.shape[0] == 4
    # 0, 4, 8, 12 in source frames
    for k, expected_global in enumerate([0, 4, 8, 12]):
        wrist_marker = round(float(frames[k, 0, 0, 0]) * 255.0)
        assert wrist_marker == expected_global, (k, wrist_marker, expected_global)


# ---------------------------------------------------------------------------
# Combined stats
# ---------------------------------------------------------------------------


def test_loads_combined_stats(synthetic_source, tmp_path):
    import json
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "action_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "action_std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "n_samples": 1234,
        "sources": [synthetic_source],
    }))
    ds = SO101DataSource(
        source_repo_ids=[synthetic_source],
        stats_path=str(stats_path),
    )
    assert ds.stats is not None
    assert torch.allclose(
        ds.stats["action_mean"], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    )


def test_missing_stats_path_does_not_crash(synthetic_source, tmp_path):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(
        source_repo_ids=[synthetic_source],
        stats_path=str(tmp_path / "does_not_exist.json"),
    )
    assert ds.stats is None


def test_stats_wrong_dim_raises(synthetic_source, tmp_path):
    import json
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "action_mean": [0.0, 0.0],  # wrong dim
        "action_std":  [1.0, 1.0],
    }))
    with pytest.raises(ValueError, match="6"):
        SO101DataSource(
            source_repo_ids=[synthetic_source],
            stats_path=str(stats_path),
        )


# ---------------------------------------------------------------------------
# pad_action_dim (mirrors LeRobotDataSource)
# ---------------------------------------------------------------------------


def test_pad_action_dim_extends_actions(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = SO101DataSource(source_repo_ids=[synthetic_source], pad_action_dim=10)
    assert ds.action_dim == 10
    traj = ds.load_trajectory(0)
    assert traj.actions.shape[-1] == 10
    # First 6 dims preserved; last 4 are zero pad.
    assert torch.all(traj.actions[:, 6:] == 0.0)


def test_pad_smaller_than_native_raises(synthetic_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    with pytest.raises(ValueError, match="pad_action_dim"):
        SO101DataSource(source_repo_ids=[synthetic_source], pad_action_dim=3)


def test_pad_with_stats(synthetic_source, tmp_path):
    """Combined stats are padded to match pad_action_dim with neutral values."""
    import json
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "action_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "action_std":  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }))
    ds = SO101DataSource(
        source_repo_ids=[synthetic_source],
        stats_path=str(stats_path),
        pad_action_dim=8,
    )
    assert ds.stats["action_mean"].shape == (8,)
    # Pad dims are mean=0, std=1 (neutral normalization).
    assert torch.allclose(ds.stats["action_mean"][6:], torch.zeros(2))
    assert torch.allclose(ds.stats["action_std"][6:], torch.ones(2))
