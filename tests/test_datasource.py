"""Unit tests for TartanDriveDataSource.

These run against synthetic trajectories from conftest. They verify the
DataSource conforms to upstream's abstract contract (TrajectoryData shape,
tensor types, [0,1] pixel range) and that frames stay aligned with actions
across slicing — the documented #1 failure mode for new datasets.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.wm_datasets.data_source.base import DataSource, TrajectoryData
from src.wm_datasets.data_source.offroad.tartandrive import TartanDriveDataSource


# --- Construction & basic API -----------------------------------------------


def test_constructs_from_synthetic_split(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    assert isinstance(ds, DataSource)
    assert ds.get_num_trajectories() == 2


def test_missing_data_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TartanDriveDataSource(data_path=str(tmp_path / "nope"))


def test_empty_data_path_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        TartanDriveDataSource(data_path=str(empty))


def test_action_and_state_dims(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    assert ds.action_dim == 2
    assert ds.state_dim == 0  # pure-vision, must be zero for upstream compat


def test_seq_lengths_match_meta(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    # traj_0001 has T=64, traj_0002 has T=32 (sorted)
    assert ds.get_seq_length(0) == 64
    assert ds.get_seq_length(1) == 32


def test_n_rollout_truncates(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    assert ds.get_num_trajectories() == 1
    assert ds.get_seq_length(0) == 64  # the first one only


def test_out_of_range_index_raises(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    with pytest.raises(IndexError):
        ds.load_trajectory(99)
    with pytest.raises(IndexError):
        ds.load_visual_frames(99, 0, 4)
    with pytest.raises(IndexError):
        ds.load_trajectory(-1)


# --- TrajectoryData contract ------------------------------------------------


def test_load_trajectory_returns_trajectory_data(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    traj = ds.load_trajectory(0)
    assert isinstance(traj, TrajectoryData)
    assert traj.seq_length == 64
    assert "episode_id" in traj.meta
    assert traj.meta["episode_id"] == "traj_0001"


def test_load_trajectory_state_action_lengths_match(synthetic_split: Path) -> None:
    # Triggers TrajectoryData.__post_init__ which asserts the same length.
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    traj = ds.load_trajectory(0)
    assert traj.actions.shape == (64, 2)
    assert traj.states.shape == (64, 0)  # zero-dim placeholder for pure-vision


def test_load_trajectory_action_dtype(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    traj = ds.load_trajectory(0)
    assert traj.actions.dtype == torch.float32
    assert traj.states.dtype == torch.float32


# --- Pixel-frame contract ---------------------------------------------------


def test_load_visual_frames_shape(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    frames = ds.load_visual_frames(0, 0, 8)
    # [N, C, H, W] -- upstream's CSGO reference enforces this
    assert frames.shape == (8, 3, 32, 32)


def test_load_visual_frames_dtype_and_range(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    frames = ds.load_visual_frames(0, 0, 8)
    assert frames.dtype == torch.float32
    assert 0.0 <= float(frames.min()) <= float(frames.max()) <= 1.0


def test_load_visual_frames_step(synthetic_split: Path) -> None:
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    frames = ds.load_visual_frames(0, 0, 8, step=2)
    # 0, 2, 4, 6 -> 4 frames
    assert frames.shape[0] == 4


# --- The alignment property (key correctness check) -------------------------


def _decode_frame_index(frame: torch.Tensor) -> int:
    """Decode marker from a [3, H, W] frame in [0, 1]."""
    return int(round(float(frame[0, 0, 0]) * 255.0))


def test_frame_marker_alignment(synthetic_split: Path) -> None:
    """Frames returned for [start, end) decode back to their source indices.

    This catches off-by-one errors and silent reordering in load_visual_frames.
    """
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    start, end = 5, 20
    frames = ds.load_visual_frames(0, start, end)
    for k in range(end - start):
        assert _decode_frame_index(frames[k]) == start + k, (
            f"frame[{k}] decoded {_decode_frame_index(frames[k])} expected {start + k}"
        )


def test_action_index_alignment(synthetic_split: Path) -> None:
    """traj.actions[t] = [t, -t] for the first trajectory."""
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    traj = ds.load_trajectory(0)
    for t in range(traj.seq_length):
        assert float(traj.actions[t, 0]) == float(t)
        assert float(traj.actions[t, 1]) == float(-t)


def test_different_trajectories_have_distinct_actions(synthetic_split: Path) -> None:
    """A common bug: every load_trajectory returns the same actions."""
    ds = TartanDriveDataSource(data_path=str(synthetic_split))
    t0 = ds.load_trajectory(0)
    t1 = ds.load_trajectory(1)
    # traj_0002 was written with action_offset=1000
    assert float(t0.actions[0, 0]) == 0.0
    assert float(t1.actions[0, 0]) == 1000.0


# --- Cached-latent mode -----------------------------------------------------


def test_use_cached_latents_requires_latents_path(synthetic_split: Path) -> None:
    with pytest.raises(ValueError, match="latents_path"):
        TartanDriveDataSource(
            data_path=str(synthetic_split),
            use_cached_latents=True,
        )


def test_use_cached_latents_missing_dir_raises(synthetic_split: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="latents"):
        TartanDriveDataSource(
            data_path=str(synthetic_split),
            latents_path=str(tmp_path / "no_such"),
            use_cached_latents=True,
        )


def test_cached_latents_load_correct_shape_and_values(
    synthetic_split: Path,
    synthetic_latents: Path,
) -> None:
    ds = TartanDriveDataSource(
        data_path=str(synthetic_split),
        latents_path=str(synthetic_latents),
        use_cached_latents=True,
    )
    # First trajectory: latents are constant = 0.0 (idx=0)
    lat0 = ds.load_visual_frames(0, 0, 8)
    assert lat0.shape == (8, 4, 4, 4)
    assert lat0.dtype == torch.float32
    assert torch.all(lat0 == 0.0)

    # Second trajectory: latents are constant = 1.0 (idx=1)
    lat1 = ds.load_visual_frames(1, 0, 4)
    assert torch.all(lat1 == 1.0)


# --- Bad on-disk data should fail loudly ------------------------------------


def test_actions_with_wrong_dim_raise(tmp_path: Path) -> None:
    """Truncated/extra action dims must be rejected at construction time."""
    import json as _json
    split = tmp_path / "train"
    traj = split / "traj_0001"
    traj.mkdir(parents=True)
    # Wrong action dim: 3 instead of 2
    np.save(traj / "frames.npy", np.zeros((16, 32, 32, 3), dtype=np.uint8))
    np.save(traj / "actions.npy", np.zeros((16, 3), dtype=np.float32))
    with open(traj / "meta.json", "w") as f:
        _json.dump({"length": 16, "source_traj_id": "x", "fps": 10.0}, f)
    with pytest.raises(ValueError, match="action"):
        TartanDriveDataSource(data_path=str(split))


def test_actions_length_mismatch_raises(tmp_path: Path) -> None:
    """meta.length must equal actions.npy length to catch preprocess bugs."""
    import json as _json
    split = tmp_path / "train"
    traj = split / "traj_0001"
    traj.mkdir(parents=True)
    np.save(traj / "frames.npy", np.zeros((16, 32, 32, 3), dtype=np.uint8))
    np.save(traj / "actions.npy", np.zeros((10, 2), dtype=np.float32))  # mismatch
    with open(traj / "meta.json", "w") as f:
        _json.dump({"length": 16, "source_traj_id": "x", "fps": 10.0}, f)
    with pytest.raises(ValueError, match="rows"):
        TartanDriveDataSource(data_path=str(split))
