"""End-to-end alignment tests through WorldModelDataset.

These wire our DataSource into upstream's WorldModelDataset and check that the
clip returned by `__getitem__` has the right shape, dtype, range, AND that
frames stay aligned with their corresponding actions across the
frame_interval reshape — the place where off-by-one errors are most likely.

The synthetic fixture encodes frame index t into pixel `[0, 0, 0] = t` and
actions as `[t, -t]`. After a slice starting at `start` with `frame_interval`,
the k-th sample in the clip should correspond to source index `start + k*fi`.
"""

from pathlib import Path

import torch

from src.wm_datasets.data_source.offroad.tartandrive import TartanDriveDataSource
from src.wm_datasets.world_model_dataset import WorldModelDataset


def _decode_marker(video_frame: torch.Tensor, normalize_pixel: bool) -> int:
    """Decode the synthetic marker from a [3, H, W] frame."""
    if normalize_pixel:
        # WorldModelDataset does video * 2 - 1, so 0..1 becomes -1..1
        v = (float(video_frame[0, 0, 0]) + 1.0) / 2.0
    else:
        v = float(video_frame[0, 0, 0])
    return int(round(v * 255.0))


def test_clip_shape_and_dtype(synthetic_split: Path) -> None:
    # n_rollout=1 keeps only traj_0001 (action_offset=0) so frame markers and
    # action values match directly without trajectory bookkeeping.
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    assert "video" in sample and "action" in sample
    assert sample["video"].shape == (4, 3, 32, 32)
    assert sample["action"].shape == (4, 2)  # frame_interval=1 -> dim = action_dim
    assert sample["video"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32


def test_frame_interval_reshapes_action(synthetic_split: Path) -> None:
    # n_rollout=1 keeps only traj_0001 (action_offset=0) so frame markers and
    # action values match directly without trajectory bookkeeping.
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=2,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    # frame_interval doubles the per-step action dim: 4 = 2 (action_dim) * 2 (fi)
    assert sample["video"].shape == (4, 3, 32, 32)
    assert sample["action"].shape == (4, 4)


def test_end_to_end_alignment_fi1(synthetic_split: Path) -> None:
    """frame_interval=1: video[k] and action[k] both correspond to source index k."""
    # n_rollout=1 keeps only traj_0001 (action_offset=0) so frame markers and
    # action values match directly without trajectory bookkeeping.
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    # First slice of the first trajectory starts at frame 0 (sorted, stride=1)
    sample = wmd[0]
    traj_idx = sample["meta_info"]["traj_idx"]
    start = sample["meta_info"]["start_idx"]
    for k in range(4):
        expected = start + k
        assert _decode_marker(sample["video"][k], normalize_pixel=False) == expected, (
            f"video[{k}] traj={traj_idx} start={start} expected marker={expected}"
        )
        assert float(sample["action"][k, 0]) == float(expected)
        assert float(sample["action"][k, 1]) == float(-expected)


def test_end_to_end_alignment_fi2(synthetic_split: Path) -> None:
    """frame_interval=2: video[k] comes from source index start+k*2.

    Action[k] holds the *pair* (action[start+2k], action[start+2k+1]) flattened
    to 4 dims: [throttle_a, steer_a, throttle_b, steer_b].
    """
    # n_rollout=1 keeps only traj_0001 (action_offset=0) so frame markers and
    # action values match directly without trajectory bookkeeping.
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=2,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    start = sample["meta_info"]["start_idx"]

    for k in range(4):
        source_idx = start + k * 2
        assert _decode_marker(sample["video"][k], normalize_pixel=False) == source_idx
        # Action: [t, -t, t+1, -(t+1)]
        a = sample["action"][k]
        assert float(a[0]) == float(source_idx)
        assert float(a[1]) == float(-source_idx)
        assert float(a[2]) == float(source_idx + 1)
        assert float(a[3]) == float(-(source_idx + 1))


def test_normalize_pixel_rescales_to_minus_one_one(synthetic_split: Path) -> None:
    # n_rollout=1 keeps only traj_0001 (action_offset=0) so frame markers and
    # action values match directly without trajectory bookkeeping.
    ds = TartanDriveDataSource(data_path=str(synthetic_split), n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=True,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    v = sample["video"]
    assert v.min() >= -1.0 - 1e-6
    assert v.max() <= 1.0 + 1e-6


def test_cached_latents_passthrough_via_world_model_dataset(
    synthetic_split: Path, synthetic_latents: Path
) -> None:
    """With use_cached_latents=True the 'video' field returns latents."""
    ds = TartanDriveDataSource(
        data_path=str(synthetic_split),
        latents_path=str(synthetic_latents),
        use_cached_latents=True,
    )
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(4, 4),  # match latent spatial dims so no resize triggers
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    # latents are [N, 4, 4, 4] not [N, 3, H, W] — confirms our DataSource is
    # returning the latents tensor unchanged through the slice path.
    assert sample["video"].shape == (4, 4, 4, 4)
