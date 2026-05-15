"""End-to-end integration tests: SO101DataSource -> WorldModelDataset.

These verify the DataSource clicks into upstream's WorldModelDataset cleanly:
shapes, dtypes, action normalization via combined stats, and most importantly,
frame/action alignment after the 30->15 fps subsample interacts with the WMD's
own frame_interval reshape.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.test_so101_datasource import FakeLeRobotDataset, fake_lerobot  # noqa: F401


@pytest.fixture
def registered_source(fake_lerobot):
    from src.wm_datasets.data_source.manipulation import so101

    repo_id = "test/so101_wmd"
    # Two long episodes so we comfortably exceed num_frames*frame_interval.
    fake_lerobot(repo_id, episode_lengths=[60, 40], fps=30)
    so101.register_source(
        repo_id,
        action_keys=so101.ACTION_KEY_MAPS["lerobot/svla_so101_pickplace"],
        camera_keys={"wrist": "observation.images.wrist", "top": "observation.images.top"},
    )
    return repo_id


def _decode_marker(video_frame: torch.Tensor, normalize_pixel: bool) -> int:
    """Decode wrist-half marker from a [3, H, W] clip frame (column 0)."""
    v = float(video_frame[0, 0, 0])
    if normalize_pixel:
        v = (v + 1.0) / 2.0
    return int(round(v * 255.0))


def test_clip_shape_and_dtype(registered_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource
    from src.wm_datasets.world_model_dataset import WorldModelDataset

    ds = SO101DataSource(source_repo_ids=[registered_source], target_fps=15, n_rollout=1)
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(32, 32),  # square; squashes 2W -> 32 from native 2*24
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    assert sample["video"].shape == (4, 3, 32, 32)
    assert sample["action"].shape == (4, 6)  # native action_dim
    assert sample["video"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32


def test_pixel_normalization_to_minus_one_one(registered_source):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource
    from src.wm_datasets.world_model_dataset import WorldModelDataset

    ds = SO101DataSource(source_repo_ids=[registered_source], target_fps=15, n_rollout=1)
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
    assert float(sample["video"].min()) >= -1.0
    assert float(sample["video"].max()) <= 1.0


def test_action_normalization_uses_combined_stats(registered_source, tmp_path):
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource
    from src.wm_datasets.world_model_dataset import WorldModelDataset

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "action_mean": [10.0, -10.0, 0.0, 0.0, 0.0, 0.0],
        "action_std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }))
    ds = SO101DataSource(
        source_repo_ids=[registered_source],
        target_fps=15,
        n_rollout=1,
        stats_path=str(stats_path),
    )
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(32, 32),
        split="train",
        split_ratio=1.0,
        normalize_action=True,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    # WMD should pick up our combined stats via data_source.stats.
    assert torch.allclose(wmd._raw_action_mean, torch.tensor([10.0, -10.0, 0.0, 0.0, 0.0, 0.0]))


def test_frame_action_alignment_at_target_fps_15(registered_source):
    """Canonical i ↔ source 2*i; the clip at slot 0 starts at canonical 0.

    With WMD frame_interval=1 and DataSource target_fps=15 (stride 2 over
    30 fps source), clip frame k decodes back to global source frame 2*k.
    """
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource
    from src.wm_datasets.world_model_dataset import WorldModelDataset

    ds = SO101DataSource(source_repo_ids=[registered_source], target_fps=15, n_rollout=1)
    # Use image_size that matches the native concat so no bilinear blurring of
    # the [0,0,0] marker pixel happens.
    H = FakeLeRobotDataset.H
    W = FakeLeRobotDataset.W
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=1,
        image_size=(H, 2 * W),  # explicit non-default; matches DataSource output
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    for k in range(4):
        marker = _decode_marker(sample["video"][k], normalize_pixel=False)
        assert marker == 2 * k, (k, marker)
        # action[k, 0] is action of the canonical frame, which the fake set to
        # the global source frame index = 2*k.
        assert float(sample["action"][k, 0]) == float(2 * k)


def test_frame_action_alignment_with_wmd_frame_interval_2(registered_source):
    """target_fps=15 + WMD frame_interval=2 -> visit every 4th source frame.

    Action gets the frame_interval reshape: per-step dim becomes
    action_dim * frame_interval = 6 * 2 = 12. The first 6 dims are the action
    at canonical k*fi, next 6 are action at canonical k*fi + 1.
    """
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource
    from src.wm_datasets.world_model_dataset import WorldModelDataset

    ds = SO101DataSource(source_repo_ids=[registered_source], target_fps=15, n_rollout=1)
    H = FakeLeRobotDataset.H
    W = FakeLeRobotDataset.W
    wmd = WorldModelDataset(
        data_source=ds,
        num_frames=4,
        frame_interval=2,
        image_size=(H, 2 * W),
        split="train",
        split_ratio=1.0,
        normalize_action=False,
        normalize_pixel=False,
        slice_mode="exhaustive",
    )
    sample = wmd[0]
    assert sample["action"].shape == (4, 12)
    for k in range(4):
        # Video frame k comes from canonical 2k -> source 4k.
        marker = _decode_marker(sample["video"][k], normalize_pixel=False)
        assert marker == 4 * k, (k, marker)
        # action[k, 0:6] is canonical action 2k -> source 4k.
        # action[k, 6:12] is canonical 2k+1 -> source 4k+2.
        assert float(sample["action"][k, 0]) == float(4 * k)
        assert float(sample["action"][k, 6]) == float(4 * k + 2)
