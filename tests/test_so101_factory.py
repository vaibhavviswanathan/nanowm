"""Tests for the so101 / so101_task branches of the upstream factory."""

from __future__ import annotations

import sys
import types
from typing import Dict, List

import pytest
import torch


# Re-use the FakeLeRobotDataset setup. Importing the test file directly is
# fine — pytest treats the tests package as a regular package via tests/__init__.py.
from tests.test_so101_datasource import FakeLeRobotDataset, fake_lerobot  # noqa: F401


@pytest.fixture
def registered_source(fake_lerobot):
    from src.wm_datasets.data_source.manipulation import so101

    repo_id = "test/so101_factory"
    fake_lerobot(repo_id, episode_lengths=[20, 12], fps=30)
    so101.register_source(
        repo_id,
        action_keys=so101.ACTION_KEY_MAPS["lerobot/svla_so101_pickplace"],
        camera_keys={"wrist": "observation.images.wrist", "top": "observation.images.top"},
    )
    return repo_id


def test_factory_dispatches_so101_with_source_repo_ids(registered_source):
    from src.wm_datasets.data_source.factory import create_data_source
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
    )
    assert isinstance(ds, SO101DataSource)
    assert ds.action_dim == 6


def test_factory_dispatches_so101_task_with_repo_id(registered_source):
    """so101_task uses single `repo_id` (the finetune ergonomic)."""
    from src.wm_datasets.data_source.factory import create_data_source
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = create_data_source(
        dataset_name="so101_task",
        repo_id=registered_source,
    )
    assert isinstance(ds, SO101DataSource)
    assert ds.source_repo_ids == [registered_source]


def test_factory_forwards_n_rollout(registered_source):
    from src.wm_datasets.data_source.factory import create_data_source

    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
        n_rollout=1,
    )
    assert ds.get_num_trajectories() == 1


def test_factory_forwards_target_fps(registered_source):
    from src.wm_datasets.data_source.factory import create_data_source

    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
        target_fps=30,  # disable subsampling
    )
    # Without subsampling, canonical length == source length.
    assert ds.get_seq_length(0) == 20


def test_factory_forwards_pad_action_dim(registered_source):
    from src.wm_datasets.data_source.factory import create_data_source

    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
        pad_action_dim=8,
    )
    assert ds.action_dim == 8


def test_factory_forwards_stats_path(registered_source, tmp_path):
    import json
    from src.wm_datasets.data_source.factory import create_data_source

    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps({
        "action_mean": [0.0] * 6, "action_std": [1.0] * 6,
    }))
    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
        stats_path=str(stats_file),
    )
    assert ds.stats is not None


def test_factory_filters_unknown_kwargs(registered_source):
    """latents_path / file_list are for other datasets — must not reach SO101."""
    from src.wm_datasets.data_source.factory import create_data_source
    from src.wm_datasets.data_source.manipulation.so101 import SO101DataSource

    ds = create_data_source(
        dataset_name="so101",
        source_repo_ids=[registered_source],
        latents_path="/irrelevant",
        file_list="csgo_only",
        use_cached_latents=True,
    )
    assert isinstance(ds, SO101DataSource)


def test_factory_so101_without_repo_id_raises(fake_lerobot):
    from src.wm_datasets.data_source.factory import create_data_source

    with pytest.raises(ValueError, match="source_repo_ids"):
        create_data_source(dataset_name="so101")
