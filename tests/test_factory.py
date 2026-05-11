"""Tests for the patched upstream factory.

Verifies our tartandrive branch is wired up and forwards kwargs correctly.
"""

from pathlib import Path

import pytest

from src.wm_datasets.data_source.factory import create_data_source
from src.wm_datasets.data_source.offroad.tartandrive import TartanDriveDataSource


def test_factory_dispatches_tartandrive(synthetic_split: Path) -> None:
    ds = create_data_source("tartandrive", data_path=str(synthetic_split))
    assert isinstance(ds, TartanDriveDataSource)
    assert ds.action_dim == 2


def test_factory_forwards_n_rollout(synthetic_split: Path) -> None:
    ds = create_data_source("tartandrive", data_path=str(synthetic_split), n_rollout=1)
    assert ds.get_num_trajectories() == 1


def test_factory_filters_unknown_kwargs(synthetic_split: Path) -> None:
    """Unrelated kwargs (e.g. file_list for csgo) must not reach our DataSource."""
    ds = create_data_source(
        "tartandrive",
        data_path=str(synthetic_split),
        file_list="csgo_only_kwarg",   # irrelevant to tartandrive; must be filtered
        use_relative_actions=True,     # same
    )
    assert isinstance(ds, TartanDriveDataSource)


def test_factory_forwards_latents_kwargs(
    synthetic_split: Path, synthetic_latents: Path
) -> None:
    ds = create_data_source(
        "tartandrive",
        data_path=str(synthetic_split),
        latents_path=str(synthetic_latents),
        use_cached_latents=True,
    )
    assert isinstance(ds, TartanDriveDataSource)
    assert ds.use_cached_latents is True


def test_factory_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        create_data_source("not_a_real_dataset", data_path="/tmp")
