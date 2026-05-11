"""Shared test fixtures.

Synthetic trajectories encode each frame's index into its pixel `[0, 0, 0]`
value and each action as `(t, -t)`. End-to-end alignment can therefore be
asserted by decoding the marker back from a `__getitem__` result.

Uint8 caps the marker at 255, so trajectory lengths must be < 256 in tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest


def write_synthetic_traj(
    traj_dir: Path,
    T: int,
    H: int = 32,
    W: int = 32,
    fps: float = 10.0,
    action_offset: int = 0,
) -> None:
    """Write a marker-encoded trajectory at `traj_dir`.

    Frame t has pixel `[0, 0, 0]` set to `t` (uint8). Action t is `[t, -t]`
    plus an optional offset so different trajectories are distinguishable.
    """
    if T > 255:
        raise ValueError(f"T={T} > 255; marker overflows uint8")
    traj_dir.mkdir(parents=True, exist_ok=True)

    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        frames[t, 0, 0, 0] = t
    np.save(traj_dir / "frames.npy", frames)

    actions = np.stack(
        [np.arange(T) + action_offset, -(np.arange(T) + action_offset)],
        axis=-1,
    ).astype(np.float32)
    np.save(traj_dir / "actions.npy", actions)

    with open(traj_dir / "meta.json", "w") as f:
        json.dump(
            {"length": int(T), "source_traj_id": traj_dir.name, "fps": float(fps)},
            f,
        )


@pytest.fixture
def synthetic_split(tmp_path: Path) -> Path:
    """Two trajectories with different lengths, written to a single split dir."""
    split_dir = tmp_path / "train"
    write_synthetic_traj(split_dir / "traj_0001", T=64, action_offset=0)
    write_synthetic_traj(split_dir / "traj_0002", T=32, action_offset=1000)
    return split_dir


@pytest.fixture
def synthetic_train_val(tmp_path: Path) -> tuple[Path, Path]:
    """Train + val split, returned as (train_dir, val_dir)."""
    train_dir = tmp_path / "data" / "train"
    val_dir = tmp_path / "data" / "val"
    write_synthetic_traj(train_dir / "traj_0001", T=64)
    write_synthetic_traj(train_dir / "traj_0002", T=48)
    write_synthetic_traj(val_dir / "traj_0001", T=32)
    return train_dir, val_dir


@pytest.fixture
def synthetic_latents(tmp_path: Path, synthetic_split: Path) -> Path:
    """Write fake latents for the synthetic_split fixture.

    Layout: tmp_path / "latents" / "train" / "traj_XXXX" / "latents.npy".
    Latents are `[T, 4, H//8, W//8]` of constant value = trajectory index,
    so we can verify the right file was loaded.
    """
    latents_root = tmp_path / "latents"
    split_basename = synthetic_split.name  # "train"
    split_dir = latents_root / split_basename
    for idx, traj_dir in enumerate(sorted(p for p in synthetic_split.iterdir() if p.is_dir())):
        out = split_dir / traj_dir.name
        out.mkdir(parents=True, exist_ok=True)
        with open(traj_dir / "meta.json") as f:
            T = int(json.load(f)["length"])
        latents = np.full((T, 4, 4, 4), fill_value=float(idx), dtype=np.float32)
        np.save(out / "latents.npy", latents)
    return latents_root
