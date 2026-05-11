"""Tests for scripts/preprocess_tartandrive.py (staging -> train/val + stats).

The script's input is what scripts/convert_bags.py produces. These tests
exercise the post-bag-conversion finalize path against synthetic fixtures so
we never need a real rosbag to run.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


# Load the script as a module (it's not on the import path under scripts/).
_PRE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_tartandrive.py"
_spec = importlib.util.spec_from_file_location("_pre", _PRE_PATH)
preprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocess)


def _write_staged_bag(staging: Path, bag_stem: str, T: int = 32) -> Path:
    """Write a fake convert_bags output for one bag."""
    d = staging / bag_stem
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.npy", np.zeros((T, 32, 32, 3), dtype=np.uint8))
    np.save(d / "actions.npy", np.full((T, 2), 0.5, dtype=np.float32))
    with open(d / "meta.json", "w") as f:
        json.dump(
            {
                "length": T,
                "source_bag": f"/tmp/{bag_stem}.bag",
                "source_traj_id": bag_stem,
                "fps": 10.0,
            },
            f,
        )
    return d


def test_split_assignment_deterministic() -> None:
    """Same (stem, seed, fraction) -> same split, every call."""
    a = preprocess._split_assignment("bag_001", 0.1, 42)
    b = preprocess._split_assignment("bag_001", 0.1, 42)
    assert a == b


def test_split_assignment_changes_with_seed() -> None:
    """Different seeds produce different distributions across many stems."""
    stems = [f"bag_{i:04d}" for i in range(200)]
    s1 = [preprocess._split_assignment(s, 0.5, seed=1) for s in stems]
    s2 = [preprocess._split_assignment(s, 0.5, seed=2) for s in stems]
    assert s1 != s2  # not the same assignment under different seeds


def test_split_assignment_respects_val_fraction() -> None:
    """Over many trajectories, val_fraction is approximated."""
    stems = [f"bag_{i:04d}" for i in range(1000)]
    assignments = [preprocess._split_assignment(s, 0.1, seed=42) for s in stems]
    val_frac = assignments.count("val") / len(assignments)
    # 10% target; binomial std at n=1000 is ~0.0095 -> generous 3-sigma window
    assert 0.07 <= val_frac <= 0.13, val_frac


def test_validate_rejects_missing_files(tmp_path: Path) -> None:
    bad = tmp_path / "no_files"
    bad.mkdir()
    assert preprocess._validate_staging_traj(bad) is False


def test_validate_rejects_length_mismatch(tmp_path: Path) -> None:
    d = tmp_path / "mismatch"
    d.mkdir()
    np.save(d / "frames.npy", np.zeros((10, 32, 32, 3), dtype=np.uint8))
    np.save(d / "actions.npy", np.zeros((20, 2), dtype=np.float32))
    with open(d / "meta.json", "w") as f:
        json.dump({"length": 10, "source_traj_id": "x", "fps": 10.0}, f)
    assert preprocess._validate_staging_traj(d) is False


def test_validate_accepts_consistent_traj(tmp_path: Path) -> None:
    _write_staged_bag(tmp_path, "ok")
    assert preprocess._validate_staging_traj(tmp_path / "ok") is True


def test_finalize_move_empties_staging(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    for i in range(3):
        _write_staged_bag(staging, f"bag_{i:04d}")
    n_train, n_val, train_actions = preprocess.finalize(
        staging, out, val_fraction=0.1, seed=42, mode="move",
    )
    assert n_train + n_val == 3
    # 'move' should empty staging
    assert sum(1 for _ in staging.iterdir()) == 0


def test_finalize_copy_preserves_staging(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    _write_staged_bag(staging, "bag_only")
    preprocess.finalize(staging, out, val_fraction=0.5, seed=1, mode="copy")
    assert (staging / "bag_only").exists()  # still there


def test_finalize_assigns_canonical_indices(tmp_path: Path) -> None:
    """Output dirs get traj_NNNN names regardless of source name."""
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    _write_staged_bag(staging, "messy_bag_name_xyz")
    n_train, n_val, _ = preprocess.finalize(
        staging, out, val_fraction=0.0, seed=42, mode="move"
    )
    assert n_train == 1 and n_val == 0
    assert (out / "train" / "traj_0001").exists()


def test_finalize_resumable(tmp_path: Path) -> None:
    """Re-running on a fresh staging dir continues numbering from the highest."""
    staging1 = tmp_path / "staging1"
    out = tmp_path / "out"
    _write_staged_bag(staging1, "bag_a")
    preprocess.finalize(staging1, out, val_fraction=0.0, seed=42, mode="move")
    staging2 = tmp_path / "staging2"
    _write_staged_bag(staging2, "bag_b")
    preprocess.finalize(staging2, out, val_fraction=0.0, seed=42, mode="move")
    # Numbering continues
    assert (out / "train" / "traj_0001").exists()
    assert (out / "train" / "traj_0002").exists()


def test_finalize_records_split_and_id_in_meta(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    _write_staged_bag(staging, "bag_q")
    preprocess.finalize(staging, out, val_fraction=0.0, seed=42, mode="move")
    with open(out / "train" / "traj_0001" / "meta.json") as f:
        meta = json.load(f)
    assert meta["assigned_split"] == "train"
    assert meta["final_traj_id"] == "traj_0001"


def test_finalize_skips_invalid(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    out = tmp_path / "out"
    _write_staged_bag(staging, "good")
    (staging / "bad").mkdir()  # missing files
    n_train, n_val, _ = preprocess.finalize(
        staging, out, val_fraction=0.0, seed=42, mode="move"
    )
    assert n_train + n_val == 1


def test_write_stats_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    a = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    b = np.array([[0.5, 0.5]], dtype=np.float32)
    stats = preprocess.write_stats(out, n_train=2, n_val=0, train_actions=[a, b])
    assert stats["action_dim"] == 2
    assert stats["n_train_frames"] == 4
    on_disk = json.loads((out / "stats.json").read_text())
    assert on_disk == stats


def test_finalize_end_to_end_produces_loadable_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output of finalize() must be consumable by TartanDriveDataSource."""
    from src.wm_datasets.data_source.offroad.tartandrive import TartanDriveDataSource

    staging = tmp_path / "staging"
    out = tmp_path / "out"
    for i in range(5):
        _write_staged_bag(staging, f"bag_{i:04d}")
    preprocess.finalize(staging, out, val_fraction=0.2, seed=42, mode="move")
    preprocess.write_stats(out, 4, 1, [])  # placeholder stats — not the focus

    # Both splits should be loadable independently.
    train_ds = TartanDriveDataSource(data_path=str(out / "train"))
    val_ds = TartanDriveDataSource(data_path=str(out / "val"))
    assert train_ds.get_num_trajectories() + val_ds.get_num_trajectories() == 5
