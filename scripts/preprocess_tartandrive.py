"""Finalize a TartanDrive staging directory into the DataSource layout.

This script consumes the output of `scripts/convert_bags.py` (which has already
done the rosbag -> per-trajectory `frames.npy` + `actions.npy` + `meta.json`
conversion at the target resolution) and does the remaining two things:

  1. Split trajectories into train/ and val/ deterministically.
  2. Compute action statistics on the train split for sanity-checking
     normalization assumptions.

Input layout (from `convert_bags.py`):

    staging_dir/
      <bag_stem>/
        frames.npy
        actions.npy
        meta.json

Output layout (consumed by `TartanDriveDataSource`):

    out_dir/
      train/traj_0001/{frames.npy, actions.npy, meta.json}
      train/traj_0002/...
      val/traj_0001/...
      stats.json

Usage:
    uv run python scripts/preprocess_tartandrive.py \
        --staging_dir $DATASET_DIR/tartandrive_staging \
        --out_dir $DATASET_DIR/tartandrive \
        --val_fraction 0.1 --seed 42 --mode move

Modes:
    --mode move (default): rename source dirs into out_dir; staging ends empty.
    --mode copy: leave staging intact; out_dir contains independent copies.

Re-run safety: existing trajectories in out_dir/{train,val} are NOT overwritten;
re-runs assign newly-staged bags to a split based on the deterministic seed and
the bag stem.
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _validate_staging_traj(staging_traj_dir: Path) -> bool:
    """Sanity check: required files present and shapes consistent."""
    f = staging_traj_dir / "frames.npy"
    a = staging_traj_dir / "actions.npy"
    m = staging_traj_dir / "meta.json"
    if not (f.exists() and a.exists() and m.exists()):
        return False
    try:
        frames_shape = np.load(f, mmap_mode="r").shape
        actions = np.load(a)
        with open(m) as fh:
            meta = json.load(fh)
    except Exception:
        return False
    if frames_shape[0] != actions.shape[0] or frames_shape[0] != int(meta["length"]):
        return False
    return True


def _split_assignment(bag_stem: str, val_fraction: float, seed: int) -> str:
    """Deterministic 'train' vs 'val' from a stem + seed (hash-based)."""
    h = hashlib.sha256(f"{seed}:{bag_stem}".encode()).hexdigest()
    u = int(h[:16], 16) / float(1 << 64)
    return "val" if u < val_fraction else "train"


def _next_traj_index(split_dir: Path) -> int:
    """Smallest unused 4-digit index in split_dir/traj_XXXX/."""
    existing = []
    for p in split_dir.glob("traj_????"):
        try:
            existing.append(int(p.name.split("_")[-1]))
        except ValueError:
            continue
    return (max(existing) + 1) if existing else 1


def _transfer(src: Path, dst: Path, mode: str) -> None:
    """Move or copy `src` -> `dst`. Parent dir must exist."""
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing: {dst}")
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"mode must be 'move' or 'copy', got {mode!r}")


def finalize(
    staging_dir: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    mode: str,
) -> Tuple[int, int, List[np.ndarray]]:
    """Walk staging_dir, split bags into train/val, transfer trajectories.

    Returns (n_train, n_val, all_train_actions_for_stats).
    """
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    staged = sorted(p for p in staging_dir.iterdir() if p.is_dir())
    n_train = 0
    n_val = 0
    train_actions: List[np.ndarray] = []

    for staged_dir in staged:
        if not _validate_staging_traj(staged_dir):
            print(f"  skip invalid: {staged_dir.name}")
            continue

        split = _split_assignment(staged_dir.name, val_fraction, seed)
        target_split_dir = out_train if split == "train" else out_val
        idx = _next_traj_index(target_split_dir)
        dst = target_split_dir / f"traj_{idx:04d}"

        _transfer(staged_dir, dst, mode)

        with open(dst / "meta.json") as fh:
            meta = json.load(fh)
        meta["assigned_split"] = split
        meta["final_traj_id"] = dst.name
        with open(dst / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        if split == "train":
            train_actions.append(np.load(dst / "actions.npy"))
            n_train += 1
        else:
            n_val += 1

    return n_train, n_val, train_actions


def write_stats(out_dir: Path, n_train: int, n_val: int, train_actions: List[np.ndarray]) -> dict:
    """Compute action stats across the train set and write `out_dir/stats.json`."""
    stats = {
        "n_train_trajectories": int(n_train),
        "n_val_trajectories": int(n_val),
    }
    if train_actions:
        all_actions = np.concatenate(train_actions, axis=0)
        stats.update(
            {
                "n_train_frames": int(len(all_actions)),
                "action_dim": int(all_actions.shape[-1]),
                "action_min": all_actions.min(axis=0).tolist(),
                "action_max": all_actions.max(axis=0).tolist(),
                "action_mean": all_actions.mean(axis=0).tolist(),
                "action_std": all_actions.std(axis=0).tolist(),
            }
        )
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--staging_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=["move", "copy"], default="move",
        help="'move' (default): staging ends empty. 'copy': keep staging intact.",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    if not args.staging_dir.exists():
        print(f"staging_dir does not exist: {args.staging_dir}", file=sys.stderr)
        return 2

    print(f"Finalizing {args.staging_dir} -> {args.out_dir} (mode={args.mode})")
    n_train, n_val, train_actions = finalize(
        args.staging_dir, args.out_dir,
        val_fraction=args.val_fraction, seed=args.seed, mode=args.mode,
    )
    print(f"\nSplit: {n_train} train, {n_val} val")

    stats = write_stats(args.out_dir, n_train, n_val, train_actions)
    print("\nAction stats (train):")
    print(json.dumps(stats, indent=2))
    print(f"\nDone. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
