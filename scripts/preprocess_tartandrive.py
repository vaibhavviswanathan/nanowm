"""Preprocess TartanDrive raw data into the layout expected by the DataSource.

TartanDrive 1.0 ships per-trajectory data; the exact on-disk format depends on
which release you downloaded. The two common cases are:

  (a) Per-trajectory PyTorch files: <traj>/data.pt with a dict containing
      keys like 'image_left', 'cmd', 'odom', etc.
  (b) Per-trajectory numpy/HDF5 files with similar keys.

If your download differs, adapt `_load_raw_trajectory` below — that's the only
function with format-specific logic.

Output layout (consumed by `tartandrive.py` DataSource):

    out_dir/
      train/
        traj_0001/
          frames.npy        # uint8 [T, H, W, 3]
          actions.npy       # float32 [T, 2] -> (throttle, steer)
          meta.json         # {length, source_traj_id, fps}
      val/
        ...
      stats.json            # action min/max/mean/std across train set

Usage:
    python preprocess_tartandrive.py \\
        --raw_dir /path/to/tartandrive_raw \\
        --out_dir $DATASET_DIR/tartandrive \\
        --resolution 256 \\
        --val_fraction 0.1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


# ---------- format-specific loader (ADAPT THIS TO YOUR DOWNLOAD) -------------

def _load_raw_trajectory(traj_dir: Path):
    """Load one trajectory from disk.

    Returns:
        frames: uint8 ndarray [T, H, W, 3] in the original resolution.
        actions: float32 ndarray [T, 2] -> (throttle, steer).
        fps: float, source frame rate (best effort; default 10.0).

    Returns None if the trajectory can't be loaded (skipped silently).
    """
    # --- Case (a): a single data.pt with everything in it ---
    pt_path = traj_dir / "data.pt"
    if pt_path.exists():
        import torch

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        # TODO: TartanDrive's exact key names vary across releases. Common ones:
        #   images: 'image_left', 'image_left_color', 'rgb_left', 'image'
        #   actions: 'cmd', 'intervention', 'control', 'action'
        # Adjust the keys below to match your download.
        img_key = next(
            (k for k in ("image_left", "image_left_color", "rgb_left", "image") if k in data),
            None,
        )
        cmd_key = next(
            (k for k in ("cmd", "intervention", "control", "action") if k in data),
            None,
        )
        if img_key is None or cmd_key is None:
            print(f"  skipping {traj_dir.name}: keys not found "
                  f"(have {list(data.keys())[:8]}...)")
            return None
        frames = np.asarray(data[img_key])
        actions = np.asarray(data[cmd_key])

    # --- Case (b): separate frames/ folder + actions.npy ---
    elif (traj_dir / "actions.npy").exists() and (traj_dir / "frames").exists():
        actions = np.load(traj_dir / "actions.npy")
        frame_paths = sorted((traj_dir / "frames").glob("*.png"))
        frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in frame_paths])

    else:
        return None

    # Normalize shapes / dtypes
    if frames.ndim != 4:
        return None
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    actions = actions.astype(np.float32)

    # TartanDrive actions are typically [throttle, steer]. If the cmd vector
    # has extra dims (e.g. brake), keep only the first two.
    if actions.shape[-1] > 2:
        actions = actions[..., :2]
    if actions.shape[-1] != 2:
        return None

    # Align lengths if they differ slightly (cameras and CAN run at different rates).
    T = min(len(frames), len(actions))
    frames, actions = frames[:T], actions[:T]
    if T < 16:  # too short to slice usefully
        return None

    return frames, actions, 10.0  # 10 fps is the documented TartanDrive cam rate


# ---------- preprocessing pipeline -------------------------------------------

def _resize_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a [T, H, W, 3] uint8 array to [T, R, R, 3]."""
    out = np.empty((len(frames), resolution, resolution, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(
            Image.fromarray(f).resize((resolution, resolution), Image.BILINEAR)
        )
    return out


def process_split(
    traj_dirs: list,
    out_split_dir: Path,
    resolution: int,
) -> tuple:
    """Process one split (train or val). Returns (n_kept, action_array_for_stats)."""
    out_split_dir.mkdir(parents=True, exist_ok=True)
    all_actions = []
    n_kept = 0

    for idx, traj_dir in enumerate(traj_dirs):
        print(f"  [{idx + 1}/{len(traj_dirs)}] {traj_dir.name}", flush=True)
        loaded = _load_raw_trajectory(traj_dir)
        if loaded is None:
            print(f"    skipped")
            continue
        frames, actions, fps = loaded

        frames = _resize_frames(frames, resolution)

        out_traj_dir = out_split_dir / f"traj_{n_kept:04d}"
        out_traj_dir.mkdir(exist_ok=True)
        np.save(out_traj_dir / "frames.npy", frames)
        np.save(out_traj_dir / "actions.npy", actions)
        with open(out_traj_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "length": int(len(frames)),
                    "source_traj_id": traj_dir.name,
                    "fps": float(fps),
                },
                f,
                indent=2,
            )

        all_actions.append(actions)
        n_kept += 1

    return n_kept, all_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Directory containing raw TartanDrive trajectory subdirs.")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Where to write the processed dataset.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Discover trajectory directories. TartanDrive typically nests one level
    # deep (e.g. raw_dir/traj_0001/data.pt). Adjust if yours is different.
    traj_dirs = sorted([p for p in args.raw_dir.iterdir() if p.is_dir()])
    if not traj_dirs:
        raise SystemExit(f"No trajectory directories found in {args.raw_dir}")

    print(f"Found {len(traj_dirs)} candidate trajectory directories.")

    # Train/val split — random across trajectories. If you have a way to split
    # spatially (e.g. by GPS region), prefer that to reduce leakage.
    random.shuffle(traj_dirs)
    n_val = max(1, int(len(traj_dirs) * args.val_fraction))
    val_dirs = traj_dirs[:n_val]
    train_dirs = traj_dirs[n_val:]
    print(f"Split: {len(train_dirs)} train, {len(val_dirs)} val")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing train split...")
    n_train, train_actions = process_split(
        train_dirs, args.out_dir / "train", args.resolution
    )

    print("\nProcessing val split...")
    n_val_kept, _ = process_split(
        val_dirs, args.out_dir / "val", args.resolution
    )

    # Compute action stats from train set (sanity check for normalization).
    if train_actions:
        all_actions = np.concatenate(train_actions, axis=0)
        stats = {
            "n_train_trajectories": int(n_train),
            "n_val_trajectories": int(n_val_kept),
            "n_train_frames": int(len(all_actions)),
            "action_dim": 2,
            "action_min": all_actions.min(axis=0).tolist(),
            "action_max": all_actions.max(axis=0).tolist(),
            "action_mean": all_actions.mean(axis=0).tolist(),
            "action_std": all_actions.std(axis=0).tolist(),
        }
        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("\nAction stats:")
        print(json.dumps(stats, indent=2))

    print(f"\nDone. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
