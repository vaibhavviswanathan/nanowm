"""Compute combined per-joint action normalization stats for the SO-101 mix.

Run once before pretrain. Writes a JSON file in the format
SO101DataSource.stats_path expects:

    {
      "action_mean": [6 floats],
      "action_std":  [6 floats],
      "n_samples":   int,
      "sources":     [list of repo_ids it covered]
    }

Why combined: per-source `meta/stats.json` are computed against each source's
own action distribution. Reusing those stats on a mix misnormalizes — the SO-100
v2.1 datasets and SO-101 v3.0 dataset have systematically different joint
midpoints because they were calibrated to different physical arms.

Usage:
    uv run python scripts/compute_so101_stats.py \\
        --output $RESULTS_DIR/so101_combined_stats.json \\
        --stride 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# Import the maps from the DataSource so the canonical joint order stays in
# one place. We don't construct the DataSource itself (it would load video
# decoders we don't need); we just iterate over per-source LeRobotDatasets.
from wm_datasets.data_source.manipulation.so101 import (
    ACTION_KEY_MAPS,
    PRETRAIN_SOURCES,
)


def _fetch_action_from_hf(hf, i, action_keys):
    """Pull a [6]-vec action from the raw HF dataset row.

    Going via `ds[i]` triggers video decode for every frame, which is
    pointless when we only want actions. The underlying `ds.hf_dataset` row
    has the action column directly.
    """
    row = hf[i]
    packed = row.get("action")
    if packed is not None:
        return torch.as_tensor(packed, dtype=torch.float64)
    # Older v2.1 sources may store per-joint columns instead.
    return torch.stack([torch.as_tensor(row[k], dtype=torch.float64) for k in action_keys], dim=-1)


def compute_stats(repo_ids, sample_stride: int = 5, root=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Pass 1: running mean.
    total_sum = torch.zeros(6, dtype=torch.float64)
    total_count = 0
    print("[stats] Pass 1: mean")
    for repo_id in repo_ids:
        # video_backend doesn't matter — we use hf_dataset which skips videos.
        ds = LeRobotDataset(repo_id, root=root, image_transforms=None, video_backend="pyav")
        keys = ACTION_KEY_MAPS[repo_id]
        hf = ds.hf_dataset
        for i in tqdm(range(0, len(hf), sample_stride), desc=repo_id):
            total_sum += _fetch_action_from_hf(hf, i, keys)
            total_count += 1
    if total_count == 0:
        raise RuntimeError("No samples seen — empty datasets?")
    mean = (total_sum / total_count).float()

    # Pass 2: running variance using the just-computed mean.
    total_sqdiff = torch.zeros(6, dtype=torch.float64)
    total_count2 = 0
    print("[stats] Pass 2: variance")
    for repo_id in repo_ids:
        ds = LeRobotDataset(repo_id, root=root, image_transforms=None, video_backend="pyav")
        keys = ACTION_KEY_MAPS[repo_id]
        hf = ds.hf_dataset
        for i in tqdm(range(0, len(hf), sample_stride), desc=repo_id):
            total_sqdiff += (_fetch_action_from_hf(hf, i, keys) - mean.double()) ** 2
            total_count2 += 1
    std = torch.sqrt(total_sqdiff / total_count2).float()
    std = std.clamp(min=1e-3)  # constant-joint guard

    return {
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "n_samples": total_count,
        "sources": list(repo_ids),
        "stride": sample_stride,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", default="so101_combined_stats.json",
        help="Path to write the JSON. Pass this to dataset.loader.stats_path at train time.",
    )
    p.add_argument(
        "--stride", type=int, default=5,
        help="Visit every Nth frame; 5 ≈ 20%% coverage, plenty for the mean/std signal.",
    )
    p.add_argument(
        "--sources", nargs="+", default=list(PRETRAIN_SOURCES),
        help="LeRobot repo_ids to include. Default = pretrain mix.",
    )
    p.add_argument("--root", default=None, help="LeRobot dataset cache root.")
    args = p.parse_args()

    stats = compute_stats(args.sources, sample_stride=args.stride, root=args.root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"[stats] Wrote {out}")
    print(f"  mean: {stats['action_mean']}")
    print(f"  std:  {stats['action_std']}")
    print(f"  n_samples: {stats['n_samples']}")


if __name__ == "__main__":
    main()
