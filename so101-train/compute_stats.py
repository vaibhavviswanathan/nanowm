"""
Compute combined-mix action normalization stats.

Run ONCE before training. Output is a small JSON used by SO101WorldModelDataset.

Usage:
  python compute_stats.py --output combined_stats.json
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset_wrapper import (
    ACTION_KEY_MAPS,
    PRETRAIN_SOURCES,
    LeRobotDataset,
)


def compute_stats(repo_ids, sample_stride: int = 5):
    """
    Streaming-friendly two-pass mean/std computation over the unified action space.
    sample_stride lets you subsample to keep this fast (every Nth frame).
    """
    # Pass 1: mean
    total_sum = torch.zeros(6, dtype=torch.float64)
    total_count = 0

    print("[stats] Pass 1: mean")
    for repo_id in repo_ids:
        ds = LeRobotDataset(repo_id)
        keys = ACTION_KEY_MAPS[repo_id]
        for i in tqdm(range(0, len(ds), sample_stride), desc=repo_id):
            sample = ds[i]
            if "action" in sample and isinstance(sample["action"], torch.Tensor) \
                    and sample["action"].numel() == 6:
                a = sample["action"].double()
            else:
                a = torch.stack([sample[k].double() for k in keys], dim=-1)
            total_sum += a
            total_count += 1

    mean = (total_sum / total_count).float()

    # Pass 2: variance
    total_sqdiff = torch.zeros(6, dtype=torch.float64)
    total_count2 = 0

    print("[stats] Pass 2: variance")
    for repo_id in repo_ids:
        ds = LeRobotDataset(repo_id)
        keys = ACTION_KEY_MAPS[repo_id]
        for i in tqdm(range(0, len(ds), sample_stride), desc=repo_id):
            sample = ds[i]
            if "action" in sample and isinstance(sample["action"], torch.Tensor) \
                    and sample["action"].numel() == 6:
                a = sample["action"].double()
            else:
                a = torch.stack([sample[k].double() for k in keys], dim=-1)
            total_sqdiff += (a - mean.double()) ** 2
            total_count2 += 1

    std = torch.sqrt(total_sqdiff / total_count2).float()
    std = std.clamp(min=1e-3)  # avoid div-by-zero on constant dims

    return {
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "n_samples": total_count,
        "sources": list(repo_ids),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="combined_stats.json")
    p.add_argument("--stride", type=int, default=5,
                   help="Sample every Nth frame (default 5 = 20%% of frames).")
    p.add_argument("--sources", nargs="+", default=PRETRAIN_SOURCES)
    args = p.parse_args()

    stats = compute_stats(args.sources, sample_stride=args.stride)
    Path(args.output).write_text(json.dumps(stats, indent=2))
    print(f"[stats] Wrote {args.output}")
    print(f"  mean: {stats['action_mean']}")
    print(f"  std:  {stats['action_std']}")
    print(f"  n_samples: {stats['n_samples']}")


if __name__ == "__main__":
    main()
