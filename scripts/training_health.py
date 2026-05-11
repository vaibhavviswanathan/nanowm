"""Print a one-shot health summary of a training run.

Designed for AM/PM checkins during Phase 5 (full training). Reports:
  - Most-recent step + loss
  - Loss slope over the last `--window` steps
  - Time per step over the last `--window` steps
  - ETA to `--target_steps`
  - Non-finite loss count
  - Time since last checkpoint

Usage:
    uv run python scripts/training_health.py $RESULTS_DIR/<run> [--target_steps 30000]
"""

import argparse
import math
import sys
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_loss(run_dir: Path):
    """Return list of (step, wall_time, value) for train loss."""
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        tb_dir = run_dir
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    for cand in ("train/loss", "train_loss", "loss/train", "loss"):
        if cand in tags:
            return [(s.step, s.wall_time, s.value) for s in ea.Scalars(cand)]
    print(f"WARN: no loss tag found in {tags[:10]}", file=sys.stderr)
    return []


def _slope_per_1k_steps(xs, ys):
    """Simple linear regression slope, per 1000 steps. Returns 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=1000,
                        help="Trailing-step window for slope and step-rate.")
    parser.add_argument("--target_steps", type=int, default=30_000)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run_dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2

    series = _load_loss(args.run_dir)
    if not series:
        print("No loss series yet.")
        return 0

    last_step, last_wall, last_val = series[-1]
    n_nan = sum(1 for (_, _, v) in series if not math.isfinite(v))

    # Tail window
    tail = [(s, w, v) for (s, w, v) in series if s >= last_step - args.window]
    if len(tail) >= 2:
        slope_per_1k = _slope_per_1k_steps([s for (s, _, _) in tail], [v for (_, _, v) in tail])
        steps_in_tail = tail[-1][0] - tail[0][0]
        secs_in_tail = tail[-1][1] - tail[0][1]
        step_rate = steps_in_tail / secs_in_tail if secs_in_tail > 0 else 0.0
        eta_h = (args.target_steps - last_step) / step_rate / 3600 if step_rate > 0 else float("inf")
    else:
        slope_per_1k = 0.0
        step_rate = 0.0
        eta_h = float("inf")

    # Checkpoints
    ckpt_dir = args.run_dir / "checkpoints"
    last_ckpt = None
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            last_ckpt = ckpts[0]

    print(f"Run: {args.run_dir}")
    print(f"  step = {last_step:>7}   loss = {last_val:.4f}")
    print(f"  window = last {args.window} steps")
    print(f"    slope = {slope_per_1k:+.4f} per 1k steps  (negative = improving)")
    print(f"    rate  = {step_rate:.2f} steps/s")
    print(f"    eta to {args.target_steps}: {eta_h:.1f} h  ({eta_h/24:.2f} d)")
    print(f"  non-finite losses: {n_nan}")
    if last_ckpt:
        age_min = (time.time() - last_ckpt.stat().st_mtime) / 60.0
        print(f"  last ckpt: {last_ckpt.name} ({age_min:.1f} min ago)")
    else:
        print("  last ckpt: none yet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
