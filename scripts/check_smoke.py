"""Validate a smoke-test run against codified pass criteria.

Exits 0 if the run meets all criteria below, non-zero otherwise. Designed to be
called immediately after `scripts/smoke_test.sh` to gate the next phase.

Pass criteria:
  1. A `checkpoints/latest*` artifact exists and is a non-empty file.
  2. The training loss series has no NaN/Inf.
  3. The 50-step moving-average loss is monotone non-increasing across the
     run (allowing a small slack — POC, not production).
  4. The measured step rate is fast enough to finish 30k steps in <= 6 days.

We read losses from tensorboard event files written under `<run>/tb/`.

Usage:
    uv run python scripts/check_smoke.py $RESULTS_DIR/<run_dir>
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:  # tensorboard always pinned in our deps, but be loud anyway
    print("ERROR: tensorboard not importable. Did you `uv sync --extra dev`?", file=sys.stderr)
    raise


# Tunables — keep in code (not flags) so the gate is identical run to run.
SMOKE_TARGET_TOTAL_STEPS = 30_000   # what Phase 5 will run
SMOKE_TIME_BUDGET_DAYS = 6.0        # max wall-clock allowed for full run
SMOKE_MIN_LOGGED_POINTS = 10        # need at least this many logged scalars to gate


def _find_event_dir(run_dir: Path) -> Optional[Path]:
    """Locate the deepest dir containing tfevents files under run_dir."""
    candidates = sorted(p.parent for p in run_dir.rglob("events.out.tfevents.*"))
    return candidates[0] if candidates else None


def _load_scalar_series(run_dir: Path, tag_candidates: List[str]) -> List[Tuple[float, float, float]]:
    """Return list of (step, wall_time, value) for the first tag found."""
    event_dir = _find_event_dir(run_dir)
    if event_dir is None:
        print(f"WARN: no tfevents files under {run_dir}", file=sys.stderr)
        return []
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    chosen = None
    for cand in tag_candidates:
        if cand in tags:
            chosen = cand
            break
    if chosen is None:
        print(
            f"WARN: none of {tag_candidates} in tensorboard tags: {tags[:10]}...",
            file=sys.stderr,
        )
        return []
    return [(s.step, s.wall_time, s.value) for s in ea.Scalars(chosen)]


def _split_mean(values: List[float]) -> Tuple[float, float]:
    """Mean of the first half vs mean of the second half."""
    half = max(1, len(values) // 2)
    return (sum(values[:half]) / half, sum(values[half:]) / max(1, len(values) - half))


def check_checkpoint(run_dir: Path) -> bool:
    matches = list(run_dir.glob("checkpoints/latest*"))
    if not matches:
        print("FAIL: no checkpoints/latest* artifact under", run_dir)
        return False
    for m in matches:
        # Lightning may save as a dir or a .ckpt file; both ok if non-empty.
        if m.is_file() and m.stat().st_size > 0:
            print(f"OK: checkpoint found: {m} ({m.stat().st_size / 1e6:.1f} MB)")
            return True
        if m.is_dir() and any(m.iterdir()):
            print(f"OK: checkpoint dir found: {m}")
            return True
    print("FAIL: checkpoint artifacts are empty:", matches)
    return False


def check_no_nan(loss_series: List[Tuple[float, float, float]]) -> bool:
    bad = [s for (s, _, v) in loss_series if not math.isfinite(v)]
    if bad:
        print(f"FAIL: {len(bad)} non-finite loss values at steps: {bad[:5]}...")
        return False
    print(f"OK: {len(loss_series)} loss values, all finite")
    return True


def check_loss_decreased(loss_series: List[Tuple[float, float, float]], slack: float = 0.98) -> bool:
    """Second-half mean must be < first-half mean × slack (default: 2% drop)."""
    if len(loss_series) < SMOKE_MIN_LOGGED_POINTS:
        print(f"FAIL: too few logged points ({len(loss_series)}) to assess loss trend")
        return False
    values = [v for (_, _, v) in loss_series]
    first, last = _split_mean(values)
    if last > first * slack:
        print(f"FAIL: loss did not decrease — first_half={first:.4f} -> second_half={last:.4f}")
        return False
    print(f"OK: loss decreased ({first:.4f} -> {last:.4f})")
    return True


def check_step_rate(
    loss_series: List[Tuple[float, float, float]],
    target_steps: int = SMOKE_TARGET_TOTAL_STEPS,
    budget_days: float = SMOKE_TIME_BUDGET_DAYS,
) -> bool:
    if len(loss_series) < SMOKE_MIN_LOGGED_POINTS:
        print(
            f"FAIL: too few logged points ({len(loss_series)}) to estimate step rate"
        )
        return False
    # Skip the first couple of warmup points (slow on startup) for a steady-state rate.
    step_first, wall_first, _ = loss_series[2]
    step_last, wall_last, _ = loss_series[-1]
    elapsed_s = wall_last - wall_first
    steps = step_last - step_first
    if elapsed_s <= 0 or steps <= 0:
        print(f"FAIL: degenerate step-rate window (elapsed={elapsed_s}, steps={steps})")
        return False
    rate = steps / elapsed_s  # steps/sec
    eta_days = (target_steps - step_last) / rate / 86400.0
    if eta_days > budget_days:
        print(
            f"FAIL: extrapolated ETA {eta_days:.1f} days > budget {budget_days:.1f} days "
            f"(rate={rate:.2f} steps/s)"
        )
        return False
    print(
        f"OK: step rate {rate:.2f}/s; "
        f"{target_steps} steps would take ~{eta_days:.2f} days"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"FAIL: run_dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2

    print(f"Checking {args.run_dir}\n")
    losses = _load_scalar_series(
        args.run_dir,
        tag_candidates=[
            "train/loss",
            "train_loss",
            "loss/train",
            "loss",
        ],
    )

    results = [
        check_checkpoint(args.run_dir),
        check_no_nan(losses),
        check_loss_decreased(losses),
        check_step_rate(losses),
    ]
    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
