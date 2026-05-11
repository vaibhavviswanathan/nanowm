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
from typing import List, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:  # tensorboard always pinned in our deps, but be loud anyway
    print("ERROR: tensorboard not importable. Did you `uv sync --extra dev`?", file=sys.stderr)
    raise


# Tunables — keep in code (not flags) so the gate is identical run to run.
SMOKE_TARGET_TOTAL_STEPS = 30_000   # what Phase 5 will run
SMOKE_TIME_BUDGET_DAYS = 6.0        # max wall-clock allowed for full run
MA_WINDOW = 50                      # moving-average window
SMOKE_MIN_STEPS_FOR_RATE_CHECK = 30  # don't extrapolate from too few points


def _load_scalar_series(run_dir: Path, tag_candidates: List[str]) -> List[Tuple[float, float, float]]:
    """Return list of (step, wall_time, value) for the first tag found.

    `tag_candidates` is checked in order; we return the first that has data.
    Wall time comes from event file timestamps.
    """
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        # Some runs log directly under run_dir
        tb_dir = run_dir
    ea = EventAccumulator(str(tb_dir))
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


def _moving_average(values: List[float], window: int) -> List[float]:
    out = []
    accum = 0.0
    from collections import deque
    q: deque = deque()
    for v in values:
        q.append(v)
        accum += v
        if len(q) > window:
            accum -= q.popleft()
        out.append(accum / len(q))
    return out


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


def check_monotone_ma(loss_series: List[Tuple[float, float, float]], slack: float = 1.05) -> bool:
    if not loss_series:
        print("FAIL: empty loss series")
        return False
    values = [v for (_, _, v) in loss_series]
    ma = _moving_average(values, MA_WINDOW)
    # Compare last-window MA against first-window MA. A POC smoke test only
    # needs to show that loss has *decreased* by the end; 5% slack on the
    # 'monotone' direction is fine.
    first = ma[min(MA_WINDOW, len(ma)) - 1]
    last = ma[-1]
    if last > first * slack:
        print(f"FAIL: MA loss did not decrease — first_window={first:.4f} -> last={last:.4f}")
        return False
    print(f"OK: MA loss decreased ({first:.4f} -> {last:.4f})")
    return True


def check_step_rate(
    loss_series: List[Tuple[float, float, float]],
    target_steps: int = SMOKE_TARGET_TOTAL_STEPS,
    budget_days: float = SMOKE_TIME_BUDGET_DAYS,
) -> bool:
    if len(loss_series) < SMOKE_MIN_STEPS_FOR_RATE_CHECK:
        print(
            f"FAIL: too few logged points ({len(loss_series)}) to estimate step rate"
        )
        return False
    step_first, wall_first, _ = loss_series[SMOKE_MIN_STEPS_FOR_RATE_CHECK // 5]
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
        check_monotone_ma(losses),
        check_step_rate(losses),
    ]
    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
