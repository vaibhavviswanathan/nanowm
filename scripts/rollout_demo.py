"""Phase 6: produce side-by-side prediction-vs-ground-truth GIFs.

Thin wrapper over upstream `src/sample/rollout.py`:

  1. Invoke upstream's rollout with our config + checkpoint.
  2. Convert the saved comparison MP4s to GIFs (with imageio) for the POC
     artifact.

We don't reimplement the autoregressive flow — it's already in upstream's
`rollout.py` (which calls `dfot_sample` from `src/diffusion/df_sample.py`).
Our value-add is just the artifact format.

Usage:
    bash scripts/rollout_demo.py \
        --config $RESULTS_DIR/<run>/.hydra/config.yaml \
        --checkpoint $RESULTS_DIR/<run>/checkpoints/latest/checkpoint.ckpt \
        --out_dir ./demo \
        --num_samples 6 --rollout_length 50 --history_length 4

The `--config` is the Hydra config that training wrote to its run directory.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "nano-world-model"
UPSTREAM_ROLLOUT = UPSTREAM_DIR / "src" / "sample" / "rollout.py"


def run_upstream_rollout(
    config: Path, checkpoint: Path, save_path: Path,
    num_samples: int, rollout_length: int, history_length: int,
    num_sampling_steps: int, fps: int,
) -> None:
    """Invoke upstream's rollout.py with our args."""
    if not UPSTREAM_ROLLOUT.exists():
        raise FileNotFoundError(
            f"Upstream rollout not found at {UPSTREAM_ROLLOUT}. "
            "Did you clone nano-world-model and run apply_upstream_patches.sh?"
        )

    cmd = [
        sys.executable, str(UPSTREAM_ROLLOUT),
        "--config", str(config),
        "--ckpt", str(checkpoint),
        "--save_path", str(save_path),
        "--num_samples", str(num_samples),
        "--rollout_length", str(rollout_length),
        "--history_length", str(history_length),
        "--num_sampling_steps", str(num_sampling_steps),
        "--fps", str(fps),
    ]
    print("[rollout_demo] invoking:", " ".join(cmd))
    # Run with PYTHONPATH so upstream's `from src.*` imports resolve.
    env = {"PYTHONPATH": str(UPSTREAM_DIR), **dict(__import__("os").environ)}
    subprocess.check_call(cmd, env=env)


def mp4_to_gif(mp4_path: Path, gif_path: Path, fps: int = 10) -> None:
    """Read an MP4 with imageio and write a GIF at the same fps."""
    reader = imageio.get_reader(str(mp4_path))
    frames = [np.asarray(frame) for frame in reader]
    duration = 1.0 / fps
    imageio.mimsave(str(gif_path), frames, duration=duration, loop=0)


def convert_all_mp4s(save_dir: Path, out_dir: Path, fps: int) -> int:
    """Convert every *_compare.mp4 in save_dir to a GIF in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(save_dir.glob("*_compare.mp4"))
    if not mp4s:
        print(f"[rollout_demo] no *_compare.mp4 in {save_dir}")
        return 0
    for mp4 in mp4s:
        gif = out_dir / (mp4.stem + ".gif")
        print(f"  {mp4.name} -> {gif.name}")
        mp4_to_gif(mp4, gif, fps=fps)
    return len(mp4s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Hydra config from the training run (e.g. <run>/.hydra/config.yaml).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Output dir for GIFs.")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--rollout_length", type=int, default=50)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mp4_dir", type=Path, default=None,
                        help="Tmp dir for upstream's MP4 outputs (default: <out_dir>/_mp4).")
    parser.add_argument("--keep_mp4", action="store_true",
                        help="Don't delete the MP4 directory after GIF conversion.")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    mp4_dir = args.mp4_dir or (args.out_dir / "_mp4")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_upstream_rollout(
        config=args.config, checkpoint=args.checkpoint, save_path=mp4_dir,
        num_samples=args.num_samples, rollout_length=args.rollout_length,
        history_length=args.history_length,
        num_sampling_steps=args.num_sampling_steps, fps=args.fps,
    )

    n = convert_all_mp4s(mp4_dir, args.out_dir, fps=args.fps)
    print(f"\nDone. {n} GIFs in {args.out_dir}.")

    if not args.keep_mp4 and mp4_dir.exists():
        shutil.rmtree(mp4_dir)
        print(f"Removed temp MP4 dir: {mp4_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
