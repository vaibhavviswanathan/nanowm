"""Phase 6: generate side-by-side prediction-vs-ground-truth GIFs.

Picks N held-out val trajectories, conditions on the first `context_frames`
real frames, then autoregressively rolls out for `rollout_length` steps using
the trajectory's *recorded* actions. Saves a GIF per episode comparing the
generated rollout against the actual recording.

Usage:
    python rollout_demo.py \\
        --checkpoint $RESULTS_DIR/<run>/checkpoints/latest/checkpoint.ckpt \\
        --data_dir $DATASET_DIR/tartandrive \\
        --out_dir ./demo \\
        --num_episodes 6 \\
        --rollout_length 50 \\
        --context_frames 2

# TODO: nano-world-model exposes its rollout API through a wrapper class —
# the exact name depends on the repo. Check `src/main.py` and the
# `docs/applications/long_rollout.md` doc for the canonical entry point.
# Likely candidates: `WorldModel.rollout(...)`, `inference.rollout_autoregressive(...)`.
# Adapt the `run_rollout` function below to match.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_model(checkpoint_path: str, device: str):
    """Load a trained nano-world-model checkpoint.

    Replace the body of this function with the canonical loading helper from
    your repo. Most likely path:

        from src.models.world_model import WorldModel
        model = WorldModel.load_from_checkpoint(checkpoint_path)
        model = model.to(device).eval()
        return model
    """
    # TODO: adapt to your checkpoint loader
    from src.models.world_model import WorldModel  # type: ignore

    model = WorldModel.load_from_checkpoint(checkpoint_path)
    return model.to(device).eval()


@torch.no_grad()
def run_rollout(
    model,
    context_frames: np.ndarray,   # uint8 [C, H, W, 3]
    actions: np.ndarray,          # float32 [T, 2], T = context + rollout
    num_ddim_steps: int = 50,
    device: str = "cuda",
):
    """Autoregressive rollout. Returns uint8 [T, H, W, 3] generated frames.

    The first `len(context_frames)` are the conditioning frames (echoed back).
    """
    # uint8 [C, H, W, 3] -> [1, C, 3, H, W] in [-1, 1]
    ctx = torch.from_numpy(context_frames).to(device)
    ctx = ctx.permute(0, 3, 1, 2).unsqueeze(0).float() / 127.5 - 1.0

    acts = torch.from_numpy(actions).to(device).unsqueeze(0)  # [1, T, 2]

    # TODO: your repo's rollout entry point. Most likely shape:
    out = model.rollout(
        context_frames=ctx,
        actions=acts,
        num_steps=num_ddim_steps,
        scheduling="sequential",
    )  # expected: [1, T, 3, H, W] in [-1, 1]

    out = out.squeeze(0).clamp(-1, 1)
    out = ((out + 1.0) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


def save_side_by_side_gif(
    pred: np.ndarray,    # uint8 [T, H, W, 3]
    truth: np.ndarray,   # uint8 [T, H, W, 3]
    out_path: Path,
    fps: int = 10,
):
    """Save a side-by-side GIF: pred on the left, ground truth on the right."""
    T = min(len(pred), len(truth))
    frames = []
    for t in range(T):
        # pad ground truth if it's shorter than pred
        gt = truth[t] if t < len(truth) else np.zeros_like(pred[t])
        side = np.concatenate([pred[t], gt], axis=1)  # along width
        frames.append(Image.fromarray(side))

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--rollout_length", type=int, default=50)
    parser.add_argument("--context_frames", type=int, default=2)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(str(args.checkpoint), args.device)

    val_dir = args.data_dir / "val"
    traj_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(
        len(traj_dirs),
        size=min(args.num_episodes, len(traj_dirs)),
        replace=False,
    )

    total_len = args.context_frames + args.rollout_length

    for i in tqdm(chosen):
        traj_dir = traj_dirs[i]
        frames = np.load(traj_dir / "frames.npy")
        actions = np.load(traj_dir / "actions.npy")

        if len(frames) < total_len:
            print(f"  skipping {traj_dir.name} (too short)")
            continue

        # Use the first total_len frames/actions of the trajectory.
        ctx_frames = frames[: args.context_frames]
        rollout_actions = actions[:total_len].astype(np.float32)
        ground_truth = frames[:total_len]

        pred = run_rollout(
            model,
            ctx_frames,
            rollout_actions,
            num_ddim_steps=args.num_ddim_steps,
            device=args.device,
        )

        out_path = args.out_dir / f"{traj_dir.name}_compare.gif"
        save_side_by_side_gif(pred, ground_truth, out_path)
        print(f"  -> {out_path}")

    # Drop a small index file with metadata so the demo is self-describing.
    with open(args.out_dir / "demo_info.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "num_episodes": int(len(chosen)),
                "rollout_length": args.rollout_length,
                "context_frames": args.context_frames,
                "num_ddim_steps": args.num_ddim_steps,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Demo GIFs in {args.out_dir}")


if __name__ == "__main__":
    main()
