"""Pre-encode every frame in the preprocessed TartanDrive dataset to VAE latents.

Single biggest speedup on a 16 GB laptop GPU: pulls the VAE forward out of the
training loop, so training only runs the diffusion transformer.

VAE choice mirrors upstream training (see docs/phase0_findings.md):
    stabilityai/sd-vae-ft-mse, scaling_factor=0.18215.

We use `.mode()` (latent distribution mean) for deterministic caching — easier
to verify, and the bias vs training's `.sample()` is small for this VAE. We
multiply by `vae.config.scaling_factor` so the cached values match what
`encode_first_stage()` produces during training.

Output (mirrors the input layout):

    out_dir/
      train/traj_0001/latents.npy        # float32 [T, C', H', W']  (scaled)
      train/traj_0001/latents.meta.json  # {src_sha256, vae_id, scaling_factor, ...}
      val/...

Resumable: existing `latents.npy` files are skipped. `latents.meta.json` carries
the source-frames SHA256, so if you re-preprocess and frames change, stale
latents are detected by --verify.

Usage:
    DATASET_DIR=~/data/nanowm uv run python scripts/precompute_latents.py \
        --data_dir $DATASET_DIR/tartandrive \
        --out_dir  $DATASET_DIR/tartandrive_latents \
        --batch_size 16

Verify ~1% of cached latents are bit-equivalent to a fresh re-encode:
    uv run python scripts/precompute_latents.py \
        --data_dir $DATASET_DIR/tartandrive \
        --out_dir  $DATASET_DIR/tartandrive_latents \
        --verify
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm


DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


def sha256_file(path: Path, chunk_mb: int = 16) -> str:
    """Streaming SHA-256 of a file. Used to key cache validity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_mb * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_vae(vae_id: str = DEFAULT_VAE, device: str = "cuda") -> AutoencoderKL:
    """Load the same VAE training will use.

    Tries `subfolder='vae'` (SD pipeline repos) first; falls back to no
    subfolder (standalone VAE repos like sd-vae-ft-mse). Matches the patched
    behavior in upstream train_experiment.py.
    """
    try:
        vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
    except (OSError, ValueError):
        vae = AutoencoderKL.from_pretrained(vae_id)
    return vae.to(device).eval()


@torch.no_grad()
def encode_frames_to_latents(
    vae: AutoencoderKL,
    frames_uint8: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """uint8 [T, H, W, 3] -> float32 [T, C', H', W'] latents (scaled).

    Pipeline: uint8 [0, 255] -> float [-1, 1] -> vae.encode -> .mode() -> * scale.
    The scaling matches what `utils.vae_ops.encode_first_stage` does in training.
    """
    x = torch.from_numpy(frames_uint8).to(device).permute(0, 3, 1, 2)
    x = x.float() / 127.5 - 1.0  # [T, 3, H, W] in [-1, 1]

    scale = float(vae.config.scaling_factor)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        chunk = x[i : i + batch_size]
        z = vae.encode(chunk).latent_dist.mode()
        z = z.mul_(scale)
        chunks.append(z.cpu().float())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def encode_one_trajectory(
    vae: AutoencoderKL,
    src_frames: Path,
    out_latents: Path,
    out_meta: Path,
    vae_id: str,
    batch_size: int,
    device: str,
) -> bool:
    """Encode one trajectory; write latents + sidecar. Returns True if encoded."""
    if out_latents.exists() and out_meta.exists():
        return False  # already cached

    frames = np.load(src_frames)
    latents = encode_frames_to_latents(vae, frames, batch_size, device)

    out_latents.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_latents, latents)

    meta = {
        "src_frames": str(src_frames),
        "src_sha256": sha256_file(src_frames),
        "vae_id": vae_id,
        "scaling_factor": float(vae.config.scaling_factor),
        "n_frames": int(latents.shape[0]),
        "latent_shape": list(latents.shape[1:]),
        "encoded_mode": "mode",  # not .sample(); deterministic cache
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    return True


def verify(
    vae: AutoencoderKL,
    data_dir: Path,
    out_dir: Path,
    splits: list,
    sample_fraction: float,
    batch_size: int,
    device: str,
    seed: int,
) -> int:
    """Re-encode a `sample_fraction` of trajectories and check bit-equivalence.

    Returns 0 if all checked latents match, 1 otherwise.
    """
    rng = random.Random(seed)
    failures = 0
    for split in splits:
        traj_dirs = sorted((data_dir / split).iterdir()) if (data_dir / split).exists() else []
        n_check = max(1, int(round(len(traj_dirs) * sample_fraction)))
        chosen = rng.sample(traj_dirs, min(n_check, len(traj_dirs)))
        print(f"[verify] {split}: re-encoding {len(chosen)}/{len(traj_dirs)} trajectories")
        for traj in tqdm(chosen):
            cached = out_dir / split / traj.name / "latents.npy"
            meta_path = out_dir / split / traj.name / "latents.meta.json"
            if not cached.exists() or not meta_path.exists():
                print(f"  MISSING cached latents for {traj.name}")
                failures += 1
                continue
            frames = np.load(traj / "frames.npy")
            re_encoded = encode_frames_to_latents(vae, frames, batch_size, device)
            cached_arr = np.load(cached)
            if not np.array_equal(re_encoded, cached_arr):
                # diffs in float math can show up under bf16 autocast; we use fp32 here
                max_abs = float(np.max(np.abs(re_encoded - cached_arr)))
                print(f"  MISMATCH {traj.name}: max_abs_diff={max_abs:.6f}")
                failures += 1
            # Cross-check: src SHA matches the meta
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("src_sha256") != sha256_file(traj / "frames.npy"):
                print(f"  STALE meta {traj.name}: src_sha256 mismatch")
                failures += 1
    if failures == 0:
        print("[verify] OK — all sampled latents bit-equivalent and meta SHAs match")
    else:
        print(f"[verify] {failures} failure(s)")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vae_id", default=DEFAULT_VAE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-encode --verify_fraction of trajectories and check bit-equivalence; do not write new latents.",
    )
    parser.add_argument("--verify_fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"data_dir does not exist: {args.data_dir}", file=sys.stderr)
        return 2

    print(f"Loading VAE: {args.vae_id} on {args.device}")
    vae = load_vae(args.vae_id, device=args.device)
    print(f"  scaling_factor={vae.config.scaling_factor}")

    if args.verify:
        return verify(
            vae, args.data_dir, args.out_dir, args.splits,
            sample_fraction=args.verify_fraction,
            batch_size=args.batch_size, device=args.device, seed=args.seed,
        )

    n_encoded = 0
    n_skipped = 0
    for split in args.splits:
        split_dir = args.data_dir / split
        if not split_dir.exists():
            print(f"[{split}] missing, skipping")
            continue
        out_split = args.out_dir / split
        traj_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        print(f"\n[{split}] encoding {len(traj_dirs)} trajectories -> {out_split}")
        for traj in tqdm(traj_dirs):
            wrote = encode_one_trajectory(
                vae,
                src_frames=traj / "frames.npy",
                out_latents=out_split / traj.name / "latents.npy",
                out_meta=out_split / traj.name / "latents.meta.json",
                vae_id=args.vae_id,
                batch_size=args.batch_size,
                device=args.device,
            )
            if wrote:
                n_encoded += 1
            else:
                n_skipped += 1
    print(f"\nDone. encoded={n_encoded}, skipped(existing)={n_skipped}")
    print("Next: set `dataset.loader.use_cached_latents=true` and re-run the smoke test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
