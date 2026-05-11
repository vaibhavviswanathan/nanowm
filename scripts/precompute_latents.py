"""Pre-encode every frame in the preprocessed dataset to VAE latents.

This is the biggest speedup on a small/laptop GPU because it pulls the VAE
forward pass out of the training loop — training only runs the diffusion
transformer over latents.

Output layout mirrors the input:

    out_dir/
      train/
        traj_0001/
          latents.npy   # float32 [T, C, H', W']
      val/
        ...

Usage:
    python precompute_latents.py \\
        --data_dir $DATASET_DIR/tartandrive \\
        --out_dir $DATASET_DIR/tartandrive_latents \\
        --batch_size 16

# TODO: nano-world-model's VAE is loaded through Hydra config — the exact
# import path depends on the repo. The simplest and most reliable approach is
# to import the VAE-loading helper that the training pipeline itself uses, so
# encoding stays consistent. Two options below; pick whichever your repo
# exposes and delete the other.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_vae(device: str = "cuda"):
    """Load the same VAE the training pipeline uses.

    Replace the body of this function with whatever your repo does. Two common
    patterns:

    Option A — Hydra config-driven:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=str(Path.cwd() / "src/configs")):
            cfg = compose(config_name="config")
        vae = instantiate(cfg.model.vae).to(device).eval()
        return vae

    Option B — diffusers AutoencoderKL (most likely; matches Latte/DFoT):
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device).eval()
        return vae
    """
    from diffusers import AutoencoderKL  # type: ignore

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(device).eval()
    return vae


@torch.no_grad()
def encode_trajectory(vae, frames_uint8: np.ndarray, batch_size: int, device: str):
    """Encode [T, H, W, 3] uint8 frames to [T, C, H', W'] float32 latents."""
    # uint8 [T, H, W, 3] -> float [T, 3, H, W] in [-1, 1]
    x = torch.from_numpy(frames_uint8).to(device)
    x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0

    latents_chunks = []
    for i in range(0, x.shape[0], batch_size):
        chunk = x[i : i + batch_size]
        # diffusers AutoencoderKL convention; use latent_dist.mean for determinism.
        # If your repo's VAE has a different interface, adapt this line.
        latent = vae.encode(chunk).latent_dist.mean
        latents_chunks.append(latent.cpu().float())
    return torch.cat(latents_chunks, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    print(f"Loading VAE on {args.device}...")
    vae = load_vae(args.device)

    for split in args.splits:
        split_dir = args.data_dir / split
        out_split_dir = args.out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        traj_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        print(f"\n[{split}] encoding {len(traj_dirs)} trajectories")

        for traj_dir in tqdm(traj_dirs):
            out_traj_dir = out_split_dir / traj_dir.name
            out_file = out_traj_dir / "latents.npy"
            if out_file.exists():
                continue  # resumable

            frames = np.load(traj_dir / "frames.npy")
            latents = encode_trajectory(vae, frames, args.batch_size, args.device)

            out_traj_dir.mkdir(exist_ok=True)
            np.save(out_file, latents.astype(np.float32))

    print("\nDone. Set use_cached_latents=true in tartandrive.yaml.")


if __name__ == "__main__":
    main()
