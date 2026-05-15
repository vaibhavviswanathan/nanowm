"""CEM planning over imagined SO-101 rollouts, scored by a VLM.

Loads a trained NanoWM checkpoint + VAE, takes a single starting image
(optionally from a held-out val episode), runs the upstream CEMPlanner with
our VLM-scored objective, and prints/saves the planned action sequence.

This script implements the imagined-rollout half of the PLAN.md pipeline.
Actually driving an SO-101 follower lives in a separate `--robot` flag
(left as a TODO — needs the SO-101 SDK calibrated against your specific
arm, which is hardware-side work).

Typical usage:

    # Smoke check (no VLM, constant reward — exercises the rollout path):
    uv run python scripts/cem_plan_so101.py \\
        --checkpoint $RESULTS_DIR/<run>/checkpoints/latest/<latest>.ckpt \\
        --config $RESULTS_DIR/<run>/.hydra/config.yaml \\
        --task "pick up the red cube" \\
        --start_image ./start.png \\
        --scorer constant

    # Real plan with Claude Haiku scoring:
    ANTHROPIC_API_KEY=... uv run python scripts/cem_plan_so101.py \\
        --checkpoint ... --config ... \\
        --task "pick up the red cube" \\
        --start_image ./start.png \\
        --scorer anthropic --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Wire upstream's src/ in (the train scripts do this too); the nano-world-model
# tree is symlinked from the manipulation worktree.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_UPSTREAM_SRC = _REPO_ROOT / "nano-world-model" / "src"
sys.path.insert(0, str(_UPSTREAM_SRC))


def _load_start_image(path: str, image_size: int, device: str) -> torch.Tensor:
    """PNG -> [1, 1, 3, H, W] in [-1, 1]. The trailing time dim is 1 — context
    is a single frame at planning time."""
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    t = torch.from_numpy(__import__("numpy").asarray(img)).float() / 255.0
    t = t.permute(2, 0, 1)  # [3, H, W]
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]


def main(args):
    from omegaconf import OmegaConf
    from diffusers.models import AutoencoderKL
    from models import get_models  # type: ignore
    from diffusion import create_diffusion  # type: ignore
    from utils.nanowm_utils import find_model  # type: ignore
    from planning.cem_planner import CEMPlanner  # type: ignore
    from planning.diffusion_world_model import DiffusionWorldModel  # type: ignore

    # vlm_objective lives in OUR src/ (worktree-tracked), not upstream's:
    sys.path.insert(0, str(_REPO_ROOT))
    from src.planning.vlm_objective import (
        AnthropicVLMScorer, ConstantScorer, VLMObjective,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print(f"[plan] Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    image_size = int(cfg.model.image_size)
    latent_size = image_size // 8
    cfg.model.latent_size = latent_size

    print("[plan] Loading model + VAE...")
    model = get_models(cfg).to(device).eval()
    state = find_model(args.checkpoint)
    if "model" in state:
        state = state["model"]
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"[plan] Loaded checkpoint {args.checkpoint}")

    vae_path = cfg.get("vae_model_path", "stabilityai/sd-vae-ft-mse")
    try:
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device).eval()
    except (OSError, ValueError):
        vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()

    diffusion = create_diffusion(
        timestep_respacing=str(args.ddim_steps),
        noise_schedule=cfg.experiment.diffusion.noise_schedule,
        pred_name=cfg.experiment.diffusion.pred_name,
        diffusion_steps=cfg.experiment.diffusion.diffusion_steps,
        snr_gamma=cfg.experiment.diffusion.snr_gamma,
        zero_terminal_snr=cfg.experiment.diffusion.zero_terminal_snr,
    )

    wm = DiffusionWorldModel(model=model, vae=vae, diffusion=diffusion, args=cfg)

    print(f"[plan] Loading start image {args.start_image}")
    start = _load_start_image(args.start_image, image_size, device)
    obs_0 = {"visual": start}
    obs_g = {"visual": None}  # task is a string; goal frame not used

    if args.scorer == "anthropic":
        scorer = AnthropicVLMScorer(task=args.task, model=args.model)
        print(f"[plan] VLM scorer: Anthropic ({args.model})")
    else:
        scorer = ConstantScorer(value=0.5)
        print(f"[plan] VLM scorer: constant (smoke test only — won't improve actions)")

    action_dim = int(cfg.dataset.spec.action_dim) * int(cfg.dataset.frame_interval)
    objective = VLMObjective(
        vae=vae, scorer=scorer,
        latent_channels=4, latent_hw=latent_size,
    )

    cem = CEMPlanner(
        world_model=wm,
        objective_fn=objective,
        action_dim=action_dim,
        horizon=args.horizon,
        num_samples=args.num_samples,
        topk=max(1, int(args.num_samples * args.elite_fraction)),
        opt_steps=args.cem_iters,
        var_scale=args.var_scale,
        device=device,
        # Actions are normalized; clamp ±3 sigmas to stay in-distribution.
        action_low=-3.0,
        action_high=3.0,
    )

    print(
        f"[plan] Running CEM: horizon={args.horizon} samples={args.num_samples} "
        f"iters={args.cem_iters} elite={cem.topk}"
    )
    planned_actions, info = cem.plan(obs_0, obs_g)
    print(f"[plan] Final loss: {info['final_loss']:.4f}")
    print(f"[plan] Loss history: {[f'{x:.4f}' for x in info['losses']]}")

    out = {
        "task": args.task,
        "horizon": args.horizon,
        "scorer": args.scorer,
        "actions": planned_actions.squeeze(0).cpu().tolist(),
        "final_loss": float(info["final_loss"]),
        "loss_history": [float(x) for x in info["losses"]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[plan] Wrote {args.out}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True,
                   help="Resolved Hydra config.yaml from the training run dir.")
    p.add_argument("--start_image", required=True,
                   help="PNG of the robot's current camera view.")
    p.add_argument("--task", required=True,
                   help='Natural-language task, e.g. "pick up the red block".')
    p.add_argument("--out", default="cem_plan.json")
    p.add_argument("--horizon", type=int, default=8,
                   help="Action horizon. PLAN.md: 8-12.")
    p.add_argument("--num_samples", type=int, default=64,
                   help="CEM population size. PLAN.md: 64.")
    p.add_argument("--cem_iters", type=int, default=3,
                   help="CEM optimization iterations. PLAN.md: 3.")
    p.add_argument("--elite_fraction", type=float, default=0.1,
                   help="Top fraction kept each iter. PLAN.md: 0.1.")
    p.add_argument("--var_scale", type=float, default=1.0,
                   help="Initial sigma per action dim (in normalized action units).")
    p.add_argument("--ddim_steps", type=int, default=25,
                   help="DDIM steps for rollout sampling. PLAN.md: 25 (50 if quality bad).")
    p.add_argument("--scorer", choices=("anthropic", "constant"), default="constant",
                   help="`constant` is smoke-only; CEM will not converge with it.")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Anthropic model id when --scorer=anthropic.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
