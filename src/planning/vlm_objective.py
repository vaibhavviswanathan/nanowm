"""VLM-scored CEM objective for SO-101 manipulation.

Plugs into upstream's `planning.cem_planner.CEMPlanner` as an `objective_fn`.
Decodes the *final* imagined latent of each candidate rollout to a pixel
frame, asks a Vision Language Model to score "did the robot complete the
task?" on [0, 1], and returns `1 - score` so CEM minimizes it.

Why only the final frame: per `so101-train/PLAN.md` the reward signal is
"VLM scoring of final imagined frame". Scoring every intermediate frame
would multiply latency by the rollout horizon; the final-frame signal is
the policy gradient.

Latency: each CEM iteration scores `num_samples` (64 by default) frames.
The VLM is the bottleneck (~0.5-2 s/image serially). We run the calls
through `asyncio.gather` so they overlap; with `claude-haiku-4-5` at
~1 s/call and the async client we get ~5-10 s per CEM iteration end-to-end
on a warm pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Callable, Dict, List, Optional

import torch
from einops import rearrange


def _frame_to_png_b64(frame: torch.Tensor) -> str:
    """Convert a [-1, 1] float tensor [3, H, W] to base64-encoded PNG."""
    from PIL import Image

    img = ((frame.clamp(-1, 1) + 1.0) / 2.0 * 255.0).byte()
    img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class AnthropicVLMScorer:
    """Async VLM scorer backed by the Anthropic SDK.

    Args:
        task: short natural-language task description, e.g. "pick up the red block".
        model: Anthropic model id. PLAN.md recommends claude-haiku-4-5 for
               latency-sensitive scoring.
        api_key: optional override; otherwise reads ANTHROPIC_API_KEY.
        timeout_s: per-call timeout. Failures are scored 0.0 (no progress).
    """

    REWARD_PROMPT = (
        'Did this robot arm complete the task: "{task}"? '
        "Reply with a single number between 0 and 1, where 1 means complete "
        "success and 0 means no progress. Output ONLY the number, no words."
    )

    def __init__(
        self,
        task: str,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        timeout_s: float = 10.0,
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK required: `uv add anthropic` or `pip install anthropic`"
            ) from e
        self._client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            timeout=timeout_s,
        )
        self.task = task
        self.model = model
        self.prompt = self.REWARD_PROMPT.format(task=task)

    async def _score_one(self, image_b64: str) -> float:
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        }},
                        {"type": "text", "text": self.prompt},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            return max(0.0, min(1.0, float(text)))
        except Exception as e:  # API errors, parse errors, timeouts
            print(f"[vlm] scoring failed ({type(e).__name__}: {e}); using 0.0")
            return 0.0

    async def score_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [N, 3, H, W] in [-1, 1]. Returns [N] in [0, 1]."""
        b64s = [_frame_to_png_b64(f) for f in frames]
        scores = await asyncio.gather(*(self._score_one(b) for b in b64s))
        return torch.tensor(scores, dtype=torch.float32, device=frames.device)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return asyncio.run(self.score_batch(frames))


class VLMObjective:
    """Objective callable for CEMPlanner.plan(objective_fn=...).

    Signature (from CEMPlanner.plan):
        objective_fn(z_obses: Dict, z_obs_g: Dict) -> Tensor[num_samples]

    We ignore `z_obs_g` (task is a string the scorer already has) and pull
    the final-frame latent from z_obses['visual'] (flat).
    """

    def __init__(
        self,
        vae,
        scorer: Callable[[torch.Tensor], torch.Tensor],
        latent_channels: int = 4,
        latent_hw: int = 32,
    ):
        self.vae = vae
        self.scorer = scorer
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw

    @torch.no_grad()
    def __call__(self, z_obses: Dict[str, torch.Tensor], z_obs_g) -> torch.Tensor:
        # z_obses['visual']: [N, T, C_lat * H_lat * W_lat], flat per DiffusionWorldModel.rollout.
        flat = z_obses["visual"]
        N, T, D = flat.shape
        C, H_l, W_l = self.latent_channels, self.latent_hw, self.latent_hw
        if C * H_l * W_l != D:
            # Fall back to inferring from D, assuming square latents.
            C = self.latent_channels
            HW = D // C
            side = int(HW ** 0.5)
            H_l = W_l = side
        latents = flat.view(N, T, C, H_l, H_l)
        final_latent = latents[:, -1] / self.vae.config.scaling_factor  # [N, C, H_l, W_l]
        decoded = self.vae.decode(final_latent).sample  # [N, 3, H, W] in roughly [-1, 1]
        rewards = self.scorer(decoded)  # [N] in [0, 1]
        return 1.0 - rewards  # CEMPlanner minimizes — flip the sign


class ConstantScorer:
    """Deterministic stand-in for the VLM scorer.

    Useful for smoke-testing the planner wiring without an API key. Returns
    `value` for every input frame; CEMPlanner's variance update is no-op
    under a constant objective, so a successful run with this just exercises
    the rollout + VAE-decode path.
    """

    def __init__(self, value: float = 0.5):
        self.value = float(value)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.full((frames.shape[0],), self.value, dtype=torch.float32, device=frames.device)
